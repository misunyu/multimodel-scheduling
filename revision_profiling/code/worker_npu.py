"""Workers for the CPU-NPU (Antara) board (Phase N1 section 2-2). All of them use the unchanged Worker.run loop of
worker_base.py: the runtime records dequeue, completion (right after the output put) and failure timestamps, so one
request's service time = preprocess + infer + postprocess below, including every host-side stage and the NPU round trip.

Stage -> process / thread (executor process = E, NPU server process = S):
  NpuResNet50Worker
    preprocess  E worker thread  resize 256 / centre crop 224 / ImageNet mean-std, NCHW float32 (same as the CPU path)
    infer       E worker thread  front partition p0 (QuantizeLinear -> uint8, ORT CPU EP)
                S                SendInput -> Launch -> ReceiveOutputs (npu_server.py); E waits on the pipe
                E worker thread  back partition p2 (ArgMax over the uint8 logits, ORT CPU EP)
    postprocess E worker thread  class id
  NpuYoloV3Worker
    preprocess  E worker thread  letterbox 608 (114 grey), BGR->RGB, /255, NCHW (image_processing.yolo_preprocess_local)
    infer       E worker thread  p0 quantize -> uint8;  S: NPU;  E: dequantize the three uint8 head outputs with the
                                 registry scale / zero point, back partition p2 (decode + top-k -> dets, labels)
    postprocess E worker thread  conf >= 0.5, cv2 NMS (IoU 0.5), de-letterbox and the box sanity filters of
                                 image_processing.yolo_postprocess_npu, without drawing; returns the box count
  YoloV3CpuWorker (common CPU ONNX, keras-style YOLOv3 with NMS inside the graph; NOT the same network as the NPU one)
    preprocess  E worker thread  letterbox 608, RGB, /255, NCHW + image_shape = [[orig_h, orig_w]]
    infer       E worker thread  ORT CPU EP
    postprocess E worker thread  number of selected indices
  CPU ResNet50 uses ClsWorker (worker_cls.py) unchanged.

Ported from the main checkout (read only):
  model_processors.py run_resnet_npu_process (lines 540-650): p0 -> NPU -> p2, output_data[:expected_size] for p2.
  model_processors.py run_yolo_npu_process: dequant order 684/688/692 and back input shapes; yolo_preprocess_local 608.
  model_processors.py run_yolo_cpu_process: image_shape input = [[orig_h, orig_w]] float32.
Differences: no drawing, no class-name lookup, no completion skipped when no box is drawn (the old YOLO NPU process put a
result only when a box was drawn), preprocess for ResNet is the new runtime's ImageNet transform, no file is written.
"""
import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import cv2
import numpy as np

from image_processing import yolo_preprocess_local
from worker_base import Worker, make_session
from worker_cls import cls_preprocess

REPO = Path(__file__).resolve().parent.parent.parent
CONF_THRES = 0.5
IOU_THRES = 0.5


def npu_id_of(execution):
    assert execution in ("npu0", "npu1"), execution
    return int(execution[-1])


class _NpuWorkerBase(Worker):
    """The base __init__ builds self.session from spec['onnx'] (= front partition p0) on the CPU EP, then calls setup()
    and runs one request before the feeder starts. setup() starts the NPU server and loads the back partition."""

    def __init__(self, log, spec, execution, inputs, in_queue, out_queue):
        self.execution = execution
        self.npu_id = npu_id_of(execution)
        self.client = None
        super().__init__(log, spec, "cpu", inputs, in_queue, out_queue)   # "cpu": EP of the p0 / p2 sessions

    def setup(self):
        from npu_server import NpuClient
        s = self.spec
        self.front_in = self.session.get_inputs()[0].name
        self.client = NpuClient(self.npu_id, str((REPO / s["driver_so"]).resolve()), str((REPO / s["npu_binary"]).resolve()))
        self.npu_info = self.client.info
        self.setup_model()

    def setup_model(self):
        pass

    def quantize_and_run(self, x):
        q = self.session.run(None, {self.front_in: x})[0]
        assert q.dtype == np.uint8 and q.size == self.spec["npu_input_bytes"], (q.dtype, q.size)
        return self.client.run(np.ascontiguousarray(q))

    def run(self):
        try:
            super().run()
        finally:
            self.close_rc = self.client.close() if self.client else None
            path = getattr(self, "sidecar", None)
            if path and self.client:
                ms = np.asarray(self.client.npu_ns, dtype=np.float64) / 1e6
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps({
                    "instance": self.log.name, "registry_key": self.spec["name"], "execution": self.execution,
                    "npu_binary": self.spec["npu_binary"], "server": self.npu_info, "close_rc": self.close_rc,
                    "driver_calls": int(ms.size), "driver_call_ms_p50": float(np.median(ms)) if ms.size else None,
                    "driver_call_ms_p95": float(np.percentile(ms, 95)) if ms.size else None,
                    "note": "driver call = SendInput -> Launch -> ReceiveOutputs inside the server process (diagnostic); "
                            "includes the request run in the worker constructor and warm-up"}, indent=1))


class NpuResNet50Worker(_NpuWorkerBase):
    """The NPU binary outputs the 2048 uint8 avgpool features (one 2048-byte buffer). The quantized FC 2048 -> 1000 runs on
    the host as in the vendor reference (parallel_model_partition_test/npu.py lines 136-166), here with the int32 bias and
    the uint8 clip of QLinear Gemm. Integer accumulation is done in float64 (exact below 2**53).
    The p2 partition (ArgMax over 1000 uint8 FC outputs) does not match p1.o and is not executed."""

    def setup_model(self):
        self.w, self.h = self.spec["input_size"]
        P = np.load(str(REPO / self.spec["host_fc_params"]))
        self.zx = float(P["0_avgpool_GlobalAveragePool_output_0_zero_point"])
        self.zw = float(P["0_fc_weight_zero_point"])
        W = P["0_fc_weight_quantized"].astype(np.float64).T          # (2048, 1000)
        self.W = np.ascontiguousarray(W)
        self.k = W.shape[0]
        self.const = -self.zx * W.sum(0) + self.k * self.zx * self.zw + P["0_fc_bias_quantized"].astype(np.float64)
        self.mult = float(P["0_avgpool_GlobalAveragePool_output_0_scale"]) * float(P["0_fc_weight_scale"]) / float(P["0_fc_Gemm_output_0_scale"])
        self.zo = float(P["0_fc_Gemm_output_0_zero_point"])
        assert self.spec["npu_output_bytes"] == self.k == 2048

    def preprocess(self, idx):
        return cls_preprocess(self.inputs[idx % len(self.inputs)], self.w)

    def infer(self, x):
        out = self.quantize_and_run(x)
        assert len(out) == 1 and out[0].size == self.k, [o.size for o in out]
        f = out[0].astype(np.float64)
        acc = f @ self.W + self.const - self.zw * f.sum()
        return np.clip(np.rint(acc * self.mult) + self.zo, 0, 255).astype(np.uint8)

    def postprocess(self, y, idx):
        return int(np.argmax(y))


class NpuYoloV3Worker(_NpuWorkerBase):
    def setup_model(self):
        self.back = make_session(str(REPO / self.spec["back_onnx"]), "cpu")
        d = json.loads((REPO / self.spec["dequant_json"]).read_text())
        self.deq = [(d["params"][k]["scale"], d["params"][k]["zero_point"]) for k in d["order"]]
        self.back_in = [(k, v, int(np.prod(v))) for k, v in d["back_inputs"].items()]
        names = [i.name for i in self.back.get_inputs()]
        assert names == [k for k, _, _ in self.back_in], names
        self.out_names = [o.name for o in self.back.get_outputs()]
        self.w, self.h = self.spec["input_size"]

    def preprocess(self, idx):
        return yolo_preprocess_local(self.inputs[idx % len(self.inputs)], (self.w, self.h))

    def infer(self, x):
        t, meta = x
        outs = self.quantize_and_run(t)
        assert len(outs) == 3, len(outs)
        feeds = {}
        for (name, shape, n), (s, z), buf in zip(self.back_in, self.deq, outs):
            assert buf.size >= n, (name, buf.size, n)
            feeds[name] = ((buf[:n].astype(np.float32) - z) * s).reshape(shape)
        res = dict(zip(self.out_names, self.back.run(None, feeds)))
        return res["dets"], res["labels"], meta

    def postprocess(self, y, idx):
        dets, labels, meta = y
        d = np.squeeze(dets, 0)
        keep = d[:, 4] >= CONF_THRES
        b, s = d[keep, :4], d[keep, 4]
        if len(b) == 0:
            return 0
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)
        k = cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), CONF_THRES, IOU_THRES)
        k = [int(i) for i in np.asarray(k).reshape(-1)] if k is not None and len(k) else []
        (dw, dh), r, w0, h0 = meta["pad"], meta["ratio"], meta["orig_w"], meta["orig_h"]
        n = 0
        for j in k:
            x1, y1, x2, y2 = b[j]
            x1, x2 = np.clip((x1 - dw) / r, 0, w0), np.clip((x2 - dw) / r, 0, w0)
            y1, y2 = np.clip((y1 - dh) / r, 0, h0), np.clip((y2 - dh) / r, 0, h0)
            bw, bh = float(x2 - x1), float(y2 - y1)
            if bw < 4.0 or bh < 4.0 or bw > 0.98 * w0 or bh > 0.98 * h0:
                continue
            ar = bw / (bh + 1e-6)
            if ar > 25.0 or ar < 1.0 / 25.0:
                continue
            n += 1
        return n


class YoloV3CpuWorker(Worker):
    def setup(self):
        ins = self.session.get_inputs()
        img = [i for i in ins if len(i.shape) == 4]
        shp = [i for i in ins if "image_shape" in i.name]
        assert len(img) == 1 and len(shp) == 1, [(i.name, i.shape) for i in ins]
        self.input_name, self.shape_name = img[0].name, shp[0].name
        self.w, self.h = self.spec["input_size"]
        assert list(img[0].shape[1:]) == [3, self.h, self.w], img[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, idx):
        frame = self.inputs[idx % len(self.inputs)]
        t, meta = yolo_preprocess_local(frame, (self.w, self.h))
        return t, np.array([[meta["orig_h"], meta["orig_w"]]], dtype=np.float32)

    def infer(self, x):
        t, shape = x
        return self.session.run(self.output_names, {self.input_name: t, self.shape_name: shape})

    def postprocess(self, y, idx):
        return int(np.asarray(y[-1]).reshape(-1, 3).shape[0])


def make_antara_worker(log, spec, execution, inputs, q_in, q_out):
    post = spec["postprocess"]
    if spec.get("device_kind") == "npu":
        assert execution in ("npu0", "npu1"), (spec["name"], execution)
    else:
        assert execution == "cpu", (spec["name"], execution)
    if post == "npu_resnet50_hostfc_argmax":
        return NpuResNet50Worker(log, spec, execution, inputs["imagenet"], q_in, q_out)
    if post == "npu_yolov3_nms_count":
        return NpuYoloV3Worker(log, spec, execution, inputs["video"], q_in, q_out)
    if post == "yolov3_keras_nms_count":
        return YoloV3CpuWorker(log, spec, "cpu", inputs["video"], q_in, q_out)
    if post == "cls_argmax":
        from worker_cls import ClsWorker
        return ClsWorker(log, spec, "cpu", inputs["imagenet"], q_in, q_out)
    raise SystemExit(f"[FATAL] unknown postprocess {post!r} for {spec['name']}")
