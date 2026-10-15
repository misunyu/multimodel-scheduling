"""YOLOv4 (tf2onnx, NHWC 416) worker: letterbox preprocess, anchor decode + NMS postprocess (ONNX model zoo recipe)."""
import sys

sys.dont_write_bytecode = True

import cv2
import numpy as np

from image_processing import yolo_preprocess_local
from worker_base import Worker

ANCHORS = np.array([[[12, 16], [19, 36], [40, 28]], [[36, 75], [76, 55], [72, 146]], [[142, 110], [192, 243], [459, 401]]],
                   dtype=np.float32)
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]
SCORE_THRES = 0.25
IOU_THRES = 0.213


class YoloV4Worker(Worker):
    def setup(self):
        i = self.session.get_inputs()[0]
        self.input_name = i.name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.w, self.h = self.spec["input_size"]
        assert self.spec["layout"] == "NHWC" and list(i.shape[1:]) == [self.h, self.w, 3], (i.shape, self.spec)

    def preprocess(self, idx):
        t, meta = yolo_preprocess_local(self.inputs[idx % len(self.inputs)], (self.w, self.h))
        return np.ascontiguousarray(np.transpose(t, (0, 2, 3, 1))), meta

    def infer(self, x):
        t, meta = x
        return self.session.run(self.output_names, {self.input_name: t}), meta

    def postprocess(self, y, idx):
        outs, meta = y
        boxes = []
        for i, pred in enumerate(outs):
            g = pred.shape[1]
            xy = np.stack(np.meshgrid(np.arange(g), np.arange(g)), axis=-1)[None, :, :, None, :].astype(np.float32)
            dxdy = 1.0 / (1.0 + np.exp(-pred[..., 0:2]))
            pxy = (dxdy * XYSCALE[i] - 0.5 * (XYSCALE[i] - 1) + xy) * STRIDES[i]
            pwh = np.exp(np.clip(pred[..., 2:4], -20, 20)) * ANCHORS[i]
            boxes.append(np.concatenate([pxy, pwh, pred[..., 4:]], axis=-1).reshape(-1, pred.shape[-1]))
        b = np.concatenate(boxes, axis=0)
        cls = np.argmax(b[:, 5:], axis=1)
        score = b[:, 4] * b[np.arange(len(b)), 5 + cls]
        keep = score > SCORE_THRES
        b, cls, score = b[keep], cls[keep], score[keep]
        if len(b) == 0:
            return 0
        dw, dh = meta["pad"]
        r = meta["ratio"]
        x1 = (b[:, 0] - b[:, 2] / 2 - dw) / r
        y1 = (b[:, 1] - b[:, 3] / 2 - dh) / r
        w = b[:, 2] / r
        h = b[:, 3] / r
        rects = np.stack([x1, y1, w, h], axis=1)
        # class-aware NMS by offsetting boxes per class
        off = (cls.astype(np.float32) * 4096.0)[:, None] * np.array([1, 1, 0, 0], dtype=np.float32)
        idxs = cv2.dnn.NMSBoxes((rects + off).tolist(), score.tolist(), SCORE_THRES, IOU_THRES)
        return int(len(idxs))
