#!/usr/bin/env python3
"""
Create ground-truth for imagenet-sample-images and evaluate ResNet50 (ONNX) on CPU.

- Ground-truth file will be saved as imagenet-sample-images/ground_truth.csv with columns:
  filename,class_name,class_index
- The ONNX model at models/resnet50_big/model/resnet50_big.onnx will be executed on CPU exactly once
  for each image, and top-1 accuracy will be reported. Progress is printed continuously.

Usage:
    python eval_resnet50_imagenet_cpu.py

Requirements: onnxruntime, opencv-python-headless, numpy, Pillow; provided in requirements.txt
"""
from __future__ import annotations
import os
import csv
import glob
import string
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import onnxruntime as ort

# Reuse repo's preprocessing for consistency with provided models
from image_processing import resnet50_preprocess_local

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(REPO_ROOT, "imagenet-sample-images")
CLASSES_TXT = os.path.join(REPO_ROOT, "imagenet_classes.txt")
GROUND_TRUTH_CSV = os.path.join(IMAGES_DIR, "ground_truth.csv")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "resnet50_big", "model", "resnet50_big.onnx")

# Logging helpers
RUN_TS = time.strftime("%Y%m%d_%H%M%S")
DEFAULT_LOG_DIR = os.path.join(REPO_ROOT, "results")

def _ensure_log_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_log_line(log_path: Optional[str], text: str) -> None:
    if not log_path:
        return
    _ensure_log_dir(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def load_imagenet_classes(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if len(classes) != 1000:
        print(f"[WARN] Expected 1000 ImageNet classes, got {len(classes)}")
    return classes


def normalize_label(text: str) -> str:
    # Lowercase, replace underscores with spaces, remove punctuation except spaces
    t = text.lower().replace("_", " ").strip()
    # Keep letters, digits and spaces
    allowed = set(string.ascii_lowercase + string.digits + " ")
    t = "".join(ch for ch in t if ch in allowed)
    # collapse multiple spaces
    t = " ".join(t.split())
    return t


def build_ground_truth(images_dir: str, classes_txt: str, output_csv: str) -> Tuple[List[str], Dict[str, int]]:
    classes = load_imagenet_classes(classes_txt)
    norm_to_index: Dict[str, int] = {normalize_label(name): i for i, name in enumerate(classes)}

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.JPEG")))
    image_paths += sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    image_paths += sorted(glob.glob(os.path.join(images_dir, "*.jpeg")))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    rows: List[Tuple[str, str, int]] = []
    filename_to_index: Dict[str, int] = {}
    errors: List[str] = []

    for p in image_paths:
        fname = os.path.basename(p)
        stem, _ = os.path.splitext(fname)
        # Expect format like n01443537_goldfish; split at the first underscore
        if "_" in stem:
            human_part = stem.split("_", 1)[1]
        else:
            # fallback: use full stem
            human_part = stem
        norm = normalize_label(human_part)
        if norm in norm_to_index:
            idx = norm_to_index[norm]
            rows.append((fname, classes[idx], idx))
            filename_to_index[fname] = idx
        else:
            errors.append(f"{fname} -> '{human_part}' (normalized '{norm}') not found in classes list")

    if errors:
        print("[WARN] Some filenames could not be mapped to ImageNet classes. Showing first 10:")
        for e in errors[:10]:
            print("  -", e)
        print(f"[WARN] Mapped {len(rows)} images; {len(errors)} images did not match.")

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_name", "class_index"])  # header
        for fname, cname, idx in rows:
            writer.writerow([fname, cname, idx])
    print(f"[INFO] Wrote ground truth for {len(rows)} images to {output_csv}")
    return classes, filename_to_index


def load_images_preprocessed(images_dir: str, selected_filenames: List[str]) -> List[Tuple[str, np.ndarray]]:
    data = []
    for fname in selected_filenames:
        path = os.path.join(images_dir, fname)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Failed to read image: {path}")
            continue
        inp = resnet50_preprocess_local(img_bgr)  # shape (1,3,224,224), float32 in [0,1]
        data.append((fname, inp))
    return data


def run_onnx_inference_cpu(
    model_path: str,
    batch_inputs: List[Tuple[str, np.ndarray]],
    repeats: int = 1,
    gt_lookup: Optional[Dict[str, int]] = None,
    classes: Optional[List[str]] = None,
    log_path: Optional[str] = None,
) -> Tuple[Dict[str, int], float, Optional[Dict[str, List[int]]]]:
    """
    Run ONNX model on CPU.
    - If repeats == 1 and gt_lookup is provided, prints per-image progress with running accuracy (top-1 and top-5).
    - Also writes the same progress lines to log_path if provided.
    - Returns (predictions_by_filename, elapsed_seconds, top5_indices_by_filename or None)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]) 

    # Determine input/output names and expected layout dynamically
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_name = inputs[0].name if inputs else "data"
    output_name = outputs[0].name if outputs else None
    input_shape = inputs[0].shape if inputs else None
    # Heuristic for layout: NCHW if channel dim is 3 at index 1; NHWC if last dim is 3
    layout = "NCHW"
    if isinstance(input_shape, (list, tuple)) and len(input_shape) == 4:
        c_dim = input_shape[1]
        last_dim = input_shape[3]
        if last_dim == 3 or (isinstance(last_dim, str) and str(last_dim).upper() in ("C", "CHANNEL", "CHANNELS")):
            layout = "NHWC"
        elif c_dim == 3 or (isinstance(c_dim, str) and str(c_dim).upper() in ("C", "CHANNEL", "CHANNELS")):
            layout = "NCHW"
    io_msg = f"[INFO] Model IO - input_name={input_name}, output_name={output_name}, input_shape={input_shape}, layout={layout}"
    print(io_msg)
    write_log_line(log_path, io_msg)

    last_preds: Dict[str, int] = {}
    last_top5: Optional[Dict[str, List[int]]] = None

    t0 = time.time()
    if repeats == 1:
        total = len(batch_inputs)
        correct_so_far_top1 = 0
        correct_so_far_top5 = 0
        last_top5 = {}
        for i, (fname, inp) in enumerate(batch_inputs, start=1):
            feed_inp = np.transpose(inp, (0, 2, 3, 1)) if layout == "NHWC" else inp
            outputs = session.run([output_name] if output_name else None, {input_name: feed_inp})
            logits = outputs[0]
            # logits may be (1,1000) or (1000,), ensure shape
            if logits.ndim == 2:
                logits_arr = logits[0]
            else:
                logits_arr = logits
            pred_idx = int(np.argmax(logits_arr))
            # compute top-5 indices (highest first)
            top5_idx = np.argsort(logits_arr)[-5:][::-1].astype(int).tolist()

            last_preds[fname] = pred_idx
            last_top5[fname] = top5_idx

            if gt_lookup is not None:
                gt_idx = gt_lookup.get(fname, None)
                correct_top1 = (gt_idx is not None and pred_idx == gt_idx)
                correct_top5 = (gt_idx is not None and gt_idx in top5_idx)
                if correct_top1:
                    correct_so_far_top1 += 1
                if correct_top5:
                    correct_so_far_top5 += 1
                pred_name = classes[pred_idx] if classes and 0 <= pred_idx < len(classes) else str(pred_idx)
                gt_name = classes[gt_idx] if (classes and gt_idx is not None and 0 <= gt_idx < len(classes)) else str(gt_idx)
                running_top1 = correct_so_far_top1 / i
                running_top5 = correct_so_far_top5 / i
                msg = (
                    f"[PROGRESS] {i}/{total} {fname} -> pred: {pred_name} (#{pred_idx}), gt: {gt_name} (#{gt_idx}), "
                    f"top-1_correct={correct_top1}, top-5_correct={correct_top5} | "
                    f"running top-1: {running_top1*100:.2f}% | running top-5: {running_top5*100:.2f}%"
                )
                print(msg, flush=True)
                write_log_line(log_path, msg)
            else:
                msg = f"[PROGRESS] Processed {i}/{total}: {fname}"
                print(msg, flush=True)
                write_log_line(log_path, msg)
    else:
        for r in range(repeats):
            for fname, inp in batch_inputs:
                feed_inp = np.transpose(inp, (0, 2, 3, 1)) if layout == "NHWC" else inp
                outputs = session.run([output_name] if output_name else None, {input_name: feed_inp})
                logits = outputs[0]
                if logits.ndim == 2:
                    pred_idx = int(np.argmax(logits[0]))
                else:
                    pred_idx = int(np.argmax(logits))
                last_preds[fname] = pred_idx
    t1 = time.time()
    elapsed = t1 - t0
    total_infers = len(batch_inputs) * repeats
    info_msg = f"[INFO] Ran {len(batch_inputs)} images x {repeats} repeats = {total_infers} inferences on CPU in {elapsed:.2f}s"
    print(info_msg)
    write_log_line(log_path, info_msg)
    return last_preds, elapsed, last_top5


def compute_accuracy(preds: Dict[str, int], gt: Dict[str, int], top5: Optional[Dict[str, List[int]]] = None) -> Tuple[float, float]:
    # top-1 accuracy
    correct1 = 0
    total = 0
    for fname, pred in preds.items():
        if fname in gt:
            total += 1
            if pred == gt[fname]:
                correct1 += 1
    top1 = correct1 / total if total else 0.0

    # top-5 accuracy (if top5 indices provided)
    if top5 is not None:
        correct5 = 0
        for fname, indices in top5.items():
            if fname in gt and gt[fname] in indices:
                correct5 += 1
        top5_acc = correct5 / total if total else 0.0
    else:
        top5_acc = float('nan')

    return top1, top5_acc


def main():
    # Prepare log file path
    log_filename = f"eval_resnet50_imagenet_cpu_{RUN_TS}.log"
    log_path = os.path.join(DEFAULT_LOG_DIR, log_filename)
    write_log_line(log_path, f"[START] ResNet50 ImageNet evaluation @ {RUN_TS}")

    print("[STEP 1] Building ground-truth from filenames...")
    write_log_line(log_path, "[STEP 1] Building ground-truth from filenames...")
    classes, filename_to_index = build_ground_truth(IMAGES_DIR, CLASSES_TXT, GROUND_TRUTH_CSV)

    print("[STEP 2] Loading and preprocessing images (once)...")
    write_log_line(log_path, "[STEP 2] Loading and preprocessing images (once)...")
    image_filenames = sorted(filename_to_index.keys())
    preprocessed = load_images_preprocessed(IMAGES_DIR, image_filenames)
    if not preprocessed:
        err = "[ERROR] No images preprocessed; aborting."
        print(err)
        write_log_line(log_path, err)
        return

    print("[STEP 3] Running ONNX model on CPU (1x per image) with live progress...")
    write_log_line(log_path, "[STEP 3] Running ONNX model on CPU (1x per image) with live progress...")
    preds, elapsed, top5_dict = run_onnx_inference_cpu(
        MODEL_PATH,
        preprocessed,
        repeats=1,
        gt_lookup=filename_to_index,
        classes=classes,
        log_path=log_path,
    )

    print("[STEP 4] Computing accuracy...")
    write_log_line(log_path, "[STEP 4] Computing accuracy...")
    top1, top5 = compute_accuracy(preds, filename_to_index, top5=top5_dict)

    # Final summary
    summary_lines = [
        "",
        "===== Evaluation Result =====",
        f"Images evaluated: {len(preprocessed)}",
        f"Model repeats: 1",
        f"Elapsed: {elapsed:.2f}s",
        f"Top-1 accuracy: {top1*100:.2f}%",
        f"Top-5 accuracy: {top5*100:.2f}%" if not np.isnan(top5) else "Top-5 accuracy: N/A",
    ]
    for line in summary_lines:
        print(line)
        write_log_line(log_path, line)

    # Rewrite log so the final accuracy appears at the very top of the file
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            original = f.read()
        header = (
            f"[FINAL] Top-1 accuracy: {top1*100:.2f}%\n"
            + (f"[FINAL] Top-5 accuracy: {top5*100:.2f}%\n" if not np.isnan(top5) else "[FINAL] Top-5 accuracy: N/A\n")
        )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(original)
    except Exception as e:
        print(f"[WARN] Failed to re-write log header with final accuracy: {e}")
        # Not critical; continue


# ===================== NPU Evaluation Utilities =====================
from typing import Iterable

try:
    import onnx
    import onnxruntime as ort  # already imported as ort
    import npu as npu_mod
except Exception:
    npu_mod = None


def load_images_raw(images_dir: str, selected_filenames: List[str]) -> List[Tuple[str, np.ndarray]]:
    data = []
    for fname in selected_filenames:
        path = os.path.join(images_dir, fname)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Failed to read image: {path}")
            continue
        data.append((fname, img_bgr))
    return data


def discover_npu_models(models_root: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Discover NPU-capable models under models/.
    Returns list of tuples: (model_key, npu_o_path, hint)
    For now, focus on resnet50_small which has known NPU binary.
    """
    found: List[Tuple[str, str, Optional[str]]] = []
    # Known ResNet50 small layout
    resnet_o = os.path.join(models_root, "resnet50_small", "npu_code", "resnet50_small_neubla_p1.o")
    if os.path.exists(resnet_o):
        found.append(("resnet50_small", resnet_o, None))
    return found


def run_resnet_inference_npu(
    npu_o_path: str,
    batch_inputs_raw: List[Tuple[str, np.ndarray]],
    gt_lookup: Optional[Dict[str, int]] = None,
    classes: Optional[List[str]] = None,
    log_path: Optional[str] = None,
    npu_id: int = 1,
) -> Tuple[Dict[str, int], float, Optional[Dict[str, List[int]]]]:
    """
    Run ResNet50 on NPU exactly like run_resnet_npu_process in model_processors.py.
    Do NOT use partitions/*.onnx. Instead, use npu.resnet50_prepare_onnx_model to get front_sess
    and quantization params, and perform the manual FC fallback when needed.
    Returns (pred_by_filename, elapsed_seconds, top5_indices_by_filename)
    """
    if npu_mod is None:
        raise RuntimeError("npu module not available; cannot run NPU evaluation")

    # Prepare host (front) session and params like run_resnet_npu_process
    try:
        front_sess, back_sess, params = npu_mod.resnet50_prepare_onnx_model(
            "../resnet/resnet50-0676ba61_opset12.neubla_u8_lwq_percentile.onnx"
        )
    except Exception as e:
        raise RuntimeError(f"Host model preparation failed: {e}")

    # Extract quantization parameters
    scale = params['/0/avgpool/GlobalAveragePool_output_0_scale'] * params['0.fc.weight_scale']
    zp_act = params['/0/avgpool/GlobalAveragePool_output_0_zero_point']
    zp_w = params['0.fc.weight_zero_point']
    scale_out = params['/0/fc/Gemm_output_0_scale']
    zp_out = params['/0/fc/Gemm_output_0_zero_point']
    weight_q = params['0.fc.weight_quantized'].T.astype(np.int32)

    # Initialize NPU driver
    driver = None
    t0 = time.time()
    try:
        driver = npu_mod.initialize_driver(npu_id, npu_o_path)

        preds: Dict[str, int] = {}
        top5_dict: Dict[str, List[int]] = {}
        total = len(batch_inputs_raw)
        correct_so_far_top1 = 0
        correct_so_far_top5 = 0

        for i, (fname, img_bgr) in enumerate(batch_inputs_raw, start=1):
            # Preprocess and run front onnx (quantization)
            pre_inp = npu_mod.resnet50_preprocess(img_bgr)
            front_out = front_sess.run(None, {"input": pre_inp})[0]
            input_bytes = front_out.tobytes()

            # Send to NPU and receive intermediate
            raw_outputs = npu_mod.send_receive_data_npu(driver, input_bytes, 3 * 224 * 224)
            output_data = np.frombuffer(raw_outputs[0], dtype=np.uint8)

            # Try back_sess first (if available), else manual FC computation
            try:
                if back_sess is not None:
                    back_output = back_sess.run(None, {"input": output_data.reshape(1, -1)})
                    logits_arr = back_output[0]
                else:
                    raise RuntimeError("Back session not available")
            except Exception:
                # Manual quantized FC as in run_resnet_npu_process
                out = np.matmul(output_data.astype(np.int32), weight_q)
                out -= zp_act * np.sum(weight_q, axis=0)
                out -= zp_w * np.sum(output_data, axis=0)
                out += zp_act * zp_w
                out = np.round(out * scale / scale_out) + zp_out
                # For argmax and top-5, cast to uint8-like range consistent with code
                logits_arr = out.astype(np.uint8)

            # Compute prediction and top-5
            pred_idx = int(np.argmax(logits_arr))
            top5_idx = np.argsort(logits_arr)[-5:][::-1].astype(int).tolist()
            preds[fname] = pred_idx
            top5_dict[fname] = top5_idx

            if gt_lookup is not None:
                gt_idx = gt_lookup.get(fname)
                correct_top1 = (gt_idx is not None and pred_idx == gt_idx)
                correct_top5 = (gt_idx is not None and gt_idx in top5_idx)
                if correct_top1:
                    correct_so_far_top1 += 1
                if correct_top5:
                    correct_so_far_top5 += 1
                running_top1 = correct_so_far_top1 / i
                running_top5 = correct_so_far_top5 / i
                pred_name = classes[pred_idx] if classes and 0 <= pred_idx < len(classes) else str(pred_idx)
                gt_name = classes[gt_idx] if (classes and gt_idx is not None and 0 <= gt_idx < len(classes)) else str(gt_idx)
                msg = (
                    f"[PROGRESS-NPU] {i}/{total} {fname} -> pred: {pred_name} (#{pred_idx}), gt: {gt_name} (#{gt_idx}), "
                    f"top-1_correct={correct_top1}, top-5_correct={correct_top5} | "
                    f"running top-1: {running_top1*100:.2f}% | running top-5: {running_top5*100:.2f}%"
                )
                print(msg, flush=True)
                write_log_line(log_path, msg)
            else:
                msg = f"[PROGRESS-NPU] Processed {i}/{total}: {fname}"
                print(msg, flush=True)
                write_log_line(log_path, msg)
    finally:
        if driver is not None:
            try:
                npu_mod.close_driver(driver)
            except Exception:
                pass
    t1 = time.time()
    elapsed = t1 - t0
    total_infers = len(batch_inputs_raw)
    info_msg = f"[INFO] Ran {total_infers} images on NPU in {elapsed:.2f}s"
    print(info_msg)
    write_log_line(log_path, info_msg)
    return preds, elapsed, top5_dict


def main():
    # Prepare log file path for CPU part (kept for compatibility)
    log_filename = f"eval_resnet50_imagenet_cpu_{RUN_TS}.log"
    log_path = os.path.join(DEFAULT_LOG_DIR, log_filename)
    write_log_line(log_path, f"[START] ResNet50 ImageNet evaluation @ {RUN_TS}")

    print("[STEP 1] Building ground-truth from filenames...")
    write_log_line(log_path, "[STEP 1] Building ground-truth from filenames...")
    classes, filename_to_index = build_ground_truth(IMAGES_DIR, CLASSES_TXT, GROUND_TRUTH_CSV)

    print("[STEP 2] Loading and preprocessing images (once)...")
    write_log_line(log_path, "[STEP 2] Loading and preprocessing images (once)...")
    image_filenames = sorted(filename_to_index.keys())
    preprocessed = load_images_preprocessed(IMAGES_DIR, image_filenames)
    if not preprocessed:
        err = "[ERROR] No images preprocessed; aborting."
        print(err)
        write_log_line(log_path, err)
        return

    # ===== NPU evaluation first =====
    npu_models = discover_npu_models(os.path.join(REPO_ROOT, "models"))
    npu_last_result = None  # (model_key, top1, top5, elapsed, images)
    if npu_models and npu_mod is not None:
        raw_imgs = load_images_raw(IMAGES_DIR, image_filenames)
        npu_log_filename = f"eval_resnet50_imagenet_npu_{RUN_TS}.log"
        npu_log_path = os.path.join(DEFAULT_LOG_DIR, npu_log_filename)
        write_log_line(npu_log_path, f"[START] ResNet50 ImageNet NPU evaluation @ {RUN_TS}")

        for model_key, o_path, _ in npu_models:
            print(f"[STEP NPU] Evaluating {model_key} on NPU using {o_path}...")
            write_log_line(npu_log_path, f"[STEP NPU] Evaluating {model_key} on NPU using {o_path}...")
            try:
                npu_preds, npu_elapsed, npu_top5 = run_resnet_inference_npu(
                    npu_o_path=o_path,
                    batch_inputs_raw=raw_imgs,
                    gt_lookup=filename_to_index,
                    classes=classes,
                    log_path=npu_log_path,
                    npu_id=1,
                )
                npu_top1, npu_top5_acc = compute_accuracy(npu_preds, filename_to_index, top5=npu_top5)

                # Log NPU results
                lines = [
                    "",
                    f"===== NPU Evaluation Result: {model_key} =====",
                    f"Images evaluated: {len(raw_imgs)}",
                    f"Elapsed: {npu_elapsed:.2f}s",
                    f"Top-1 accuracy: {npu_top1*100:.2f}%",
                    f"Top-5 accuracy: {npu_top5_acc*100:.2f}%" if not np.isnan(npu_top5_acc) else "Top-5 accuracy: N/A",
                ]
                for l in lines:
                    print(l)
                    write_log_line(npu_log_path, l)

                # Store last NPU result to be included in combined metrics after CPU eval
                npu_last_result = (model_key, npu_top1, (None if np.isnan(npu_top5_acc) else npu_top5_acc), npu_elapsed, len(raw_imgs))
            except Exception as e:
                err = f"[ERROR] NPU evaluation failed for {model_key}: {e}"
                print(err)
                write_log_line(npu_log_path, err)
    else:
        print("[INFO] No NPU-capable ResNet models discovered or npu module unavailable; skipping NPU eval.")

    # ===== CPU evaluation afterwards =====
    print("[STEP 3] Running ONNX model on CPU (1x per image) with live progress...")
    write_log_line(log_path, "[STEP 3] Running ONNX model on CPU (1x per image) with live progress...")
    preds, elapsed, top5_dict = run_onnx_inference_cpu(
        MODEL_PATH,
        preprocessed,
        repeats=1,
        gt_lookup=filename_to_index,
        classes=classes,
        log_path=log_path,
    )

    print("[STEP 4] Computing accuracy...")
    write_log_line(log_path, "[STEP 4] Computing accuracy...")
    top1, top5 = compute_accuracy(preds, filename_to_index, top5=top5_dict)

    # Final summary for CPU
    summary_lines = [
        "",
        "===== CPU Evaluation Result =====",
        f"Images evaluated: {len(preprocessed)}",
        f"Model repeats: 1",
        f"Elapsed: {elapsed:.2f}s",
        f"Top-1 accuracy: {top1*100:.2f}%",
        f"Top-5 accuracy: {top5*100:.2f}%" if not np.isnan(top5) else "Top-5 accuracy: N/A",
    ]
    for line in summary_lines:
        print(line)
        write_log_line(log_path, line)

    # Rewrite log so the final accuracy appears at the very top of the file
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            original = f.read()
        header = (
            f"[FINAL] CPU Top-1: {top1*100:.2f}%\n"
            + (f"[FINAL] CPU Top-5: {top5*100:.2f}%\n" if not np.isnan(top5) else "[FINAL] CPU Top-5: N/A\n")
        )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(original)
    except Exception as e:
        print(f"[WARN] Failed to re-write log header with final accuracy: {e}")

    # Write combined metrics once, after CPU evaluation, if NPU results exist
    if npu_last_result is not None:
        model_key, npu_top1_val, npu_top5_val, npu_elapsed_val, npu_images = npu_last_result
        metrics_path = os.path.join(
            DEFAULT_LOG_DIR, f"imagenet_resnet50_accuracy_{RUN_TS}.json"
        )
        record = {
            "timestamp": RUN_TS,
            "images": npu_images,
            "cpu": {"top1": top1, "top5": (None if np.isnan(top5) else top5), "elapsed_sec": elapsed},
            "npu": {"model": model_key, "top1": npu_top1_val, "top5": npu_top5_val, "elapsed_sec": npu_elapsed_val},
        }
        _ensure_log_dir(metrics_path)
        with open(metrics_path, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(record, f, indent=2)
        print(f"[INFO] Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
