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
from image_processing import resnet50_preprocess_local, letterbox

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(REPO_ROOT, "imagenet-sample-images")
CLASSES_TXT = os.path.join(REPO_ROOT, "imagenet_classes.txt")
GROUND_TRUTH_CSV = os.path.join(IMAGES_DIR, "ground_truth.csv")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "resnet50_big", "model", "resnet50_big.onnx")
# Partitioned model paths (CPU pipeline)
# Default CPU-only pipeline is p1 -> p2 per new requirement
P0_MODEL_PATH = os.path.join(REPO_ROOT, "model_partitions", "partition1", "resnet50_cpu_1", "resnet50_cpu_1_p0.onnx")
P1_MODEL_PATH = os.path.join(REPO_ROOT, "model_partitions", "partition1", "resnet50_cpu_1", "resnet50_cpu_1_p1.onnx")
P2_MODEL_PATH = os.path.join(REPO_ROOT, "model_partitions", "partition1", "resnet50_cpu_1", "resnet50_cpu_1_p2.onnx")

# Logging helpers
RUN_TS = time.strftime("%Y%m%d_%H%M%S")
DEFAULT_LOG_DIR = os.path.join(REPO_ROOT, "results")

# ===================== Configurable Preprocessing (CLI) =====================
import argparse
from typing import Callable

CLI_ARGS = None  # populated in __main__

def _parse_csv_floats(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]

def build_preprocessor_from_args(args) -> Callable[[np.ndarray], np.ndarray]:
    target = (224, 224)
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    interp = interp_map.get(args.interp, cv2.INTER_LINEAR)

    # Mean/std handling
    mean = None
    std = None
    if args.imagenet_norm:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if args.mean:
            mean = _parse_csv_floats(args.mean)
        if args.std:
            std = _parse_csv_floats(args.std)

    def preprocess(img_bgr: np.ndarray) -> np.ndarray:
        img = img_bgr
        # Resize/crop pipeline
        if args.pp_mode == "letterbox":
            img, _, _ = letterbox(img, target)
        elif args.pp_mode == "resize":
            img = cv2.resize(img, target, interpolation=interp)
        elif args.pp_mode == "center-crop":
            short = int(args.short_side)
            h, w = img.shape[:2]
            scale = short / min(h, w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
            # center crop to target
            y0 = max(0, (new_h - target[1]) // 2)
            x0 = max(0, (new_w - target[0]) // 2)
            img = img[y0:y0 + target[1], x0:x0 + target[0]]
            if img.shape[0] != target[1] or img.shape[1] != target[0]:
                img = cv2.resize(img, target, interpolation=interp)
        else:
            # fallback: plain resize
            img = cv2.resize(img, target, interpolation=interp)

        # Color
        if args.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Scaling
        img = img.astype(np.float32)
        if args.scale_255:
            img = img / 255.0

        # Normalize
        if mean is not None and std is not None:
            m = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
            s = np.array(std, dtype=np.float32).reshape(1, 1, 3)
            img = (img - m) / s

        # To NCHW
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    return preprocess

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


def load_images_preprocessed(images_dir: str, selected_filenames: List[str], preprocessor=None) -> List[Tuple[str, np.ndarray]]:
    data = []
    if preprocessor is None:
        preprocessor = resnet50_preprocess_local
    for fname in selected_filenames:
        path = os.path.join(images_dir, fname)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Failed to read image: {path}")
            continue
        inp = preprocessor(img_bgr)  # shape (1,3,224,224), float32 typically
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


def run_onnx_inference_cpu_partitioned(
    p0_model_path: str,
    p1_model_path: str,
    batch_inputs: List[Tuple[str, np.ndarray]],
    gt_lookup: Optional[Dict[str, int]] = None,
    classes: Optional[List[str]] = None,
    log_path: Optional[str] = None,
) -> Tuple[Dict[str, int], float, Optional[Dict[str, List[int]]]]:
    """
    Run two ONNX models on CPU in sequence (partitioned pipeline):
    - Input goes to p0 (partition 0)
    - p0's output is fed into p1 (partition 1)
    - Final logits are taken from p1's output
    Returns (predictions_by_filename, elapsed_seconds, top5_indices_by_filename)
    """
    if not os.path.exists(p0_model_path):
        raise FileNotFoundError(f"Model p0 not found: {p0_model_path}")
    if not os.path.exists(p1_model_path):
        raise FileNotFoundError(f"Model p1 not found: {p1_model_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
    sess0 = ort.InferenceSession(p0_model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]) 
    sess1 = ort.InferenceSession(p1_model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]) 

    # IO info for p0
    p0_inputs = sess0.get_inputs()
    p0_outputs = sess0.get_outputs()
    p0_in_name = p0_inputs[0].name if p0_inputs else "input"
    p0_out_name = p0_outputs[0].name if p0_outputs else None
    p0_in_shape = p0_inputs[0].shape if p0_inputs else None

    # Determine layout for p0 input
    layout = "NCHW"
    if isinstance(p0_in_shape, (list, tuple)) and len(p0_in_shape) == 4:
        c_dim = p0_in_shape[1]
        last_dim = p0_in_shape[3]
        if last_dim == 3 or (isinstance(last_dim, str) and str(last_dim).upper() in ("C", "CHANNEL", "CHANNELS")):
            layout = "NHWC"
        elif c_dim == 3 or (isinstance(c_dim, str) and str(c_dim).upper() in ("C", "CHANNEL", "CHANNELS")):
            layout = "NCHW"

    p0_io_msg = f"[INFO] P0 IO - input_name={p0_in_name}, output_name={p0_out_name}, input_shape={p0_in_shape}, layout={layout}"
    print(p0_io_msg)
    write_log_line(log_path, p0_io_msg)

    # IO info for p1
    p1_inputs = sess1.get_inputs()
    p1_outputs = sess1.get_outputs()
    p1_in_name = p1_inputs[0].name if p1_inputs else "input"
    p1_out_name = p1_outputs[0].name if p1_outputs else None
    p1_in_shape = p1_inputs[0].shape if p1_inputs else None

    p1_io_msg = f"[INFO] P1 IO - input_name={p1_in_name}, output_name={p1_out_name}, input_shape={p1_in_shape}"
    print(p1_io_msg)
    write_log_line(log_path, p1_io_msg)

    last_preds: Dict[str, int] = {}
    last_top5: Dict[str, List[int]] = {}

    t0 = time.time()
    total = len(batch_inputs)
    correct_so_far_top1 = 0
    correct_so_far_top5 = 0

    for i, (fname, inp) in enumerate(batch_inputs, start=1):
        # Conform input for p0
        feed_inp0 = np.transpose(inp, (0, 2, 3, 1)) if layout == "NHWC" else inp
        p0_out_list = sess0.run([p0_out_name] if p0_out_name else None, {p0_in_name: feed_inp0})
        inter = p0_out_list[0]

        # Try to align inter with p1's expected shape in a minimal way
        try:
            if isinstance(p1_in_shape, (list, tuple)) and len(p1_in_shape) > 0:
                # Handle batch dimension heuristically
                want_batch = p1_in_shape[0]
                if want_batch == 1 and (inter.ndim == len(p1_in_shape) - 1):
                    # Add batch axis
                    inter = np.expand_dims(inter, axis=0)
                elif want_batch in (None, 'N', 'n', 'batch', 'Batch', 'B'):
                    # Do nothing; dynamic batch
                    pass
                elif want_batch is None:
                    pass
                # If p1 expects no batch but inter has batch of 1
                if isinstance(p1_in_shape, (list, tuple)) and (p1_in_shape[0] in (None,)) and inter.ndim == len(p1_in_shape) + 1 and inter.shape[0] == 1:
                    inter = np.squeeze(inter, axis=0)
        except Exception:
            # Best-effort only; fallback uses raw inter
            pass

        p1_out_list = sess1.run([p1_out_name] if p1_out_name else None, {p1_in_name: inter})
        logits = p1_out_list[0]
        if logits.ndim == 2:
            logits_arr = logits[0]
        else:
            logits_arr = logits

        pred_idx = int(np.argmax(logits_arr))
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
            running_top1 = correct_so_far_top1 / i
            running_top5 = correct_so_far_top5 / i
            pred_name = classes[pred_idx] if classes and 0 <= pred_idx < len(classes) else str(pred_idx)
            gt_name = classes[gt_idx] if (classes and gt_idx is not None and 0 <= gt_idx < len(classes)) else str(gt_idx)
            msg = (
                f"[PROGRESS-PART] {i}/{total} {fname} -> pred: {pred_name} (#{pred_idx}), gt: {gt_name} (#{gt_idx}), "
                f"top-1_correct={correct_top1}, top-5_correct={correct_top5} | "
                f"running top-1: {running_top1*100:.2f}% | running top-5: {running_top5*100:.2f}%"
            )
            print(msg, flush=True)
            write_log_line(log_path, msg)
        else:
            msg = f"[PROGRESS-PART] Processed {i}/{total}: {fname}"
            print(msg, flush=True)
            write_log_line(log_path, msg)

    t1 = time.time()
    elapsed = t1 - t0
    info_msg = f"[INFO] Ran {len(batch_inputs)} images through partitioned models on CPU in {elapsed:.2f}s"
    print(info_msg)
    write_log_line(log_path, info_msg)
    return last_preds, elapsed, last_top5 if last_top5 else None


def run_onnx_inference_cpu_partitioned_three(
    p0_model_path: str,
    p1_model_path: str,
    p2_model_path: str,
    batch_inputs: List[Tuple[str, np.ndarray]],
    gt_lookup: Optional[Dict[str, int]] = None,
    classes: Optional[List[str]] = None,
    log_path: Optional[str] = None,
) -> Tuple[Dict[str, int], float, Optional[Dict[str, List[int]]]]:
    """
    Run three ONNX models on CPU in sequence (partitioned pipeline):
    - Input goes to p0 (partition 0)
    - p0's output is fed into p1 (partition 1)
    - p1's output is fed into p2 (partition 2)
    - Final logits are taken from p2's output
    Returns (predictions_by_filename, elapsed_seconds, top5_indices_by_filename)
    """
    for path, label in [(p0_model_path, "p0"), (p1_model_path, "p1"), (p2_model_path, "p2")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {label} not found: {path}")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
    sess0 = ort.InferenceSession(p0_model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]) 
    sess1 = ort.InferenceSession(p1_model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]) 
    sess2 = ort.InferenceSession(p2_model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]) 

    # IO info for p0
    p0_inputs = sess0.get_inputs(); p0_outputs = sess0.get_outputs()
    p0_in_name = p0_inputs[0].name if p0_inputs else "input"
    p0_out_name = p0_outputs[0].name if p0_outputs else None
    p0_in_shape = p0_inputs[0].shape if p0_inputs else None

    # Determine layout for p0 input
    layout0 = "NCHW"
    if isinstance(p0_in_shape, (list, tuple)) and len(p0_in_shape) == 4:
        c_dim = p0_in_shape[1]; last_dim = p0_in_shape[3]
        if last_dim == 3 or (isinstance(last_dim, str) and str(last_dim).upper() in ("C", "CHANNEL", "CHANNELS")):
            layout0 = "NHWC"
        elif c_dim == 3 or (isinstance(c_dim, str) and str(c_dim).upper() in ("C", "CHANNEL", "CHANNELS")):
            layout0 = "NCHW"
    print(f"[INFO] P0 IO - input_name={p0_in_name}, output_name={p0_out_name}, input_shape={p0_in_shape}, layout={layout0}")
    write_log_line(log_path, f"[INFO] P0 IO - input_name={p0_in_name}, output_name={p0_out_name}, input_shape={p0_in_shape}, layout={layout0}")

    # IO info for p1
    p1_inputs = sess1.get_inputs(); p1_outputs = sess1.get_outputs()
    p1_in_name = p1_inputs[0].name if p1_inputs else "input"
    p1_out_name = p1_outputs[0].name if p1_outputs else None
    p1_in_shape = p1_inputs[0].shape if p1_inputs else None
    print(f"[INFO] P1 IO - input_name={p1_in_name}, output_name={p1_out_name}, input_shape={p1_in_shape}")
    write_log_line(log_path, f"[INFO] P1 IO - input_name={p1_in_name}, output_name={p1_out_name}, input_shape={p1_in_shape}")

    # IO info for p2
    p2_inputs = sess2.get_inputs(); p2_outputs = sess2.get_outputs()
    p2_in_name = p2_inputs[0].name if p2_inputs else "input"
    p2_out_name = p2_outputs[0].name if p2_outputs else None
    p2_in_shape = p2_inputs[0].shape if p2_inputs else None
    print(f"[INFO] P2 IO - input_name={p2_in_name}, output_name={p2_out_name}, input_shape={p2_in_shape}")
    write_log_line(log_path, f"[INFO] P2 IO - input_name={p2_in_name}, output_name={p2_out_name}, input_shape={p2_in_shape}")

    last_preds: Dict[str, int] = {}
    last_top5: Dict[str, List[int]] = {}

    t0 = time.time()
    total = len(batch_inputs)
    correct_so_far_top1 = 0
    correct_so_far_top5 = 0

    for i, (fname, inp) in enumerate(batch_inputs, start=1):
        # p0
        feed_inp0 = np.transpose(inp, (0, 2, 3, 1)) if layout0 == "NHWC" else inp
        p0_out = sess0.run([p0_out_name] if p0_out_name else None, {p0_in_name: feed_inp0})[0]

        # Align p0_out -> p1 input
        inter1 = p0_out
        try:
            if isinstance(p1_in_shape, (list, tuple)) and len(p1_in_shape) > 0:
                want_batch = p1_in_shape[0]
                if want_batch == 1 and (inter1.ndim == len(p1_in_shape) - 1):
                    inter1 = np.expand_dims(inter1, axis=0)
                elif want_batch in (None, 'N', 'n', 'batch', 'Batch', 'B'):
                    pass
                if (p1_in_shape[0] in (None,) and inter1.ndim == len(p1_in_shape) + 1 and inter1.shape[0] == 1):
                    inter1 = np.squeeze(inter1, axis=0)
        except Exception:
            pass
        p1_out = sess1.run([p1_out_name] if p1_out_name else None, {p1_in_name: inter1})[0]

        # Align p1_out -> p2 input
        inter2 = p1_out
        try:
            if isinstance(p2_in_shape, (list, tuple)) and len(p2_in_shape) > 0:
                want_batch2 = p2_in_shape[0]
                if want_batch2 == 1 and (inter2.ndim == len(p2_in_shape) - 1):
                    inter2 = np.expand_dims(inter2, axis=0)
                elif want_batch2 in (None, 'N', 'n', 'batch', 'Batch', 'B'):
                    pass
                if (p2_in_shape[0] in (None,) and inter2.ndim == len(p2_in_shape) + 1 and inter2.shape[0] == 1):
                    inter2 = np.squeeze(inter2, axis=0)
        except Exception:
            pass
        p2_out = sess2.run([p2_out_name] if p2_out_name else None, {p2_in_name: inter2})[0]

        logits = p2_out
        if logits.ndim == 2:
            logits_arr = logits[0]
        else:
            logits_arr = logits

        pred_idx = int(np.argmax(logits_arr))
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
            running_top1 = correct_so_far_top1 / i
            running_top5 = correct_so_far_top5 / i
            pred_name = classes[pred_idx] if classes and 0 <= pred_idx < len(classes) else str(pred_idx)
            gt_name = classes[gt_idx] if (classes and gt_idx is not None and 0 <= gt_idx < len(classes)) else str(gt_idx)
            msg = (
                f"[PROGRESS-PART-3] {i}/{total} {fname} -> pred: {pred_name} (#{pred_idx}), gt: {gt_name} (#{gt_idx}), "
                f"top-1_correct={correct_top1}, top-5_correct={correct_top5} | "
                f"running top-1: {running_top1*100:.2f}% | running top-5: {running_top5*100:.2f}%"
            )
            print(msg, flush=True)
            write_log_line(log_path, msg)
        else:
            msg = f"[PROGRESS-PART-3] Processed {i}/{total}: {fname}"
            print(msg, flush=True)
            write_log_line(log_path, msg)

    t1 = time.time()
    elapsed = t1 - t0
    info_msg = f"[INFO] Ran {len(batch_inputs)} images through partitioned models (p0->p1->p2) on CPU in {elapsed:.2f}s"
    print(info_msg)
    write_log_line(log_path, info_msg)
    return last_preds, elapsed, last_top5 if last_top5 else None


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


def run_resnet_cpu_p0_then_npu_p1(
    p0_model_path: str,
    npu_p1_o_path: str,
    batch_inputs_pre: List[Tuple[str, np.ndarray]],
    gt_lookup: Optional[Dict[str, int]] = None,
    classes: Optional[List[str]] = None,
    log_path: Optional[str] = None,
    npu_id: int = 0,
    ref_p1_onnx_path: Optional[str] = None,
) -> Tuple[Dict[str, int], float, Optional[Dict[str, List[int]]]]:
    """
    Run ResNet50 partitioned: CPU p0 ONNX, then NPU p1.o.
    - p0_model_path: CPU ONNX path for partition 0.
    - npu_p1_o_path: Neubla .o path for partition 1.
    - batch_inputs_pre: list of (filename, preprocessed_input[NCHW float32]).
    - ref_p1_onnx_path: reference ONNX used to compile p1.o (to validate input/output sizes)
    The p0 output is quantized to uint8 before sending to NPU.
    If NPU output is a feature (e.g., 2048), finish FC on CPU using quantized params; if it's 1000, treat as logits.
    Returns (pred_by_filename, elapsed_seconds, top5_indices_by_filename)
    """
    if npu_mod is None:
        raise RuntimeError("npu module not available; cannot run NPU evaluation")

    if not os.path.exists(p0_model_path):
        raise FileNotFoundError(f"P0 ONNX not found: {p0_model_path}")
    if not os.path.exists(npu_p1_o_path):
        raise FileNotFoundError(f"NPU P1 .o not found: {npu_p1_o_path}")

    # Prepare CPU p0 session
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
    p0_sess = ort.InferenceSession(p0_model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]) 
    p0_inputs = p0_sess.get_inputs()
    p0_outputs = p0_sess.get_outputs()
    p0_in_name = p0_inputs[0].name if p0_inputs else "input"
    p0_out_name = p0_outputs[0].name if p0_outputs else None
    p0_in_shape = p0_inputs[0].shape if p0_inputs else None

    # Layout heuristic for p0
    layout = "NCHW"
    if isinstance(p0_in_shape, (list, tuple)) and len(p0_in_shape) == 4:
        c_dim = p0_in_shape[1]
        last_dim = p0_in_shape[3]
        if last_dim == 3 or (isinstance(last_dim, str) and str(last_dim).upper() in ("C", "CHANNEL", "CHANNELS")):
            layout = "NHWC"
        elif c_dim == 3 or (isinstance(c_dim, str) and str(c_dim).upper() in ("C", "CHANNEL", "CHANNELS")):
            layout = "NCHW"
    io_msg = f"[INFO] P0 IO - input_name={p0_in_name}, output_name={p0_out_name}, input_shape={p0_in_shape}, layout={layout}"
    print(io_msg)
    write_log_line(log_path, io_msg)

    # Inspect reference p1 ONNX (compiled source) to align IO sizes
    expected_in_elems = None
    expected_out_elems = None
    if ref_p1_onnx_path and os.path.exists(ref_p1_onnx_path):
        try:
            ref_sess = ort.InferenceSession(ref_p1_onnx_path, providers=["CPUExecutionProvider"])  
            ref_in = ref_sess.get_inputs()[0]
            ref_out = ref_sess.get_outputs()[0]
            ref_in_shape = ref_in.shape
            ref_out_shape = ref_out.shape
            # Compute static element counts if possible (ignore dynamic dims)
            def _num_elems(shape):
                if not isinstance(shape, (list, tuple)):
                    return None
                prod = 1
                for d in shape:
                    if isinstance(d, int) and d > 0:
                        prod *= d
                    else:
                        return None
                return prod
            expected_in_elems = _num_elems(ref_in_shape)
            expected_out_elems = _num_elems(ref_out_shape)
            ref_msg = f"[INFO] Ref P1 ONNX IO - in_name={ref_in.name}, in_shape={ref_in_shape}, out_name={ref_out.name}, out_shape={ref_out_shape}, in_elems={expected_in_elems}, out_elems={expected_out_elems}"
            print(ref_msg)
            write_log_line(log_path, ref_msg)
        except Exception as e:
            warn = f"[WARN] Failed to open ref P1 ONNX: {e}"
            print(warn)
            write_log_line(log_path, warn)

    # Prepare params for manual FC fallback (from full quantized ResNet model)
    fc_params = None
    try:
        _, _, params = npu_mod.resnet50_prepare_onnx_model(
            "../resnet/resnet50-0676ba61_opset12.neubla_u8_lwq_percentile.onnx"
        )
        fc_params = params
    except Exception as e:
        warn = f"[WARN] Could not prepare FC params for manual postprocess: {e}"
        print(warn)
        write_log_line(log_path, warn)

    # Initialize NPU driver with p1.o
    driver = None
    t0 = time.time()
    try:
        driver = npu_mod.initialize_driver(npu_id, npu_p1_o_path)

        preds: Dict[str, int] = {}
        top5_dict: Dict[str, List[int]] = {}
        total = len(batch_inputs_pre)
        correct_so_far_top1 = 0
        correct_so_far_top5 = 0

        for i, (fname, inp) in enumerate(batch_inputs_pre, start=1):
            # Run p0 on CPU
            feed_inp = np.transpose(inp, (0, 2, 3, 1)) if layout == "NHWC" else inp
            p0_out = p0_sess.run([p0_out_name] if p0_out_name else None, {p0_in_name: feed_inp})[0]

            # Quantize p0 output per-tensor to uint8
            inter = p0_out.astype(np.float32)
            vmin = float(np.min(inter))
            vmax = float(np.max(inter))
            if vmax <= vmin:
                scale_q = 1.0
                zp_q = 0.0
            else:
                scale_q = (vmax - vmin) / 255.0
                zp_q = -vmin / scale_q
            q = np.round(inter / scale_q + zp_q).astype(np.int32)
            q = np.clip(q, 0, 255).astype(np.uint8)

            # Validate/adjust size according to ref input
            q_flat = q.reshape(-1)
            if expected_in_elems is not None and q_flat.size != expected_in_elems:
                warn = f"[WARN] Quantized p0 output elems {q_flat.size} != expected p1 input elems {expected_in_elems}. Sending actual size."
                print(warn)
                write_log_line(log_path, warn)

            # Send to NPU
            input_bytes = q_flat.tobytes()
            send_count = q_flat.size if expected_in_elems is None else expected_in_elems
            raw_outputs = npu_mod.send_receive_data_npu(driver, input_bytes, send_count)

            # Interpret NPU output
            if not raw_outputs:
                raise RuntimeError("NPU returned no outputs")
            out0 = np.frombuffer(raw_outputs[0], dtype=np.uint8)

            logits_arr: np.ndarray
            used_manual_fc = False
            if out0.size == 1000:
                # Already logits
                logits_arr = out0
            elif (out0.size == 2048) and fc_params is not None:
                # Recognized pre-FC feature length for ResNet50; run quantized FC
                try:
                    scale = fc_params['/0/avgpool/GlobalAveragePool_output_0_scale'] * fc_params['0.fc.weight_scale']
                    zp_act = fc_params['/0/avgpool/GlobalAveragePool_output_0_zero_point']
                    zp_w = fc_params['0.fc.weight_zero_point']
                    scale_out = fc_params['/0/fc/Gemm_output_0_scale']
                    zp_out = fc_params['/0/fc/Gemm_output_0_zero_point']
                    weight_q = fc_params['0.fc.weight_quantized'].T.astype(np.int32)

                    x = out0.astype(np.int32)
                    out = np.matmul(x, weight_q)
                    out -= zp_act * np.sum(weight_q, axis=0)
                    out -= zp_w * np.sum(x, axis=0)
                    out += zp_act * zp_w
                    out = np.round(out * scale / scale_out) + zp_out
                    logits_arr = out.astype(np.uint8)
                    used_manual_fc = True
                except Exception as e:
                    warn = f"[WARN] Manual FC failed: {e}. Falling back to argmax on NPU output."
                    print(warn)
                    write_log_line(log_path, warn)
                    logits_arr = out0
            elif expected_out_elems is not None and out0.size == expected_out_elems and fc_params is not None:
                # Likely feature vector before FC (e.g., 2048). Run quantized FC using ref shape match
                try:
                    scale = fc_params['/0/avgpool/GlobalAveragePool_output_0_scale'] * fc_params['0.fc.weight_scale']
                    zp_act = fc_params['/0/avgpool/GlobalAveragePool_output_0_zero_point']
                    zp_w = fc_params['0.fc.weight_zero_point']
                    scale_out = fc_params['/0/fc/Gemm_output_0_scale']
                    zp_out = fc_params['/0/fc/Gemm_output_0_zero_point']
                    weight_q = fc_params['0.fc.weight_quantized'].T.astype(np.int32)

                    x = out0.astype(np.int32)
                    out = np.matmul(x, weight_q)
                    out -= zp_act * np.sum(weight_q, axis=0)
                    out -= zp_w * np.sum(x, axis=0)
                    out += zp_act * zp_w
                    out = np.round(out * scale / scale_out) + zp_out
                    logits_arr = out.astype(np.uint8)
                    used_manual_fc = True
                except Exception as e:
                    warn = f"[WARN] Manual FC failed: {e}. Falling back to argmax on NPU output."
                    print(warn)
                    write_log_line(log_path, warn)
                    logits_arr = out0
            elif out0.size == 2048:
                # Recognized feature size but no FC params available; avoid warning
                info_msg = "[INFO] NPU output size 2048 detected (pre-FC features); FC params unavailable – using argmax on features."
                print(info_msg)
                write_log_line(log_path, info_msg)
                logits_arr = out0
            else:
                # Unknown size; treat as logits for argmax but log it
                warn_msg = f"[WARN] Unexpected NPU output size {out0.size}; treating as logits for argmax."
                print(warn_msg)
                write_log_line(log_path, warn_msg)
                logits_arr = out0

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
                note = " | post=FC" if used_manual_fc else ""
                msg = (
                    f"[PROGRESS CPU->NPU] {i}/{total} {fname} -> pred: {pred_name} (#{pred_idx}), gt: {gt_name} (#{gt_idx}), "
                    f"top-1_correct={correct_top1}, top-5_correct={correct_top5} | "
                    f"running top-1: {running_top1*100:.2f}% | running top-5: {running_top5*100:.2f}%{note}"
                )
                print(msg, flush=True)
                write_log_line(log_path, msg)
            else:
                msg = f"[PROGRESS CPU->NPU] Processed {i}/{total}: {fname}"
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
    total_infers = len(batch_inputs_pre)
    info_msg = f"[INFO] Ran {total_infers} images through CPU p0 + NPU p1 in {elapsed:.2f}s"
    print(info_msg)
    write_log_line(log_path, info_msg)
    return preds, elapsed, top5_dict


def main():
    global CLI_ARGS
    # Prepare log file path for CPU part (kept for compatibility)
    log_filename = f"eval_resnet50_imagenet_cpu_{RUN_TS}.log"
    log_path = os.path.join(DEFAULT_LOG_DIR, log_filename)
    write_log_line(log_path, f"[START] ResNet50 ImageNet evaluation @ {RUN_TS}")

    # Log selected preprocessing/runtime options
    if CLI_ARGS is not None:
        sel = [
            f"pp_mode={CLI_ARGS.pp_mode}",
            f"interp={CLI_ARGS.interp}",
            f"rgb={CLI_ARGS.rgb}",
            f"scale_255={CLI_ARGS.scale_255}",
            f"imagenet_norm={CLI_ARGS.imagenet_norm}",
            f"mean={CLI_ARGS.mean}",
            f"std={CLI_ARGS.std}",
            f"short_side={CLI_ARGS.short_side}",
            f"limit_images={CLI_ARGS.limit_images}",
            f"npu_id={CLI_ARGS.npu_id}",
            f"cpu_npu_only={getattr(CLI_ARGS, 'cpu_npu_only', False)}",
        ]
        log_line = "[CONFIG] " + ", ".join(sel)
        print(log_line)
        write_log_line(log_path, log_line)
        preprocessor = build_preprocessor_from_args(CLI_ARGS)
    else:
        preprocessor = None

    print("[STEP 1] Building ground-truth from filenames...")
    write_log_line(log_path, "[STEP 1] Building ground-truth from filenames...")
    classes, filename_to_index = build_ground_truth(IMAGES_DIR, CLASSES_TXT, GROUND_TRUTH_CSV)

    print("[STEP 2] Loading and preprocessing images (once)...")
    write_log_line(log_path, "[STEP 2] Loading and preprocessing images (once)...")
    image_filenames = sorted(filename_to_index.keys())
    # Optionally limit the number of images
    selected = image_filenames
    if CLI_ARGS is not None and CLI_ARGS.limit_images is not None and CLI_ARGS.limit_images > 0:
        selected = image_filenames[: CLI_ARGS.limit_images]
        info_msg = f"[INFO] Limiting to {len(selected)} image(s)."
        print(info_msg)
        write_log_line(log_path, info_msg)
    preprocessed = load_images_preprocessed(IMAGES_DIR, selected, preprocessor=preprocessor)
    if not preprocessed:
        err = "[ERROR] No images preprocessed; aborting."
        print(err)
        write_log_line(log_path, err)
        return

    # ===== CPU-only evaluation (partitioned p0->p1->p2 ONNX on CPU) =====
    top1_cpu = float('nan'); top5acc_cpu = float('nan'); elapsed_cpu = float('nan')
    if not (CLI_ARGS is not None and getattr(CLI_ARGS, 'cpu_npu_only', False)):
        print("[STEP 3] Running partitioned ONNX models on CPU (p0 -> p1 -> p2, 1x per image) with live progress...")
        write_log_line(log_path, "[STEP 3] Running partitioned ONNX models on CPU (p0 -> p1 -> p2, 1x per image) with live progress...")
        p0_stage_path = CLI_ARGS.p0_model if (CLI_ARGS and CLI_ARGS.p0_model) else P0_MODEL_PATH
        p1_stage_path = CLI_ARGS.p1_model if (CLI_ARGS and CLI_ARGS.p1_model) else P1_MODEL_PATH
        p2_stage_path = CLI_ARGS.p2_model if (CLI_ARGS and CLI_ARGS.p2_model) else P2_MODEL_PATH
        preds_cpu, elapsed_cpu, top5_cpu = run_onnx_inference_cpu_partitioned_three(
            p0_stage_path,
            p1_stage_path,
            p2_stage_path,
            preprocessed,
            gt_lookup=filename_to_index,
            classes=classes,
            log_path=log_path,
        )

        print("[STEP 4] Computing CPU-only accuracy...")
        write_log_line(log_path, "[STEP 4] Computing CPU-only accuracy...")
        top1_cpu, top5acc_cpu = compute_accuracy(preds_cpu, filename_to_index, top5=top5_cpu)

        cpu_summary = [
            "",
            "===== CPU-only Evaluation Result =====",
            f"Images evaluated: {len(preprocessed)}",
            f"Model repeats: 1",
            f"Elapsed: {elapsed_cpu:.2f}s",
            f"Top-1 accuracy: {top1_cpu*100:.2f}%",
            f"Top-5 accuracy: {top5acc_cpu*100:.2f}%" if not np.isnan(top5acc_cpu) else "Top-5 accuracy: N/A",
        ]
        for line in cpu_summary:
            print(line)
            write_log_line(log_path, line)
    else:
        msg = "[STEP 3] Skipping CPU-only evaluation (--cpu-npu-only)"
        print(msg)
        write_log_line(log_path, msg)

    # ===== Hybrid evaluation: CPU p0 ONNX -> quantize -> NPU p1.o =====
    npu_o_path = CLI_ARGS.npu_o if (CLI_ARGS and CLI_ARGS.npu_o) else os.path.join(REPO_ROOT, "model_partitions", "partition1", "resnet50_npu_1", "p1.o")
    ref_p1_onnx = CLI_ARGS.ref_p1_onnx if (CLI_ARGS and CLI_ARGS.ref_p1_onnx) else os.path.join(REPO_ROOT, "model_partitions", "partition1", "resnet50_npu_1", "NeublaExecutionProvider_NEUBLA_0_0_resnet50_p1.onnx")
    print("[STEP 5] Running CPU p0 -> NPU p1 pipeline (1x per image) with live progress...")
    write_log_line(log_path, "[STEP 5] Running CPU p0 -> NPU p1 pipeline (1x per image) with live progress...")
    # Select p0 for hybrid stage (kept for backward compatibility)
    p0_path = CLI_ARGS.p0_model if (CLI_ARGS and CLI_ARGS.p0_model) else P0_MODEL_PATH
    preds_hyb, elapsed_hyb, top5_hyb = run_resnet_cpu_p0_then_npu_p1(
        p0_path,
        npu_o_path,
        preprocessed,
        gt_lookup=filename_to_index,
        classes=classes,
        log_path=log_path,
        npu_id=(CLI_ARGS.npu_id if CLI_ARGS is not None else 1),
        ref_p1_onnx_path=ref_p1_onnx,
    )

    print("[STEP 6] Computing CPU+NPU accuracy...")
    write_log_line(log_path, "[STEP 6] Computing CPU+NPU accuracy...")
    top1_hyb, top5acc_hyb = compute_accuracy(preds_hyb, filename_to_index, top5=top5_hyb)

    hyb_summary = [
        "",
        "===== CPU+NPU Evaluation Result =====",
        f"Images evaluated: {len(preprocessed)}",
        f"Model repeats: 1",
        f"Elapsed: {elapsed_hyb:.2f}s",
        f"Top-1 accuracy: {top1_hyb*100:.2f}%",
        f"Top-5 accuracy: {top5acc_hyb*100:.2f}%" if not np.isnan(top5acc_hyb) else "Top-5 accuracy: N/A",
    ]
    for line in hyb_summary:
        print(line)
        write_log_line(log_path, line)

    # Rewrite log so the final accuracies appear at the very top of the file
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            original = f.read()
        if not np.isnan(top1_cpu):
            cpu_header = (
                f"[FINAL] CPU-only Top-1: {top1_cpu*100:.2f}%\n"
                + (f"[FINAL] CPU-only Top-5: {top5acc_cpu*100:.2f}%\n" if not np.isnan(top5acc_cpu) else "[FINAL] CPU-only Top-5: N/A\n")
            )
        else:
            cpu_header = "[FINAL] CPU-only: SKIPPED (--cpu-npu-only)\n"
        header = (
            cpu_header
            + f"[FINAL] CPU+NPU Top-1: {top1_hyb*100:.2f}%\n"
            + (f"[FINAL] CPU+NPU Top-5: {top5acc_hyb*100:.2f}%\n" if not np.isnan(top5acc_hyb) else "[FINAL] CPU+NPU Top-5: N/A\n")
        )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(original)
    except Exception as e:
        print(f"[WARN] Failed to re-write log header with final accuracies: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate partitioned ResNet50 on ImageNet sample set (CPU-only and CPU->NPU), with configurable preprocessing and paths.")
    # Preprocessing options (aligned with eval_resnet50_imagenet.py)
    parser.add_argument("--pp-mode", choices=["letterbox", "resize", "center-crop"], default="center-crop", help="Preprocess mode: letterbox, resize, or center-crop")
    parser.add_argument("--interp", choices=["nearest", "linear", "cubic", "area", "lanczos4"], default="cubic", help="Interpolation method for resizing")
    parser.add_argument("--rgb", action="store_true", default=True, help="Use RGB color order (BGR->RGB). Default: True")
    parser.add_argument("--no-rgb", action="store_false", dest="rgb")
    parser.add_argument("--scale-255", action="store_true", default=True, help="Divide by 255.0 to scale to [0,1]. Default: True")
    parser.add_argument("--no-scale-255", action="store_false", dest="scale_255")
    parser.add_argument("--imagenet-norm", action="store_true", default=True, help="Apply ImageNet mean/std normalization. Default: True")
    parser.add_argument("--no-imagenet-norm", action="store_false", dest="imagenet_norm")
    parser.add_argument("--mean", type=str, default="", help="Comma-separated per-channel mean (RGB order if --rgb) e.g. 0.485,0.456,0.406")
    parser.add_argument("--std", type=str, default="", help="Comma-separated per-channel std e.g. 0.229,0.224,0.225")
    parser.add_argument("--short-side", type=int, default=256, help="Shorter-side length used before center-crop when pp-mode=center-crop")
    # Execution control
    parser.add_argument("--limit-images", type=int, default=0, help="If >0, evaluate only the first N images")
    # Paths for partitioned models and NPU
    parser.add_argument("--p0-model", type=str, default="", help="Override path to CPU partition p0 ONNX (first CPU stage; also used for CPU->NPU hybrid)")
    parser.add_argument("--p1-model", type=str, default="", help="Override path to CPU partition p1 ONNX (first CPU stage)")
    parser.add_argument("--p2-model", type=str, default="", help="Override path to CPU partition p2 ONNX (second CPU stage)")
    parser.add_argument("--npu-o", type=str, default="", help="Path to NPU p1 compiled object (.o)")
    parser.add_argument("--ref-p1-onnx", type=str, default="", help="Path to reference p1 ONNX that p1.o was compiled from")
    parser.add_argument("--npu-id", type=int, default=1, help="NPU device id")
    # Mode control
    parser.add_argument("--cpu-npu-only", action="store_true", help="Skip CPU-only partitions and run only the CPU->NPU hybrid pipeline")

    CLI_ARGS = parser.parse_args()
    main()
