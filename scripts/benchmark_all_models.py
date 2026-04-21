"""
Benchmark total latency (preprocess + inference + postprocess) for all 9 models.
Measures on GPU (CUDAExecutionProvider) with warmup iterations.
"""
import time
import sys
import os
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from image_processing import yolo_preprocess_local, resnet50_preprocess_local, image_preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models_onnx")
WARMUP = 10
ITERATIONS = 100

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

def get_session(model_path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model_path, sess_options=so, providers=PROVIDERS)
    print(f"  Providers: {sess.get_providers()}")
    return sess

def make_dummy_image(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

def benchmark_classification_model(model_path, name, input_size=224):
    """Benchmark a standard image classification model (NCHW, 224x224 or similar)."""
    print(f"\n[{name}] Loading...")
    sess = get_session(model_path)
    inp = sess.get_inputs()[0]
    inp_name = inp.name
    inp_shape = inp.shape

    # Determine layout and input size
    layout = "NCHW"
    if isinstance(inp_shape, (list, tuple)) and len(inp_shape) == 4:
        if inp_shape[3] == 3:
            layout = "NHWC"
        h_dim = inp_shape[2] if layout == "NCHW" else inp_shape[1]
        w_dim = inp_shape[3] if layout == "NCHW" else inp_shape[2]
        if isinstance(h_dim, int) and isinstance(w_dim, int):
            input_size_h, input_size_w = h_dim, w_dim
        else:
            input_size_h = input_size_w = input_size
    else:
        input_size_h = input_size_w = input_size

    print(f"  Input: {inp_name}, shape={inp_shape}, layout={layout}, size={input_size_h}x{input_size_w}")

    img = make_dummy_image()
    latencies = []

    for i in range(WARMUP + ITERATIONS):
        t0 = time.time()
        # Preprocess
        tensor = image_preprocess(img, input_size_w, input_size_h)
        if layout == "NHWC":
            tensor = np.transpose(tensor, (0, 2, 3, 1))
        # Inference
        out = sess.run(None, {inp_name: tensor})
        # Postprocess
        logits = out[0]
        logits_arr = np.squeeze(logits)
        max_idx = int(np.argmax(logits_arr))
        t1 = time.time()

        if i >= WARMUP:
            latencies.append((t1 - t0) * 1000.0)

    avg = np.mean(latencies)
    std = np.std(latencies)
    print(f"  [{name}] Avg latency: {avg:.2f} ± {std:.2f} ms ({ITERATIONS} iters)")
    return avg, std


def benchmark_yolov4(model_path):
    """Benchmark YOLOv4 detection model."""
    print(f"\n[YOLOv4] Loading...")
    sess = get_session(model_path)
    inputs = sess.get_inputs()

    img_input = None
    for inp in inputs:
        if isinstance(inp.shape, (list, tuple)) and len(inp.shape) == 4:
            img_input = inp
            break
    if img_input is None:
        img_input = inputs[0]

    input_name = img_input.name
    in_shape = img_input.shape
    # Detect layout: NCHW vs NHWC
    yolo_layout = "NCHW"
    if isinstance(in_shape, (list, tuple)) and len(in_shape) == 4:
        if in_shape[3] == 3 or (isinstance(in_shape[3], str) and '3' not in str(in_shape[1])):
            # Last dim is 3 -> NHWC
            if (isinstance(in_shape[3], int) and in_shape[3] == 3) or (isinstance(in_shape[1], int) and in_shape[1] != 3):
                yolo_layout = "NHWC"
                input_w = int(in_shape[2]) if isinstance(in_shape[2], int) else 416
                input_h = int(in_shape[1]) if isinstance(in_shape[1], int) else 416
            else:
                input_w = int(in_shape[3]) if isinstance(in_shape[3], int) else 416
                input_h = int(in_shape[2]) if isinstance(in_shape[2], int) else 416
        else:
            input_w = int(in_shape[3]) if isinstance(in_shape[3], int) else 416
            input_h = int(in_shape[2]) if isinstance(in_shape[2], int) else 416
    else:
        input_w = input_h = 416

    # Check for image_shape input
    image_shape_input = None
    image_shape_dtype = np.float32
    for inp in inputs:
        if 'image_shape' in inp.name:
            image_shape_input = inp
            t = (inp.type or '').lower()
            if 'int64' in t:
                image_shape_dtype = np.int64
            elif 'int32' in t:
                image_shape_dtype = np.int32
            break

    print(f"  Input: {input_name}, shape={in_shape}, layout={yolo_layout}, size={input_w}x{input_h}")

    img = make_dummy_image()
    latencies = []

    for i in range(WARMUP + ITERATIONS):
        t0 = time.time()
        # Preprocess
        input_tensor, meta = yolo_preprocess_local(img, (input_w, input_h))
        if yolo_layout == "NHWC":
            input_tensor = np.transpose(input_tensor, (0, 2, 3, 1))  # NCHW -> NHWC
        # Inference
        feeds = {input_name: input_tensor}
        if image_shape_input is not None:
            h0 = img.shape[0]
            w0 = img.shape[1]
            feeds[image_shape_input.name] = np.array([[h0, w0]], dtype=image_shape_dtype)
        out = sess.run(None, feeds)
        # Postprocess (basic NMS equivalent - just access output)
        for o in out:
            _ = o.shape
        t1 = time.time()

        if i >= WARMUP:
            latencies.append((t1 - t0) * 1000.0)

    avg = np.mean(latencies)
    std = np.std(latencies)
    print(f"  [YOLOv4] Avg latency: {avg:.2f} ± {std:.2f} ms ({ITERATIONS} iters)")
    return avg, std


def benchmark_gpt2(model_path):
    """Benchmark GPT-2 language model."""
    print(f"\n[GPT-2] Loading...")
    sess = get_session(model_path)
    inputs = sess.get_inputs()
    print(f"  Inputs: {[(i.name, i.shape, i.type) for i in inputs]}")

    # GPT-2 input: input_ids (int64), typically [batch, seq_len]
    input_name = inputs[0].name
    seq_len = 128
    dummy_ids = np.random.randint(0, 50257, (1, seq_len), dtype=np.int64)

    # Check for attention_mask or position_ids
    feeds_template = {input_name: dummy_ids}
    for inp in inputs[1:]:
        if 'attention_mask' in inp.name:
            feeds_template[inp.name] = np.ones((1, seq_len), dtype=np.int64)
        elif 'position_ids' in inp.name:
            feeds_template[inp.name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    latencies = []
    for i in range(WARMUP + ITERATIONS):
        t0 = time.time()
        out = sess.run(None, feeds_template)
        _ = out[0].shape
        t1 = time.time()
        if i >= WARMUP:
            latencies.append((t1 - t0) * 1000.0)

    avg = np.mean(latencies)
    std = np.std(latencies)
    print(f"  [GPT-2] Avg latency: {avg:.2f} ± {std:.2f} ms ({ITERATIONS} iters)")
    return avg, std


def benchmark_tinyllama(model_dir):
    """Benchmark TinyLLaMA-Chat model."""
    model_path = os.path.join(model_dir, "model.onnx")
    print(f"\n[TinyLLaMA-Chat] Loading...")
    sess = get_session(model_path)
    inputs = sess.get_inputs()
    print(f"  Inputs: {[(i.name, i.shape, i.type) for i in inputs]}")

    seq_len = 128
    past_seq_len = 0  # No past context
    feeds_template = {}
    for inp in inputs:
        if 'input_ids' in inp.name:
            feeds_template[inp.name] = np.random.randint(0, 32000, (1, seq_len), dtype=np.int64)
        elif 'attention_mask' in inp.name:
            feeds_template[inp.name] = np.ones((1, seq_len + past_seq_len), dtype=np.int64)
        elif 'position_ids' in inp.name:
            feeds_template[inp.name] = np.arange(past_seq_len, past_seq_len + seq_len, dtype=np.int64).reshape(1, -1)
        elif 'past_key_values' in inp.name:
            # Shape: [batch_size, num_heads, past_sequence_length, head_dim]
            shape = inp.shape
            num_heads = shape[1] if isinstance(shape[1], int) else 4
            head_dim = shape[3] if isinstance(shape[3], int) else 64
            feeds_template[inp.name] = np.zeros((1, num_heads, past_seq_len, head_dim), dtype=np.float32)

    latencies = []
    for i in range(WARMUP + ITERATIONS):
        t0 = time.time()
        out = sess.run(None, feeds_template)
        _ = out[0].shape
        t1 = time.time()
        if i >= WARMUP:
            latencies.append((t1 - t0) * 1000.0)

    avg = np.mean(latencies)
    std = np.std(latencies)
    print(f"  [TinyLLaMA-Chat] Avg latency: {avg:.2f} ± {std:.2f} ms ({ITERATIONS} iters)")
    return avg, std


def main():
    results = {}

    # Classification models
    cls_models = {
        "MNasNet":           ("mnasnet.onnx", 224),
        "ResNet50":          ("resnet50.onnx", 224),
        "ResNeXt50":         ("resnext50.onnx", 224),
        "ShuffleNet-v2-1.2": ("shufflenet-v2-12.onnx", 224),
        "SqueezeNet-v1.2":   ("squeezenet1.0-12.onnx", 224),
        "VGG19":             ("vgg19.onnx", 224),
    }

    for name, (fname, size) in cls_models.items():
        path = os.path.join(MODELS_DIR, fname)
        avg, std = benchmark_classification_model(path, name, size)
        results[name] = (avg, std)

    # YOLOv4
    yolo_path = os.path.join(MODELS_DIR, "yolov4.onnx")
    avg, std = benchmark_yolov4(yolo_path)
    results["YOLOv4"] = (avg, std)

    # GPT-2
    gpt2_path = os.path.join(MODELS_DIR, "gpt2.onnx")
    avg, std = benchmark_gpt2(gpt2_path)
    results["GPT-2"] = (avg, std)

    # TinyLLaMA-Chat
    tinyllama_dir = os.path.join(MODELS_DIR, "tiny-llama-chat-onnx")
    avg, std = benchmark_tinyllama(tinyllama_dir)
    results["TinyLLaMA-Chat"] = (avg, std)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Avg Latency (ms)':>18} {'Std (ms)':>10}")
    print("-" * 60)
    order = ["GPT-2", "TinyLLaMA-Chat", "MNasNet", "ResNet50", "ResNeXt50",
             "ShuffleNet-v2-1.2", "SqueezeNet-v1.2", "VGG19", "YOLOv4"]
    for name in order:
        avg, std = results[name]
        print(f"{name:<25} {avg:>15.2f} ms {std:>8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
