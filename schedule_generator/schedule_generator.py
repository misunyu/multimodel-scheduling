import os
import time
import numpy as np
import onnx
import onnxruntime as ort

from typing import List, Tuple, Dict, Any, Optional

# Custom operation prefixes for detection (Mobilint / vendor NPU ops)
CUSTOM_OP_PREFIXES = ["com.mobilint", "mobilint"]

class ModelProfiler:
    """
    Class for profiling ONNX and NPU models.
    Handles the profiling of models on CPU and NPU devices.
    """
    
    def __init__(self, log_callback=None):
        """
        Initialize the model profiler.
        
        Args:
            log_callback: Function to call for logging messages
        """
        self.log_callback = log_callback
        
    def log(self, message):
        """Log a message using the callback if available."""
        if self.log_callback:
            self.log_callback(message)
    
    def safe_shape_value(self, s):
        """Convert shape dimension to int if possible, otherwise keep as is."""
        try:
            return int(s)
        except:
            return s
    
    def get_dummy_input(self, input_tensor):
        """
        Generate dummy input data for a tensor based on its shape and type.
        
        Args:
            input_tensor: ONNX tensor info (onnxruntime.NodeArg)
            
        Returns:
            Numpy array with appropriate shape and data type
        """
        shape = [self.safe_shape_value(s) for s in input_tensor.shape]
        
        # Handle dynamic dimensions with reasonable defaults
        for i, dim in enumerate(shape):
            if not isinstance(dim, int) or dim <= 0:
                if i == 0:  # Batch dimension
                    shape[i] = 1
                elif "height" in input_tensor.name.lower() or input_tensor.name.lower() in ("h",):
                    shape[i] = 224  # Common image height
                elif "width" in input_tensor.name.lower() or input_tensor.name.lower() in ("w",):
                    shape[i] = 224  # Common image width
                else:
                    shape[i] = 128  # Default for other dimensions
        
        # Map ONNXRuntime type string (e.g., 'tensor(uint8)') to numpy dtype
        t = str(getattr(input_tensor, 'type', '') or '').lower()
        # Also handle raw ONNX ElementType names like 'FLOAT', 'INT32'
        def rand_float(dtype, low=0.0, high=1.0):
            arr = np.random.rand(*shape).astype(np.float32)
            if low != 0.0 or high != 1.0:
                arr = (arr * (high - low)) + low
            return arr.astype(dtype, copy=False)
        if 'uint8' in t:
            return np.random.randint(0, 256, size=shape, dtype=np.uint8)
        if 'int8' in t:
            return np.random.randint(-128, 128, size=shape, dtype=np.int8)
        if 'uint16' in t:
            return np.random.randint(0, 65536, size=shape, dtype=np.uint16)
        if 'int16' in t:
            return np.random.randint(-32768, 32768, size=shape, dtype=np.int16)
        if 'uint32' in t:
            return np.random.randint(0, np.iinfo(np.uint32).max, size=shape, dtype=np.uint32)
        if 'int32' in t:
            return np.random.randint(-2**31, 2**31 - 1, size=shape, dtype=np.int32)
        if 'uint64' in t:
            return np.random.randint(0, 2**32 - 1, size=shape, dtype=np.uint64)  # limit to 32-bit range for randint
        if 'int64' in t:
            # numpy randint upper bound limited for int64; use 32-bit range to avoid overflow
            return np.random.randint(-2**31, 2**31 - 1, size=shape, dtype=np.int64)
        if 'bool' in t:
            return np.random.choice([True, False], size=shape).astype(np.bool_)
        if 'float16' in t or 'fp16' in t:
            return rand_float(np.float16)
        if 'float' in t or t == 'float' or t == 'tensor(float)':
            return rand_float(np.float32)
        if 'double' in t or 'float64' in t:
            return rand_float(np.float64)
        
        # Fallback to float32
        return rand_float(np.float32)
    
    def profile_model_cpu(self, model_path: str) -> Tuple[float, float, Dict[str, Any]]:
        """
        Profile an ONNX model on CPU.
        
        Args:
            model_path: Path to the ONNX model file
            
        Returns:
            Tuple of (load_time_ms, inference_time_ms, model_info)
        """
        model_info = {}
        
        # Measure model loading time
        start_time = time.time()
        model = onnx.load(model_path)
        load_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Function to profile CPU inference
        def profile_cpu():
            # Create inference session
            session_options = ort.SessionOptions()
            session = ort.InferenceSession(model_path, session_options)
            
            # Prepare inputs
            input_tensors = {}
            for input_tensor in session.get_inputs():
                input_tensors[input_tensor.name] = self.get_dummy_input(input_tensor)
            
            # Warm-up run
            session.run(None, input_tensors)
            
            # Timed runs
            num_runs = 10
            start_time = time.time()
            for _ in range(num_runs):
                session.run(None, input_tensors)
            end_time = time.time()
            
            return (end_time - start_time) * 1000 / num_runs  # Average time in ms
        
        # Profile inference
        inference_time = profile_cpu()
        
        # Collect model info
        model_info["path"] = model_path
        model_info["load_time_ms"] = load_time
        model_info["inference_time_ms"] = inference_time
        
        return load_time, inference_time, model_info

    def profile_model_npu(self, o_path: str, label: str) -> Tuple[float, float, Dict[str, Any]]:
        """
        Profile a vision model on the Mobilint Aries NPU.

        Args:
            o_path: Model reference (path or name); the vision model is inferred
                    from it ("yolo*" -> yolo11s, "resnet*" -> resnet50).
            label: NPU label (informational).

        Returns:
            Tuple of (load_time_ms, inference_time_ms, model_info)

        NOTE: For full multi-device profiling (CPU/GPU/NPU + LLM tokens/sec) use
        the standalone `profile_models.py` tool, which is the canonical profiler.
        """
        from runtime.mobilint_vision import build_vision_npu

        path_lower = (o_path or "").lower()
        model_name = "yolo11s" if "yolo" in path_lower else "resnet50"
        h = w = 640 if model_name.startswith("yolo") else 224
        frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)

        model = None
        try:
            start_load = time.time()
            model = build_vision_npu(model_name, infer_mode="global8")
            load_time_ms = (time.time() - start_load) * 1000.0

            pre = model.preprocess(frame)
            model(pre)  # warmup
            start_infer = time.time()
            model(pre)
            infer_time_ms = (time.time() - start_infer) * 1000.0
        except Exception as e:
            self.log(f"[Error] {label}: {e}")
            raise
        finally:
            try:
                if model is not None:
                    model.dispose()
            except Exception:
                pass

        model_info = {"path": o_path, "device": label,
                      "load_time_ms": load_time_ms, "inference_time_ms": infer_time_ms}
        return load_time_ms, infer_time_ms, model_info

    def contains_custom_op(self, onnx_path: str) -> bool:
        """
        Check if an ONNX model contains custom operations.
        
        Args:
            onnx_path: Path to the ONNX model file
            
        Returns:
            True if the model contains custom operations, False otherwise
        """
        try:
            def check_custom_ops():
                model = onnx.load(onnx_path)
                for node in model.graph.node:
                    for prefix in CUSTOM_OP_PREFIXES:
                        if node.domain.startswith(prefix):
                            return True
                return False
            
            return check_custom_ops()
        except Exception as e:
            self.log(f"Error checking for custom ops in {onnx_path}: {str(e)}")
            return False