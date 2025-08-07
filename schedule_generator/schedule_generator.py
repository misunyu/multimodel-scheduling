import os
import time
import numpy as np
import onnx
import onnxruntime as ort

from typing import List, Tuple, Dict, Any, Optional

# Custom operation prefixes for detection
CUSTOM_OP_PREFIXES = ["com.neubla"]

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
            input_tensor: ONNX tensor info
            
        Returns:
            Numpy array with appropriate shape and data type
        """
        shape = [self.safe_shape_value(s) for s in input_tensor.shape]
        
        # Handle dynamic dimensions with reasonable defaults
        for i, dim in enumerate(shape):
            if not isinstance(dim, int) or dim <= 0:
                if i == 0:  # Batch dimension
                    shape[i] = 1
                elif "height" in input_tensor.name.lower() or "h" == input_tensor.name.lower():
                    shape[i] = 224  # Common image height
                elif "width" in input_tensor.name.lower() or "w" == input_tensor.name.lower():
                    shape[i] = 224  # Common image width
                else:
                    shape[i] = 128  # Default for other dimensions
        
        # Create appropriate numpy array based on data type
        if input_tensor.type == "FLOAT":
            return np.random.rand(*shape).astype(np.float32)
        elif input_tensor.type == "INT32":
            return np.random.randint(0, 10, size=shape).astype(np.int32)
        elif input_tensor.type == "INT64":
            return np.random.randint(0, 10, size=shape).astype(np.int64)
        elif input_tensor.type == "BOOL":
            return np.random.choice([True, False], size=shape)
        else:
            # Default to float32 for other types
            return np.random.rand(*shape).astype(np.float32)
    
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
        Profile a model on NPU.
        
        Args:
            o_path: Path to the .o model file
            label: NPU label (e.g., "NPU1", "NPU2")
            
        Returns:
            Tuple of (load_time_ms, inference_time_ms, model_info)
        """
        # This is a placeholder for actual NPU profiling
        # In a real implementation, this would use the appropriate NPU driver
        
        # Simulate load and inference times
        load_time = np.random.uniform(5, 15)  # Random time between 5-15ms
        inference_time = np.random.uniform(2, 10)  # Random time between 2-10ms
        
        model_info = {
            "path": o_path,
            "device": label,
            "load_time_ms": load_time,
            "inference_time_ms": inference_time
        }
        
        return load_time, inference_time, model_info
    
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