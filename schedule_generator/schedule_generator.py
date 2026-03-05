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
    
    def _is_llm(self, model_path: str) -> bool:
        name = os.path.basename(model_path).lower() if model_path else ""
        return ("gpt2" in name) or ("tiny-llama" in name)
    
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
        name_l = (getattr(input_tensor, 'name', '') or '').lower()
        t = str(getattr(input_tensor, 'type', '') or '').lower()

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
                    # For sequence-like dims, keep it small (e.g., 16) to avoid long runs
                    if any(k in name_l for k in ("seq", "token", "position", "input_ids", "attention")):
                        shape[i] = 16
                    else:
                        shape[i] = 128  # Default for other dimensions
        
        # NLP-aware fast path to avoid out-of-range Gather indices
        # Heuristics by common input names
        # Default vocab size for LLMs if not inferred elsewhere
        default_vocab = 32000
        # Helper to choose best integer dtype matching the declared type
        def int_dtype_for_type():
            if 'int64' in t:
                return np.int64
            if 'uint64' in t:
                return np.uint64
            if 'int16' in t:
                return np.int16
            if 'uint16' in t:
                return np.uint16
            if 'int8' in t:
                return np.int8
            if 'uint8' in t:
                return np.uint8
            if 'uint32' in t:
                return np.uint32
            # default
            return np.int32

        try:
            if any(k in name_l for k in ("input_ids", "token", "tokens")):
                # Token IDs must be within [0, vocab_size-1]
                low, high = 0, max(1, default_vocab) - 1
                return np.random.randint(low, high + 1, size=shape, dtype=int_dtype_for_type())
            if "attention" in name_l and "mask" in name_l:
                # Binary attention mask with leading ones (valid tokens) then zeros
                dtype = int_dtype_for_type()
                mask = np.zeros(shape, dtype=dtype)
                # Assume shape is [B, S] or [B, 1, 1, S]; handle common cases
                if len(shape) == 2:
                    B, S = shape
                    valid = min(S, 16)
                    mask[:, :valid] = 1
                elif len(shape) == 4:
                    B, H1, H2, S = shape
                    valid = min(S, 16)
                    mask[:, :, :, :valid] = 1
                else:
                    mask[...] = 1
                return mask
            if "position" in name_l and ("ids" in name_l or "index" in name_l or "indices" in name_l):
                dtype = int_dtype_for_type()
                if len(shape) == 2:
                    B, S = shape
                    base = np.arange(S, dtype=dtype)[None, :]
                    return np.tile(base, (B, 1))
                # fallback
                return np.zeros(shape, dtype=dtype)
        except Exception:
            # Fall through to generic generation on any error
            pass

        # Map ONNXRuntime type string (e.g., 'tensor(uint8)') to numpy dtype
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
    
    def _extract_llm_input_token_count(self, input_tensors: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """
        Heuristically extract input token count information from prepared input tensors.
        Looks for inputs like 'input_ids' or names containing 'token'.
        Returns a dict with keys: batch, seq, total, name.
        """
        try:
            # Priority to typical token id tensors
            preferred_keys = []
            for k in input_tensors.keys():
                kl = (k or "").lower()
                if "input_ids" in kl:
                    preferred_keys.append(k)
            if not preferred_keys:
                for k in input_tensors.keys():
                    kl = (k or "").lower()
                    if "token" in kl or "tokens" in kl:
                        preferred_keys.append(k)

            key = preferred_keys[0] if preferred_keys else None
            if key is None:
                return None

            arr = input_tensors[key]
            shape = list(getattr(arr, 'shape', []) or [])
            if not shape:
                return None

            # Heuristics: sequence length is last dim for token ids
            seq = int(shape[-1]) if isinstance(shape[-1], (int, np.integer)) else None
            batch = int(shape[0]) if len(shape) >= 2 and isinstance(shape[0], (int, np.integer)) else 1
            if seq is None:
                return None
            total = batch * seq
            return {"batch": batch, "seq": seq, "total": total, "name": key}
        except Exception:
            return None
    
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

            # If this looks like an LLM, log the input token count
            if self._is_llm(model_path):
                tok_info = self._extract_llm_input_token_count(input_tensors)
                if tok_info is not None:
                    self.log(f"[Tokens][CPU] {os.path.basename(model_path)}: input '{tok_info['name']}' -> batch={tok_info['batch']}, seq={tok_info['seq']}, total={tok_info['total']}")

            # Warm-up run (always exclude from average); log only for LLM
            t0w = time.time()
            session.run(None, input_tensors)
            warmup_ms = (time.time() - t0w) * 1000.0
            if self._is_llm(model_path):
                self.log(f"[Warmup][CPU] {os.path.basename(model_path)}: {warmup_ms:.1f} ms")

            # Timed runs (10x) average in ms
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

    def profile_model_gpu(self, model_path: str) -> Tuple[float, float, Dict[str, Any]]:
        """
        NVIDIA (TRT/CUDA) or Apple (CoreML) GPU profiling.
        Prefers NVIDIA providers on Linux, falling back to CPU.
        
        Args:
            model_path: Path to the ONNX model file
        Returns:
            Tuple of (load_time_ms, inference_time_ms, model_info)
        """
        model_info: Dict[str, Any] = {}

        # Measure model loading time (ONNX parse)
        start_time = time.time()
        _ = onnx.load(model_path)
        load_time = (time.time() - start_time) * 1000.0

        # Build requested providers list in priority order
        available = ort.get_available_providers()
        requested_providers = []
        
        if "CUDAExecutionProvider" in available:
            requested_providers.append("CUDAExecutionProvider")
        
        # Add CoreML only if no NVIDIA provider was added
        if not requested_providers:
            if "CoreMLExecutionProvider" in available:
                requested_providers.append("CoreMLExecutionProvider")
        
        # Always append CPUExecutionProvider as fallback for profiling purposes
        # but we should note if it actually fell back.
        requested_providers.append("CPUExecutionProvider")

        # Try to create session with providers one by one to avoid noisy ORT internal fallbacks
        session = None
        sess_options = ort.SessionOptions()
        
        for provider in requested_providers:
            try:
                # Try only the current provider.
                # Note: ORT still prints some error messages to stderr even if we catch exceptions here
                # because some errors happen deep inside the shared library loading code.
                session = ort.InferenceSession(model_path, sess_options, providers=[provider])
                active_providers = session.get_providers()
                if provider in active_providers:
                    break
                if provider == "CPUExecutionProvider":
                    break
            except Exception:
                # Silently try next provider in priority list
                continue

        if session is None:
            # Last resort fallback if everything failed (should not happen as CPU is in list)
            session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        
        # Log ORT info
        active_providers = session.get_providers()
        self.log(f"[ORT] {os.path.basename(model_path)} available={available} requested={requested_providers} active={active_providers}")

        # Store active providers in model_info
        try:
            model_info["ort_active_providers"] = active_providers
        except Exception:
            pass

        # Prepare inputs
        input_tensors = {}
        for input_tensor in session.get_inputs():
            input_tensors[input_tensor.name] = self.get_dummy_input(input_tensor)

        # If this looks like an LLM, log the input token count
        if self._is_llm(model_path):
            tok_info = self._extract_llm_input_token_count(input_tensors)
            if tok_info is not None:
                self.log(f"[Tokens][GPU] {os.path.basename(model_path)}: input '{tok_info['name']}' -> batch={tok_info['batch']}, seq={tok_info['seq']}, total={tok_info['total']}")

        # Warm-up (exclude from average); log only for LLM
        t0w = time.time()
        session.run(None, input_tensors)
        warmup_ms = (time.time() - t0w) * 1000.0
        if self._is_llm(model_path):
            self.log(f"[Warmup][GPU] {os.path.basename(model_path)}: {warmup_ms:.1f} ms")

        # Timed runs
        num_runs = 10
        t0 = time.time()
        for _ in range(num_runs):
            session.run(None, input_tensors)
        t1 = time.time()
        infer_time = (t1 - t0) * 1000.0 / num_runs

        model_info["path"] = model_path
        model_info["device"] = "GPU"
        model_info["load_time_ms"] = load_time
        model_info["inference_time_ms"] = infer_time

        return load_time, infer_time, model_info

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