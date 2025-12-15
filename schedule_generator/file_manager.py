import os
import json
import yaml
from typing import Dict, List, Tuple, Any, Optional, Set
from PyQt5.QtCore import Qt

class FileManager:
    """
    Class for managing files and model data.
    Handles file operations, model collection, and configuration management.
    """
    
    def __init__(self, log_callback=None):
        """
        Initialize the file manager.
        
        Args:
            log_callback: Function to call for logging messages
        """
        self.log_callback = log_callback
        
    def log(self, message):
        """Log a message using the callback if available."""
        if self.log_callback:
            self.log_callback(message)
    
    def collect_model_files(self, selected_paths):
        """
        Collect ONNX and O files from selected paths.
        
        Args:
            selected_paths: List of selected file or directory paths
            
        Returns:
            Tuple of (onnx_files, o_files) lists
        """
        onnx_files = []
        o_files = []

        for path in selected_paths:
            if os.path.isdir(path):
                # If it's a directory, walk through it
                for dirpath, _, filenames in os.walk(path):
                    for f in filenames:
                        full_path = os.path.join(dirpath, f)
                        if f.endswith(".onnx"):
                            onnx_files.append(full_path)
                        elif f.endswith(".o"):
                            o_files.append(full_path)
            elif os.path.isfile(path):
                # If it's a file, check its extension
                if path.endswith(".onnx"):
                    onnx_files.append(path)
                elif path.endswith(".o"):
                    o_files.append(path)
                    
        return onnx_files, o_files
    
    def get_selected_paths(self, model_tree_view, fs_model, root_folder):
        """
        Get selected paths from the model tree view.
        
        Args:
            model_tree_view: QTreeView containing the model tree
            fs_model: QFileSystemModel for the tree view
            root_folder: Root folder path
            
        Returns:
            List of selected file or directory paths
        """
        selected_indices = model_tree_view.selectedIndexes()
        selected_paths = []

        # If no selection, log a message and return
        if not selected_indices:
            self.log("[Warning] No items selected in the model tree. Please select folders or files to profile.\n")
            return []

        # Get file paths from selected indices (only column 0 to avoid duplicates)
        for index in selected_indices:
            if index.column() == 0:  # Only process column 0 to avoid duplicates
                file_path = fs_model.filePath(index)
                selected_paths.append(file_path)

                # Log selected items
                rel_path = os.path.relpath(file_path, root_folder)
                self.log(f"[Selected] {rel_path}")

        self.log("")  # Add empty line for readability
        return selected_paths
    
    def get_models_from_selection(self, model_tree_view, fs_model, root_folder):
        """
        Get model names from selected items in the tree view.
        
        Args:
            model_tree_view: QTreeView containing the model tree
            fs_model: QFileSystemModel for the tree view
            root_folder: Root folder path
            
        Returns:
            List of model names
        """
        selected_indices = model_tree_view.selectedIndexes()
        
        # Filter to only include column 0 indices to avoid duplicates
        selected_indices = [idx for idx in selected_indices if idx.column() == 0]
        
        if not selected_indices:
            self.log("[Warning] No items selected. Please select models to process.")
            return []
        
        # Get file paths from selected indices
        selected_paths = self._get_selected_file_paths(selected_indices, fs_model)
        
        # Extract model names from paths
        return self._extract_model_names(selected_paths, root_folder)
    
    def _get_selected_file_paths(self, selected_indices, fs_model):
        """
        Get file paths from selected indices.
        
        Args:
            selected_indices: List of selected QModelIndex objects
            fs_model: QFileSystemModel for the tree view
            
        Returns:
            List of file paths
        """
        return [fs_model.filePath(idx) for idx in selected_indices]
    
    def _extract_model_names(self, selected_paths, root_folder):
        """
        Extract model names from selected paths.
        
        Args:
            selected_paths: List of selected file or directory paths
            root_folder: Root folder path
            
        Returns:
            List of model names
        """
        model_dirs = set()
        
        for path in selected_paths:
            self._extract_model_from_directory(path, model_dirs)
        
        return self._prepare_model_list(model_dirs)
    
    def _extract_model_from_directory(self, path, model_dirs):
        """
        Extract model name from a directory path.
        
        Args:
            path: Directory or file path
            model_dirs: Set to store model directories
            
        Returns:
            None (updates model_dirs set)
        """
        if os.path.isdir(path):
            # Check if this is a model directory (contains a "model" subdirectory)
            model_subdir = os.path.join(path, "model")
            if os.path.isdir(model_subdir):
                model_dirs.add(path)
            else:
                # Check subdirectories
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        self._extract_model_from_directory(item_path, model_dirs)
        elif os.path.isfile(path):
            # If it's a file, check if its parent directory is a model directory
            parent_dir = os.path.dirname(path)
            if os.path.basename(parent_dir) == "model":
                model_dir = os.path.dirname(parent_dir)
                model_dirs.add(model_dir)
    
    def _prepare_model_list(self, model_dirs):
        """
        Prepare a list of model names from model directories.
        
        Args:
            model_dirs: Set of model directory paths
            
        Returns:
            List of model names
        """
        models = []
        
        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            models.append(model_name)
            
        # Sort models alphabetically
        models.sort()
        
        # Log the models
        self._log_model_list(models)
        
        return models
    
    def _log_model_list(self, models):
        """
        Log the list of models.
        
        Args:
            models: List of model names
            
        Returns:
            None
        """
        if not models:
            self.log("[Info] No models found in selection.")
        else:
            self.log(f"[Info] Found {len(models)} models: {', '.join(models)}")
    

    def load_device_settings(self, device_settings_file):
        """
        Load device settings from a YAML file.
        
        Args:
            device_settings_file: Path to the device settings file
            
        Returns:
            Dictionary of device settings
        """
        device_settings = {}
        
        try:
            if os.path.isfile(device_settings_file):
                with open(device_settings_file, 'r') as f:
                    device_settings = yaml.safe_load(f)
                self.log(f"[Info] Loaded device settings from {device_settings_file}")
            else:
                self.log(f"[Warning] Device settings file not found: {device_settings_file}")
                # Create default settings
                device_settings = {
                    "devices": {
                        "CPU": {"enabled": True},
                        "NPU1": {"enabled": True},
                        "NPU2": {"enabled": True}
                    }
                }
        except Exception as e:
            self.log(f"[Error] Failed to load device settings: {str(e)}")
            # Create default settings
            device_settings = {
                "devices": {
                    "CPU": {"enabled": True},
                    "NPU1": {"enabled": True},
                    "NPU2": {"enabled": True}
                }
            }
        
        return device_settings
    
    def save_sample_data(self, cpu_table, npu1_table, npu2_table, total_table,
                         extra_meta: Optional[Dict[str, Any]] = None,
                         llm_input_tokens: Optional[Dict[str, Dict[str, int]]] = None,
                         filename: Optional[str] = None):
        """
        Save profiling data to a sample file.
        
        Args:
            cpu_table, npu1_table, npu2_table, total_table: Tables with profiling data
            
        Returns:
            Path to the saved file
        """
        # Prepare data structure
        # Save CPU, GPU, and Total only. We no longer store NPU1/NPU2 fields.
        # Note: GPU data is sourced from the existing npu1_table in the UI.
        sample_data = {
            "cpu_data": [],   # list of {model, load, infer, tokens?}
            "gpu_data": [],  # list of {model, load, infer, tokens?}
            "total_data": []  # list of {model, cpu_infer, gpu_infer, cpu_tokens, gpu_tokens}
        }
        # Merge optional metadata (e.g., selected paths, root folder) for restoring UI state
        if extra_meta and isinstance(extra_meta, dict):
            try:
                for k, v in extra_meta.items():
                    sample_data[k] = v
            except Exception:
                # Ignore meta merge errors silently
                pass
        
        # Helper to parse a float from table item text
        def _safe_float_text(item):
            try:
                return float(item.text()) if item and item.text() not in (None, "", "-") else 0.0
            except Exception:
                return 0.0

        # Extract CPU data
        for row in range(cpu_table.rowCount()):
            model = cpu_table.item(row, 0).text() if cpu_table.item(row, 0) else ""
            load = _safe_float_text(cpu_table.item(row, 1))
            infer = _safe_float_text(cpu_table.item(row, 2))
            tokens = None
            tok_item = cpu_table.item(row, 3)
            # tokens/s column may be '-' for non-LLM
            if tok_item and tok_item.text() not in (None, "", "-"):
                try:
                    tokens = float(tok_item.text())
                except Exception:
                    tokens = None
            # LLM input token count captured from logs, indexed by display model name
            input_tokens = None
            try:
                if llm_input_tokens and isinstance(llm_input_tokens, dict):
                    input_tokens = llm_input_tokens.get("CPU", {}).get(model)
            except Exception:
                input_tokens = None

            entry = {"model": model, "load": load, "infer": infer}
            if tokens is not None:
                entry["tokens"] = tokens
            if input_tokens is not None:
                entry["input_tokens"] = int(input_tokens)
            sample_data["cpu_data"].append(entry)
        
        # Extract GPU data (from npu1_table in current UI)
        for row in range(npu1_table.rowCount()):
            model = npu1_table.item(row, 0).text() if npu1_table.item(row, 0) else ""
            load = _safe_float_text(npu1_table.item(row, 1))
            infer = _safe_float_text(npu1_table.item(row, 2))
            tokens = None
            tok_item = npu1_table.item(row, 3)
            if tok_item and tok_item.text() not in (None, "", "-"):
                try:
                    tokens = float(tok_item.text())
                except Exception:
                    tokens = None
            input_tokens = None
            try:
                if llm_input_tokens and isinstance(llm_input_tokens, dict):
                    input_tokens = llm_input_tokens.get("GPU", {}).get(model)
            except Exception:
                input_tokens = None

            entry = {"model": model, "load": load, "infer": infer}
            if tokens is not None:
                entry["tokens"] = tokens
            if input_tokens is not None:
                entry["input_tokens"] = int(input_tokens)
            sample_data["gpu_data"].append(entry)

        # NPU2 is no longer saved. Ignore npu2_table content on save.
        
        # Extract total data directly from the total table (skipping summary row)
        # Build quick lookup maps from per-device tables to enable fallback when FPS is not available
        cpu_infer_map: Dict[str, float] = {}
        for row in range(cpu_table.rowCount()):
            m = cpu_table.item(row, 0).text() if cpu_table.item(row, 0) else ""
            try:
                inf = float(cpu_table.item(row, 2).text()) if cpu_table.item(row, 2) and cpu_table.item(row, 2).text() not in (None, "", "-") else 0.0
            except Exception:
                inf = 0.0
            if m:
                cpu_infer_map[m] = inf

        gpu_infer_map: Dict[str, float] = {}
        for row in range(npu1_table.rowCount()):
            m = npu1_table.item(row, 0).text() if npu1_table.item(row, 0) else ""
            try:
                inf = float(npu1_table.item(row, 2).text()) if npu1_table.item(row, 2) and npu1_table.item(row, 2).text() not in (None, "", "-") else 0.0
            except Exception:
                inf = 0.0
            if m:
                gpu_infer_map[m] = inf

        for row in range(total_table.rowCount()):
            # Skip the last total summary row if present
            if row == total_table.rowCount() - 1:
                # Heuristic: if first column text equals 'Total', it's a summary row
                first = total_table.item(row, 0)
                if first and (first.text() or "").strip().lower() == "total":
                    continue
            name_item = total_table.item(row, 0)
            model = name_item.text() if name_item else ""
            if not model:
                continue
            # Read FPS from table (columns 1 and 2 now show FPS). Prefer hidden Qt.UserRole numeric.
            def _read_fps(col: int) -> float:
                it = total_table.item(row, col)
                if not it:
                    return 0.0
                try:
                    data = it.data(Qt.UserRole)
                    if isinstance(data, (int, float)):
                        return float(data)
                except Exception:
                    pass
                try:
                    txt = it.text()
                    return float(txt) if txt and txt != "-" else 0.0
                except Exception:
                    return 0.0

            cpu_fps_val = _read_fps(1)
            gpu_fps_val = _read_fps(2)
            # Convert FPS to ms for storage (backward compatible)
            cpu_infer = (1000.0 / cpu_fps_val) if cpu_fps_val > 0 else 0.0
            gpu_infer = (1000.0 / gpu_fps_val) if gpu_fps_val > 0 else 0.0
            # Fallback: if FPS not available (0), read per-device infer ms directly from the respective tables
            if cpu_infer == 0.0:
                try:
                    cpu_infer = float(cpu_infer_map.get(model, 0.0))
                except Exception:
                    cpu_infer = 0.0
            if gpu_infer == 0.0:
                try:
                    gpu_infer = float(gpu_infer_map.get(model, 0.0))
                except Exception:
                    gpu_infer = 0.0
            cpu_tokens = None
            gpu_tokens = None
            cpu_tok_item = total_table.item(row, 3)
            gpu_tok_item = total_table.item(row, 4)
            if cpu_tok_item and cpu_tok_item.text() not in (None, "", "-"):
                try:
                    cpu_tokens = float(cpu_tok_item.text())
                except Exception:
                    cpu_tokens = None
            if gpu_tok_item and gpu_tok_item.text() not in (None, "", "-"):
                try:
                    gpu_tokens = float(gpu_tok_item.text())
                except Exception:
                    gpu_tokens = None

            entry = {
                "model": model,
                "cpu_infer": cpu_infer,
                # Store only GPU naming (no NPU1 alias)
                "gpu_infer": gpu_infer,
                # Also store FPS values explicitly for forward compatibility/debugging
                "cpu_fps": round(cpu_fps_val, 2),
                "gpu_fps": round(gpu_fps_val, 2)
            }
            if cpu_tokens is not None:
                entry["cpu_tokens"] = cpu_tokens
            if gpu_tokens is not None:
                entry["gpu_tokens"] = gpu_tokens
            # Include input token counts if present
            try:
                if llm_input_tokens and isinstance(llm_input_tokens, dict):
                    cpu_inp = llm_input_tokens.get("CPU", {}).get(model)
                    gpu_inp = llm_input_tokens.get("GPU", {}).get(model)
                    if cpu_inp is not None:
                        entry["cpu_input_tokens"] = int(cpu_inp)
                    if gpu_inp is not None:
                        entry["gpu_input_tokens"] = int(gpu_inp)
            except Exception:
                pass
            sample_data["total_data"].append(entry)
        
        # Save to file
        # If a filename is provided by caller, use it; otherwise, use default name.
        filename = filename or "sample_profiling_data.json"
        try:
            with open(filename, 'w') as f:
                json.dump(sample_data, f, indent=2)
            self.log(f"[Info] Sample data saved to {filename}")
            return filename
        except Exception as e:
            self.log(f"[Error] Failed to save sample data: {str(e)}")
            return None
    
    def load_sample_data(self, filename=None):
        """
        Load sample profiling data from a file.
        
        Args:
            filename: Path to the sample data file (optional)
            
        Returns:
            Dictionary of sample data
        """
        if not filename:
            filename = "sample_profiling_data.json"
        
        try:
            if os.path.isfile(filename):
                with open(filename, 'r') as f:
                    sample_data = json.load(f)
                self.log(f"[Info] Loaded sample data from {filename}")
                return sample_data
            else:
                self.log(f"[Warning] Sample data file not found: {filename}")
                return None
        except Exception as e:
            self.log(f"[Error] Failed to load sample data: {str(e)}")
            return None