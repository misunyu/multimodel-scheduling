import os
import json
import yaml
from typing import Dict, List, Tuple, Any, Optional, Set

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
    
    def save_sample_data(self, cpu_table, npu1_table, npu2_table, total_table):
        """
        Save profiling data to a sample file.
        
        Args:
            cpu_table, npu1_table, npu2_table, total_table: Tables with profiling data
            
        Returns:
            Path to the saved file
        """
        # Prepare data structure
        # Requirement: remove NPU2-related contents and rename NPU1 to GPU
        sample_data = {
            "cpu_data": [],
            "gpu_data": [],
            "total_data": []
        }
        
        # Extract CPU data
        for row in range(cpu_table.rowCount()):
            model = cpu_table.item(row, 0).text() if cpu_table.item(row, 0) else ""
            load = float(cpu_table.item(row, 1).text()) if cpu_table.item(row, 1) else 0.0
            infer = float(cpu_table.item(row, 2).text()) if cpu_table.item(row, 2) else 0.0
            
            sample_data["cpu_data"].append({
                "model": model,
                "load": load,
                "infer": infer
            })
        
        # Extract GPU data (previously NPU1)
        for row in range(npu1_table.rowCount()):
            model = npu1_table.item(row, 0).text() if npu1_table.item(row, 0) else ""
            load = float(npu1_table.item(row, 1).text()) if npu1_table.item(row, 1) else 0.0
            infer = float(npu1_table.item(row, 2).text()) if npu1_table.item(row, 2) else 0.0

            sample_data["gpu_data"].append({
                "model": model,
                "load": load,
                "infer": infer
            })
        
        # Do not extract/save NPU2 data (requirement)
        
        # Extract total data
        # Previous implementation only copied rows that were present in total_table.
        # If a model exists in CPU/GPU tables but didn't get a row in total_table,
        # it was missed. To be robust, collect the union of models from CPU table,
        # GPU(NPU1) table, and total_table, and then fill values preferring
        # total_table if present, otherwise falling back to per-device tables.

        def _safe_float(item):
            try:
                return float(item.text()) if item else 0.0
            except Exception:
                return 0.0

        # 1) Build model sets from each table
        models_cpu: Set[str] = set()
        for row in range(cpu_table.rowCount()):
            name_item = cpu_table.item(row, 0)
            model = name_item.text() if name_item else ""
            if model:
                models_cpu.add(model)

        models_gpu: Set[str] = set()
        for row in range(npu1_table.rowCount()):
            name_item = npu1_table.item(row, 0)
            model = name_item.text() if name_item else ""
            if model:
                models_gpu.add(model)

        models_total: Set[str] = set()
        total_cpu_map: Dict[str, float] = {}
        total_gpu_map: Dict[str, float] = {}
        for row in range(total_table.rowCount()):
            # Skip the last total summary row if present
            if row == total_table.rowCount() - 1:
                continue
            name_item = total_table.item(row, 0)
            model = name_item.text() if name_item else ""
            if not model:
                continue
            models_total.add(model)
            total_cpu_map[model] = _safe_float(total_table.item(row, 1))
            total_gpu_map[model] = _safe_float(total_table.item(row, 2))

        # 2) Build fallback sums from CPU/GPU tables
        cpu_sum_map: Dict[str, float] = {}
        for row in range(cpu_table.rowCount()):
            name_item = cpu_table.item(row, 0)
            model = name_item.text() if name_item else ""
            if not model:
                continue
            infer_val = _safe_float(cpu_table.item(row, 2))
            cpu_sum_map[model] = cpu_sum_map.get(model, 0.0) + infer_val

        gpu_sum_map: Dict[str, float] = {}
        for row in range(npu1_table.rowCount()):
            name_item = npu1_table.item(row, 0)
            model = name_item.text() if name_item else ""
            if not model:
                continue
            infer_val = _safe_float(npu1_table.item(row, 2))
            gpu_sum_map[model] = gpu_sum_map.get(model, 0.0) + infer_val

        # 3) Union of all models and emit total_data for each
        all_models: Set[str] = set()
        all_models.update(models_cpu)
        all_models.update(models_gpu)
        all_models.update(models_total)

        for model in sorted(all_models):
            cpu_infer = total_cpu_map.get(model, cpu_sum_map.get(model, 0.0))
            gpu_infer = total_gpu_map.get(model, gpu_sum_map.get(model, 0.0))
            sample_data["total_data"].append({
                "model": model,
                "cpu_infer": cpu_infer,
                "gpu_infer": gpu_infer
            })
        
        # Save to file
        filename = "sample_profiling_data.json"
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