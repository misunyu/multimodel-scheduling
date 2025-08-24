import os
import sys
import time
import json
import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QDialog,
    QTreeView, QPlainTextEdit, QTableWidget, QAction,
    QFileSystemModel, QCheckBox, QTabWidget, QWidget,
    QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton,
    QHeaderView, QLabel, QAbstractItemView
)

from schedule_generator import (
    ModelProfiler, DataProcessor, UIComponents, FileManager
)
from image_processing import (
    yolo_preprocess_local,
    resnet50_preprocess_local,
    yolo_postprocess_cpu,
    yolo_postprocess_npu,
)

class ONNXProfilerApp(QMainWindow):
    """
    Main application class for the ONNX Profiler GUI.
    Integrates the various components for model profiling.
    """
    
    def __init__(self, target_device_file=None):
        """
        Initialize the application.
        
        Args:
            target_device_file (str, optional): Path to the target device information file.
                                               Defaults to None, which will use "target_device.yaml".
        """
        super().__init__()
        
        # Load UI from file
        uic.loadUi("schedule_generator_display.ui", self)
        
        # Initialize profiled_times and profiled_models attributes
        self.profiled_times = []
        self.profiled_models = []
        
        # Initialize component modules
        self.profiler = ModelProfiler(log_callback=self.log_message)
        self.data_processor = DataProcessor(log_callback=self.log_message)
        self.ui_components = UIComponents(log_callback=self.log_message)
        self.file_manager = FileManager(log_callback=self.log_message)
        
        # Find UI elements
        self.setup_ui_elements()
        
        # Connect signals
        self.connect_signals()
        
        # Set up file system model
        self.setup_file_system_model()
        
        # Default device settings file
        self.device_settings_file = target_device_file if target_device_file else "target_device.yaml"
        
        # Load device settings from file
        self.device_settings = self.file_manager.load_device_settings(self.device_settings_file)
        
        # Assignment results storage
        self.assignment_results = []
        
        # Flag to prevent multiple simultaneous profiling runs
        self._profiling_in_progress = False
    
    def setup_ui_elements(self):
        """Find and set up UI elements."""
        # Find main UI elements
        self.enable_npu2_checkbox = self.findChild(QCheckBox, "npu2_enable_checkbox")
        self.result_tabs = self.findChild(QTabWidget, "result_tab_widget")
        self.npu2_tab = self.findChild(QWidget, "npu2_tab")
        
        # Set up main layout
        main_layout = self.findChild(QHBoxLayout, "mainLayout")
        if main_layout:
            main_layout.setStretch(0, 3)
            main_layout.setStretch(1, 7)
        
        # Find input and control elements
        self.folder_input = self.findChild(QLineEdit, "folder_input")
        self.browse_button = self.findChild(QPushButton, "browse_button")
        self.profile_button = self.findChild(QPushButton, "profile_button")
        self.generate_static_button = self.findChild(QPushButton, "generate_static_button")
        self.generate_all_button = self.findChild(QPushButton, "generate_all_button")
        self.model_tree_view = self.findChild(QTreeView, "model_tree_view")
        self.log_output = self.findChild(QPlainTextEdit, "log_output")
        self.total_table = self.findChild(QTableWidget, "total_table")
        
        # Find table elements
        self.cpu_table = self.findChild(QTableWidget, "cpu_table")
        self.npu1_table = self.findChild(QTableWidget, "npu1_table")
        self.npu2_table = self.findChild(QTableWidget, "npu2_table")
        self.pre_post_table = self.findChild(QTableWidget, "pre_post_table")
        
        # Set up NPU2 visibility
        self.npu2_table.setVisible(self.enable_npu2_checkbox.isChecked())
        
        # Set up table headers
        for table in [self.cpu_table, self.npu1_table, self.npu2_table]:
            header = table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # Initialize Pre/Post table headers
        if self.pre_post_table:
            self.pre_post_table.clear()
            self.pre_post_table.setColumnCount(2)
            self.pre_post_table.setHorizontalHeaderLabels(["Function", "Avg (ms)"])
            self.pre_post_table.setRowCount(0)
            ph = self.pre_post_table.horizontalHeader()
            ph.setStretchLastSection(True)
            ph.setSectionResizeMode(0, QHeaderView.Stretch)
            ph.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        # Create and add legend label
        self.legend_label = QLabel()
        self.legend_label.setText(
            "<span style='background-color:#cce6ff;'>&nbsp;&nbsp;&nbsp;</span> CPU &nbsp;&nbsp;"
            "<span style='background-color:#ffffcc;'>&nbsp;&nbsp;&nbsp;</span> use NPU1 &nbsp;&nbsp;"
            "<span style='background-color:#ffd699;'>&nbsp;&nbsp;&nbsp;</span> use NPU2"
        )
        self.legend_label.setStyleSheet("font-size: 12px; padding: 2px;")
        
        right_layout = self.findChild(QVBoxLayout, "rightLayout")
        if right_layout:
            index = right_layout.indexOf(self.log_output)
            if index != -1:
                right_layout.insertWidget(index, self.legend_label)
        
        # Find additional buttons
        self.show_assignment_button = self.findChild(QPushButton, "show_assignment_button")
        self.load_sample_button = self.findChild(QPushButton, "load_sample_button")
        
        # Find menu actions
        self.actionLoad_Test_Data = self.findChild(QAction, "actionLoad_Test_Data")
        self.actionSave_Sample_Data = self.findChild(QAction, "actionSave_Sample_Data")
        self.actionSettings = self.findChild(QAction, "actionSettings")
    
    def connect_signals(self):
        """Connect UI signals to slots."""
        # Connect NPU2 checkbox
        def update_npu2_tab_enabled():
            index = self.result_tabs.indexOf(self.npu2_tab)
            if index != -1:
                self.result_tabs.setTabEnabled(index, self.enable_npu2_checkbox.isChecked())
        
        self.enable_npu2_checkbox.stateChanged.connect(update_npu2_tab_enabled)
        self.enable_npu2_checkbox.stateChanged.connect(
            lambda: self.npu2_table.setVisible(self.enable_npu2_checkbox.isChecked())
        )
        update_npu2_tab_enabled()
        
        # Connect buttons
        self.browse_button.clicked.connect(self.browse_folder)
        self.profile_button.clicked.connect(self.run_profiling)
        self.generate_static_button.clicked.connect(
            lambda: self.ui_components.highlight_deploy_results(
                self.total_table, self.profiled_times, self.profiled_models
            )
        )
        self.generate_all_button.clicked.connect(self.generate_all_combinations)
        
        # Connect additional buttons
        if self.show_assignment_button:
            self.show_assignment_button.clicked.connect(self.show_partition_assignment_dialog)
        
        if self.load_sample_button:
            self.load_sample_button.clicked.connect(self.load_sample_data)
        
        # Connect menu actions
        if self.actionLoad_Test_Data:
            self.actionLoad_Test_Data.triggered.connect(self.load_sample_data)
        
        if self.actionSave_Sample_Data:
            self.actionSave_Sample_Data.triggered.connect(self.save_sample_data)
        
        if self.actionSettings:
            self.actionSettings.triggered.connect(self.show_settings_dialog)
    
    def setup_file_system_model(self):
        """Set up the file system model for the tree view."""
        self.fs_model = QFileSystemModel()
        self.fs_model.setReadOnly(True)
        self.fs_model.setNameFilters(["*.onnx", "*.o"])
        self.fs_model.setNameFilterDisables(False)
        self.model_tree_view.setModel(self.fs_model)
        self.model_tree_view.setMinimumWidth(320)
        self.model_tree_view.header().setStretchLastSection(True)
        self.model_tree_view.header().setDefaultSectionSize(300)
        self.model_tree_view.setColumnWidth(1, 60)
        self.model_tree_view.setColumnHidden(2, True)
        self.model_tree_view.setColumnHidden(3, True)
        self.model_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)
        
        # Connect selection changed signal
        self.model_tree_view.selectionModel().selectionChanged.connect(self.handle_tree_selection_changed)
        
        # Set default folder
        default_folder = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(default_folder):
            default_folder = os.getcwd()
        
        self.folder_input.setText(default_folder)
        self.set_tree_root(default_folder)
        QTimer.singleShot(100, lambda: self.expand_parents_of_onnx_files(default_folder))
    
    def log_message(self, message):
        """Log a message to the output text area."""
        if self.log_output:
            self.log_output.appendPlainText(message)
            QApplication.processEvents()
    
    def browse_folder(self):
        """Browse for a folder to profile."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", os.getcwd())
        if folder:
            self.folder_input.setText(folder)
            self.set_tree_root(folder)
            QTimer.singleShot(100, lambda: self.expand_parents_of_onnx_files(folder))
    
    def handle_tree_selection_changed(self, selected, deselected):
        """
        Handle selection changes in the tree view.
        Only top-level folders (model names) can be toggled.
        Subfolders and their Size tabs are automatically deselected.
        
        Args:
            selected: QItemSelection of newly selected items
            deselected: QItemSelection of newly deselected items
        """
        # Process only if there are selected items
        if not selected.indexes():
            return
            
        # Get the root path and index
        root_path = self.folder_input.text().strip()
        root_index = self.fs_model.index(root_path)
        
        # Process each newly selected item
        for index in selected.indexes():
            # Get the row index regardless of column
            row_index = self.fs_model.index(self.fs_model.filePath(index), 0)
            
            # Check if this is a top-level folder (direct child of root)
            parent = row_index.parent()
            if parent == root_index:
                # This is a top-level folder, allow it to remain selected
                continue
            else:
                # This is a subfolder or its Size tab, deselect it
                self.model_tree_view.selectionModel().select(
                    index, 
                    self.model_tree_view.selectionModel().Deselect
                )
    
    def set_tree_root(self, folder):
        """Set the root folder for the tree view."""
        self.fs_model.setRootPath(folder)
        index = self.fs_model.index(folder)
        self.model_tree_view.setRootIndex(index)
    
    def expand_parents_of_onnx_files(self, root_folder):
        """Expand tree view items that contain ONNX or O files."""
        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                if f.endswith(".onnx") or f.endswith(".o"):
                    file_path = os.path.join(dirpath, f)
                    index = self.fs_model.index(file_path)
                    parent = index.parent()
                    while parent.isValid():
                        self.model_tree_view.expand(parent)
                        parent = parent.parent()
    
    def run_profiling(self):
        """Run profiling on selected models."""
        # Prevent multiple simultaneous executions
        if self._profiling_in_progress:
            return
        
        self._profiling_in_progress = True
        self.profile_button.setEnabled(False)  # Disable button during profiling
        
        try:
            root_folder = self.folder_input.text().strip()
            if not os.path.isdir(root_folder):
                return
            
            # Initialize UI for profiling
            self._initialize_profiling_ui()
            
            # Get selected paths
            selected_paths = self.file_manager.get_selected_paths(
                self.model_tree_view, self.fs_model, root_folder
            )
            
            if not selected_paths:
                return
            
            # Collect model files
            onnx_files, o_files = self.file_manager.collect_model_files(selected_paths)
            
            # Profile ONNX models
            valid_model_onnx = self._profile_onnx_models(onnx_files, root_folder)
            
            # Profile O models
            self._profile_o_models(o_files, root_folder)
            
            # Process profiling results
            if self.total_table:
                self._process_profiling_results(valid_model_onnx, root_folder)

            # At the end of profiling, compute and display pre/post average times
            self._profile_pre_post_avg_times()
        
        except Exception as e:
            self.log_message(f"[Error] Profiling failed: {str(e)}\n")
        finally:
            # Re-enable button and reset progress flag
            self._profiling_in_progress = False
            self.profile_button.setEnabled(True)
            self.log_message("[Complete] Profiling finished.\n")
    
    def _initialize_profiling_ui(self):
        """Initialize UI for profiling."""
        if self.log_output:
            self.log_output.clear()
            self.log_message("[Start] Profiling models...\n")
        
        # Clear previous results
        self.ui_components.init_table(self.cpu_table)
        self.ui_components.init_table(self.npu1_table)
        self.ui_components.init_table(self.npu2_table)
        
        # Reset Pre/Post table
        if hasattr(self, 'pre_post_table') and self.pre_post_table is not None:
            self.pre_post_table.clear()
            self.pre_post_table.setColumnCount(2)
            self.pre_post_table.setHorizontalHeaderLabels(["Function", "Avg (ms)"])
            self.pre_post_table.setRowCount(0)
        
        # Clear profiled data
        self.profiled_times = []
        self.profiled_models = []
    
    def _profile_onnx_models(self, onnx_files, root_folder):
        """Profile ONNX models and update CPU table."""
        valid_model_onnx = {}
        
        for path in onnx_files:
            try:
                if self.profiler.contains_custom_op(path):
                    self.log_message(f"[Skip] {path} contains custom ops\n")
                    continue
                
                load_ms, infer_ms, _ = self.profiler.profile_model_cpu(path)
                rel_path = os.path.relpath(path, root_folder)
                self.ui_components.insert_result_row(self.cpu_table, rel_path, load_ms, infer_ms)
                
                self.log_message(f"[CPU] {rel_path}")
                self.log_message(f"       Load: {load_ms:.1f} ms, Inference: {infer_ms:.1f} ms\n")
                
                parts = rel_path.split(os.sep)
                if len(parts) == 3 and parts[1] == "model" and parts[2].endswith(".onnx"):
                    model_key = parts[0]
                    if model_key not in valid_model_onnx:
                        valid_model_onnx[model_key] = [0.0, 0.0]
                    valid_model_onnx[model_key][0] += load_ms
                    valid_model_onnx[model_key][1] += infer_ms
            
            except Exception as e:
                self.log_message(f"[Error] Skipping {path}: {str(e)}\n")
        
        return valid_model_onnx
    
    def _profile_o_models(self, o_files, root_folder):
        """Profile O models and update NPU tables."""
        for path in o_files:
            name = os.path.relpath(path, root_folder)
            load_npu1, infer_npu1, _ = self.profiler.profile_model_npu(path, "NPU1")
            self.ui_components.insert_result_row(self.npu1_table, name, load_npu1, infer_npu1)
            
            self.log_message(f"[NPU1] {name}")
            self.log_message(f"       Load: {load_npu1:.1f} ms, Inference: {infer_npu1:.1f} ms\n")
            
            if self.enable_npu2_checkbox and self.enable_npu2_checkbox.isChecked():
                load_npu2, infer_npu2, _ = self.profiler.profile_model_npu(path, "NPU2")
                self.ui_components.insert_result_row(self.npu2_table, name, load_npu2, infer_npu2)
                
                self.log_message(f"[NPU2] {name}")
                self.log_message(f"       Load: {load_npu2:.1f} ms, Inference: {infer_npu2:.1f} ms\n")
    
    def _process_profiling_results(self, valid_model_onnx, root_folder):
        """Process profiling results and update total table."""
        # Initialize total table
        self.ui_components.initialize_total_table(self.total_table)
        
        # Process results using data processor
        all_models, cpu_infer_per_partition, npu1_load, npu1_infer, npu2_load, npu2_infer = \
            self.data_processor.process_profiling_results(
                valid_model_onnx, self.cpu_table, self.npu1_table, self.npu2_table
            )
        
        # Prepare profiled_times and profiled_models for highlight_deploy_results
        self.profiled_times = []
        self.profiled_models = []
        
        # Populate total table
        self.ui_components.populate_total_table(
            self.total_table, all_models, valid_model_onnx, 
            npu1_load, npu1_infer, npu2_load, npu2_infer, 
            cpu_infer_per_partition
        )
        
        # Calculate and display totals
        if self.total_table.rowCount() > 0:
            self._calculate_and_display_totals()
    
    def _calculate_and_display_totals(self):
        """Calculate and display total values in the total table."""
        # Calculate totals
        cpu_infer_total = 0.0
        npu1_load_total = 0.0
        npu1_infer_total = 0.0
        npu2_load_total = 0.0
        npu2_infer_total = 0.0
        
        for row in range(self.total_table.rowCount()):
            try:
                cpu_infer_total += float(self.total_table.item(row, 1).text())
                npu1_load_total += float(self.total_table.item(row, 2).text())
                npu1_infer_total += float(self.total_table.item(row, 3).text())
                npu2_load_total += float(self.total_table.item(row, 4).text())
                npu2_infer_total += float(self.total_table.item(row, 5).text())
            except:
                pass
        
        # Add total row
        self.ui_components.add_total_row(
            self.total_table, cpu_infer_total, npu1_load_total, 
            npu1_infer_total, npu2_load_total, npu2_infer_total
        )

    def _profile_pre_post_avg_times(self):
        """Run specified pre/post-processing functions 10 times with dummy inputs and display their average times in pre_post_table."""
        if not hasattr(self, 'pre_post_table') or self.pre_post_table is None:
            return

        # Prepare dummy images
        img_h, img_w = 720, 1280
        raw_img = (np.random.randint(0, 256, size=(img_h, img_w, 3), dtype=np.uint8))

        # Helper to time a callable 10 times
        def avg_time_ms(fn, *args, **kwargs):
            total = 0.0
            for _ in range(10):
                t0 = time.perf_counter()
                _ = fn(*args, **kwargs)
                t1 = time.perf_counter()
                total += (t1 - t0) * 1000.0
            return total / 10.0

        # Build dummy outputs for postprocess functions
        # For YOLO CPU postprocess: output[0] -> rows of [cx, cy, w, h, obj_conf, class_probs...]
        rows_cpu = 50
        cols_cpu = 85  # 4 + 1 + 80 class probs (typical YOLOv5), enough for indexing
        yolo_cpu_output0 = np.zeros((rows_cpu, cols_cpu), dtype=np.float32)
        # center x,y around input size 608 used inside image_processing, but any values ok
        yolo_cpu_output0[:, 0:4] = np.random.rand(rows_cpu, 4).astype(np.float32) * 608.0
        yolo_cpu_output0[:, 4] = np.random.rand(rows_cpu).astype(np.float32)  # object confidence
        yolo_cpu_output0[:, 5:] = np.random.rand(rows_cpu, cols_cpu - 5).astype(np.float32)
        yolo_cpu_output = [yolo_cpu_output0]

        # For YOLO NPU postprocess: output[0]=[left, top, right, bottom, conf], output[1]=class_ids
        rows_npu = 50
        left = np.random.rand(rows_npu).astype(np.float32) * 608.0
        top = np.random.rand(rows_npu).astype(np.float32) * 608.0
        right = left + np.random.rand(rows_npu).astype(np.float32) * 100.0 + 1.0
        bottom = top + np.random.rand(rows_npu).astype(np.float32) * 100.0 + 1.0
        conf = np.random.rand(rows_npu).astype(np.float32)
        yolo_npu_output0 = np.stack([left, top, right, bottom, conf], axis=1)
        yolo_npu_output1 = np.random.randint(0, 80, size=(rows_npu,), dtype=np.int32)
        yolo_npu_output = [yolo_npu_output0, yolo_npu_output1]

        # Compute averages
        results = []
        try:
            avg1 = avg_time_ms(yolo_preprocess_local, raw_img)
            results.append(("yolo_preprocess_local", avg1))
        except Exception as e:
            self.log_message(f"[Warn] yolo_preprocess_local timing failed: {e}")
        try:
            avg2 = avg_time_ms(resnet50_preprocess_local, raw_img)
            results.append(("resnet50_preprocess_local", avg2))
        except Exception as e:
            self.log_message(f"[Warn] resnet50_preprocess_local timing failed: {e}")
        try:
            # Wrap CPU output to include batch dimension and build meta with no letterbox
            cpu_out_wrapped = [yolo_cpu_output0[None, ...]]
            meta_dummy = {"orig_w": img_w, "orig_h": img_h, "ratio": 1.0, "pad": (0, 0), "input_size": (608, 608)}
            avg3 = avg_time_ms(yolo_postprocess_cpu, cpu_out_wrapped, raw_img.copy(), meta_dummy)
            results.append(("yolo_postprocess_cpu", avg3))
        except Exception as e:
            self.log_message(f"[Warn] yolo_postprocess_cpu timing failed: {e}")
        try:
            meta_dummy = {"orig_w": img_w, "orig_h": img_h, "ratio": 1.0, "pad": (0, 0), "input_size": (608, 608)}
            avg4 = avg_time_ms(yolo_postprocess_npu, yolo_npu_output, raw_img.copy(), meta_dummy)
            results.append(("yolo_postprocess_npu", avg4))
        except Exception as e:
            self.log_message(f"[Warn] yolo_postprocess_npu timing failed: {e}")

        # Update table
        self.pre_post_table.setRowCount(0)
        self.pre_post_table.setColumnCount(2)
        self.pre_post_table.setHorizontalHeaderLabels(["Function", "Avg (ms)"])
        for name, avg_ms in results:
            row = self.pre_post_table.rowCount()
            self.pre_post_table.insertRow(row)
            from PyQt5.QtWidgets import QTableWidgetItem
            self.pre_post_table.setItem(row, 0, QTableWidgetItem(name))
            self.pre_post_table.setItem(row, 1, QTableWidgetItem(f"{avg_ms:.2f}"))

        # Log results
        if results:
            self.log_message("[Pre-Post] Average times (10 runs):")
            for name, avg_ms in results:
                self.log_message(f"  - {name}: {avg_ms:.2f} ms")
        
        # Save results to static_pre_post_time.json in project root
        try:
            data = {name: round(float(avg_ms), 2) for name, avg_ms in results}
            out_path = os.path.join(os.path.dirname(__file__), "static_pre_post_time.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            self.log_message(f"[Pre-Post] Saved averages to {out_path}\n")
        except Exception as e:
            self.log_message(f"[Warn] Failed to save static_pre_post_time.json: {e}\n")
    
    def generate_all_combinations(self, models=None):
        """Generate all possible model-to-device combinations."""
        import yaml
        
        root_folder = self.folder_input.text().strip()
        if not os.path.isdir(root_folder):
            return

        # Get selected paths (same as run_profiling)
        selected_paths = self.file_manager.get_selected_paths(
            self.model_tree_view, self.fs_model, root_folder
        )

        if not selected_paths:
            return

        # Collect model files (same as run_profiling)
        onnx_files, o_files = self.file_manager.collect_model_files(selected_paths)

        # Extract model names from paths
        models = []
        for path in onnx_files + o_files:
            rel_path = os.path.relpath(path, root_folder)
            parts = rel_path.split(os.sep)
            if len(parts) >= 1:
                model_name = parts[0]
                if model_name not in models:
                    models.append(model_name)

        if models is None or len(models) == 0:
            self.log_message("[Warning] No models selected for assignment.")
            return
            
        # Limit to maximum 4 models as specified
        if len(models) > 4:
            self.log_message(f"[Warning] More than 4 models selected. Using only the first 4.")
            models = models[:4]
            
        # Load device information from target device file
        try:
            with open(self.device_settings_file, "r") as f:
                device_config = yaml.safe_load(f)
                
            # Extract device information
            cpu_count = device_config.get("devices", {}).get("cpu", {}).get("count", 1)
            npu_count = device_config.get("devices", {}).get("npu", {}).get("count", 0)
            npu_ids = device_config.get("devices", {}).get("npu", {}).get("ids", [])
            
            self.log_message(f"[Info] Found {cpu_count} CPU(s) and {npu_count} NPU(s) with IDs {npu_ids}")
        except Exception as e:
            self.log_message(f"[Error] Failed to load {self.device_settings_file}: {e}")
            return
            
        # Generate all possible combinations
        combinations = []
        
        # Helper function to generate combinations recursively
        def generate_combinations(model_idx, current_assignment, available_npus):
            # Base case: all models have been assigned
            if model_idx >= len(models):
                combinations.append(current_assignment.copy())
                return
                
            model = models[model_idx]
            
            # Option 1: Assign to CPU (always possible since CPU can run multiple models)
            current_assignment[model] = "cpu"
            generate_combinations(model_idx + 1, current_assignment, available_npus)
            
            # Option 2: Assign to available NPUs (one model per NPU)
            for npu_id in available_npus:
                current_assignment[model] = f"npu{npu_id}"
                # Remove this NPU from available NPUs for recursive calls
                new_available = available_npus.copy()
                new_available.remove(npu_id)
                generate_combinations(model_idx + 1, current_assignment, new_available)
                
        # Start the recursive generation
        generate_combinations(0, {}, set(npu_ids))
        
        # Update assignment_results for display in the UI
        # Only include unique model-device pairs from the first combination
        self.assignment_results = []
        if combinations:
            # Use the first combination for display
            for model, device in combinations[0].items():
                self.assignment_results.append((model, device))
            
            # Log the number of total combinations
            self.log_message(f"[Info] Generated {len(combinations)} possible combinations")
        
        # Create the model_schedules.yaml content
        schedules = {}
        
        for i, combination in enumerate(combinations):
            combination_name = f"combination_{i+1}"
            schedules[combination_name] = {}
            
            for j, (model, device) in enumerate(combination.items()):
                # Create a unique ID for this model-device pair
                model_id = f"{model}_{device}"
                
                # Add the model configuration
                schedules[combination_name][model_id] = {
                    "model": model,
                    "execution": device,
                    "display": f"view{j+1}"  # Assign views in order
                }
        
        # Write to model_schedules.yaml
        try:
            with open("model_schedules.yaml", "w") as f:
                # Add header comments
                f.write("# model_schedules.yaml\n")
                f.write("# Auto-generated configuration for model execution on CPU or NPU\n\n")
                
                # Add target device file information
                f.write(f"# Target device file: {self.device_settings_file}\n")
                f.write("# Available devices:\n")
                f.write(f"# - CPU: {cpu_count}\n")
                if npu_count > 0:
                    f.write(f"# - NPU: {npu_count} (IDs: {', '.join(map(str, npu_ids))})\n")
                f.write("\n")
                
                # Add available models comment
                f.write("# Available models:\n")
                for model in models:
                    f.write(f"# - {model}\n")
                f.write("\n")
                
                # Add combinations
                f.write("# Model-execution configurations with unique IDs\n")
                
                # Custom YAML dumping to add blank lines between combinations
                f.write(yaml.dump(schedules, default_flow_style=False).replace("combination_", "\ncombination_"))
                
            self.log_message(f"[Success] Generated {len(combinations)} combinations in model_schedules.yaml")
        except Exception as e:
            self.log_message(f"[Error] Failed to write model_schedules.yaml: {e}")
            
        # Log assignments
        self.log_message("\n[Model Assignments]")
        for model, device in self.assignment_results:
            self.log_message(f"{model}: {device}")
        

    
    def show_partition_assignment_dialog(self):
        """Show dialog with partition assignments."""
        if not self.assignment_results:
            self.log_message("[Warning] No assignments available. Run profiling first.")
            return
        
        dialog = self.ui_components.show_partition_assignment_dialog(self, self.assignment_results)
        dialog.exec_()
    
    def save_sample_data(self):
        """Save current profiling data as a sample."""
        self.file_manager.save_sample_data(
            self.cpu_table, self.npu1_table, self.npu2_table, self.total_table
        )
    
    def load_sample_data(self):
        """Load sample profiling data."""
        # Clear existing data
        self.ui_components.init_table(self.cpu_table)
        self.ui_components.init_table(self.npu1_table)
        self.ui_components.init_table(self.npu2_table)
        
        # Load sample data
        sample_data = self.file_manager.load_sample_data()
        if not sample_data:
            return
        
        # Fill tables with sample data
        for item in sample_data.get("cpu_data", []):
            self.ui_components.insert_result_row(
                self.cpu_table, item["model"], item["load"], item["infer"]
            )
        
        for item in sample_data.get("npu1_data", []):
            self.ui_components.insert_result_row(
                self.npu1_table, item["model"], item["load"], item["infer"]
            )
        
        for item in sample_data.get("npu2_data", []):
            self.ui_components.insert_result_row(
                self.npu2_table, item["model"], item["load"], item["infer"]
            )
        
        # Process total table
        self.ui_components.initialize_total_table(self.total_table)
        
        # Extract model data from sample
        valid_model_onnx = {}
        for item in sample_data.get("total_data", []):
            model = item["model"]
            cpu_infer = item["cpu_infer"]
            valid_model_onnx[model] = [0.0, cpu_infer]  # Dummy load time, real infer time
        
        # Process results
        all_models = set(item["model"] for item in sample_data.get("total_data", []))
        
        # Extract NPU data
        npu1_load = {item["model"]: item["npu1_load"] for item in sample_data.get("total_data", [])}
        npu1_infer = {item["model"]: item["npu1_infer"] for item in sample_data.get("total_data", [])}
        npu2_load = {item["model"]: item["npu2_load"] for item in sample_data.get("total_data", [])}
        npu2_infer = {item["model"]: item["npu2_infer"] for item in sample_data.get("total_data", [])}
        
        # Populate total table
        self.ui_components.populate_total_table(
            self.total_table, all_models, valid_model_onnx,
            npu1_load, npu1_infer, npu2_load, npu2_infer, {}
        )
        
        # Calculate and display totals
        self._calculate_and_display_totals()
        
        self.log_message("[Info] Sample data loaded successfully.")
    
    def show_settings_dialog(self):
        """Show settings dialog."""
        dialog, self.device_settings_file = self.ui_components.show_settings_dialog(
            self, self.device_settings_file, self.load_device_settings
        )
        dialog.exec_()
    
    def load_device_settings(self):
        """Load device settings from file."""
        self.device_settings = self.file_manager.load_device_settings(self.device_settings_file)


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = ONNXProfilerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()