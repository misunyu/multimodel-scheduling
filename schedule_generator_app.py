import os
import sys
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QDialog,
    QTreeView, QPlainTextEdit, QTableWidget, QAction,
    QFileSystemModel, QTabWidget, QWidget,
    QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton,
    QHeaderView, QLabel, QAbstractItemView, QMessageBox
)

from schedule_generator import (
    ModelProfiler, DataProcessor, UIComponents, FileManager
)

class CPUProfileWorker(QObject):
    """Background worker to profile models on CPU without blocking UI."""
    result = pyqtSignal(str, float, float)  # rel_path, load_ms, infer_ms
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(dict)  # valid_model_onnx mapping

    def __init__(self, onnx_files, root_folder):
        super().__init__()
        self.onnx_files = list(onnx_files)
        self.root_folder = root_folder
        # Forward profiler logs (e.g., LLM warmup) to GUI via progress signal
        self.profiler = ModelProfiler(log_callback=self.progress.emit)

    def run(self):
        valid_model_onnx = {}
        for path in self.onnx_files:
            try:
                if self.profiler.contains_custom_op(path):
                    self.progress.emit(f"[Skip] {path} contains custom ops\n")
                    continue
                load_ms, infer_ms, _ = self.profiler.profile_model_cpu(path)
                rel_path = os.path.relpath(path, self.root_folder)
                self.result.emit(rel_path, load_ms, infer_ms)
                parts = rel_path.split(os.sep)
                if len(parts) == 3 and parts[1] == "model" and parts[2].endswith(".onnx"):
                    model_key = parts[0]
                    if model_key not in valid_model_onnx:
                        valid_model_onnx[model_key] = [0.0, 0.0]
                    valid_model_onnx[model_key][0] += load_ms
                    valid_model_onnx[model_key][1] += infer_ms
            except Exception as e:
                self.error.emit(f"Skipping {path}: {e}")
        self.finished.emit(valid_model_onnx)


class GPUProfileWorker(QObject):
    """Background worker to profile models on GPU without blocking UI."""
    result = pyqtSignal(str, float, float)  # rel_path, load_ms, infer_ms
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, onnx_files, root_folder):
        super().__init__()
        self.onnx_files = list(onnx_files)
        self.root_folder = root_folder
        # Forward profiler logs (e.g., LLM warmup) to GUI via progress signal
        self.profiler = ModelProfiler(log_callback=self.progress.emit)

    def run(self):
        for path in self.onnx_files:
            try:
                rel_path = os.path.relpath(path, self.root_folder)
                load_ms, infer_ms, _ = self.profiler.profile_model_gpu(path)
                self.result.emit(rel_path, load_ms, infer_ms)
            except Exception as e:
                self.error.emit(f"GPU profiling failed for {path}: {e}")
        self.finished.emit()

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

        # Selection limiting state (max 4 top-level selections)
        self._suppress_tree_selection_handler = False
        self._last_selected_top_keys = set()
        
        # Default device settings file
        self.device_settings_file = target_device_file if target_device_file else "target_device.yaml"
        
        # Load device settings from file
        self.device_settings = self.file_manager.load_device_settings(self.device_settings_file)
        
        # Assignment results storage
        self.assignment_results = []
        
        # Flag to prevent multiple simultaneous profiling runs
        self._profiling_in_progress = False
        # Threads
        self._cpu_thread = None
        self._gpu_thread = None
        # Cache tab pages
        self.cpu_tab_widget = self.findChild(QWidget, "cpu_tab")
        self.gpu_tab_widget = self.findChild(QWidget, "gpu_tab")
    
    def setup_ui_elements(self):
        """Find and set up UI elements."""
        # Find main UI elements
        self.result_tabs = self.findChild(QTabWidget, "result_tab_widget")
        
        # Set up main layout
        main_layout = self.findChild(QHBoxLayout, "mainLayout")
        if main_layout:
            main_layout.setStretch(0, 3)
            main_layout.setStretch(1, 7)
        
        # Find input and control elements
        self.folder_input = self.findChild(QLineEdit, "folder_input")
        self.profile_button = self.findChild(QPushButton, "profile_button")
        self.generate_static_button = self.findChild(QPushButton, "generate_static_button")
        self.generate_all_button = self.findChild(QPushButton, "generate_all_button")
        self.model_tree_view = self.findChild(QTreeView, "model_tree_view")
        self.log_output = self.findChild(QPlainTextEdit, "log_output")
        self.total_table = self.findChild(QTableWidget, "total_table")
        
        # Find table elements
        self.cpu_table = self.findChild(QTableWidget, "cpu_table")
        # GPU table (renamed from NPU0)
        self.gpu_table = self.findChild(QTableWidget, "gpu_table")
        # For backward compatibility in downstream functions, alias gpu table as npu1_table
        self.npu1_table = self.gpu_table
        # Create hidden placeholder for legacy NPU2 table
        self.npu2_table = QTableWidget(self)
        self.npu2_table.setVisible(False)
        
        # Set up table headers
        for table in [self.cpu_table, self.npu1_table]:
            header = table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # Removed Pre/Post table (pre_post_table) and related UI setup
        
        # Create and add legend label
        self.legend_label = QLabel()
        self.legend_label.setText(
            "<span style='background-color:#cce6ff;'>&nbsp;&nbsp;&nbsp;</span> CPU &nbsp;&nbsp;"
            "<span style='background-color:#ffffcc;'>&nbsp;&nbsp;&nbsp;</span> GPU (Apple M-series)"
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
        # Connect buttons
        self.profile_button.clicked.connect(self.run_profiling)
        self.generate_static_button.clicked.connect(
            lambda: self.ui_components.highlight_deploy_results(
                self.total_table, self.profiled_times, self.profiled_models, self.device_settings
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
        self.fs_model.setNameFilters(["*.onnx"])
        self.fs_model.setNameFilterDisables(False)
        self.model_tree_view.setModel(self.fs_model)
        self.model_tree_view.setMinimumWidth(320)
        self.model_tree_view.header().setStretchLastSection(True)
        self.model_tree_view.header().setDefaultSectionSize(300)
        self.model_tree_view.setColumnWidth(1, 60)
        self.model_tree_view.setColumnHidden(2, True)
        self.model_tree_view.setColumnHidden(3, True)
        self.model_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)
        # Ensure selection applies to whole rows (all visible columns)
        try:
            self.model_tree_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        except Exception:
            pass
        
        # Connect selection changed signal
        self.model_tree_view.selectionModel().selectionChanged.connect(self.handle_tree_selection_changed)
        
        # Set default folder to models_onnx instead of models
        default_folder = os.path.join(os.getcwd(), "models_onnx")
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
    
    def handle_tree_selection_changed(self, selected, deselected):
        """
        Handle selection changes in the tree view.
        - Only top-level folders (model names) can be toggled.
        - No limit on how many top-level folders can be selected.
        Subfolders and their Size tabs are automatically deselected.

        Args:
            selected: QItemSelection of newly selected items
            deselected: QItemSelection of newly deselected items
        """
        # Prevent re-entrancy
        if self._suppress_tree_selection_handler:
            return
            
        # Get the root path and index
        root_path = self.folder_input.text().strip()
        root_index = self.fs_model.index(root_path)
        
        def _is_top_level(index):
            row_index = self.fs_model.index(self.fs_model.filePath(index), 0)
            return row_index.parent() == root_index

        def _key_for_index(index):
            row_index = self.fs_model.index(self.fs_model.filePath(index), 0)
            path = self.fs_model.filePath(row_index)
            return os.path.basename(path)

        try:
            # First: ensure only top-level entries remain selected (deselect any sub-items)
            if selected.indexes():
                self._suppress_tree_selection_handler = True
                try:
                    for index in selected.indexes():
                        if not _is_top_level(index):
                            # Deselect the entire row (all columns)
                            row_index = self.fs_model.index(self.fs_model.filePath(index), 0)
                            cols = self.fs_model.columnCount()
                            for c in range(cols):
                                sib = row_index.sibling(row_index.row(), c)
                                try:
                                    self.model_tree_view.selectionModel().select(
                                        sib,
                                        self.model_tree_view.selectionModel().Deselect
                                    )
                                except Exception:
                                    pass
                finally:
                    self._suppress_tree_selection_handler = False

            # Collect all currently selected top-level keys (column 0 only to avoid duplicates)
            current_keys = set()
            for idx in self.model_tree_view.selectedIndexes():
                if idx.column() != 0:
                    continue
                if _is_top_level(idx):
                    current_keys.add(_key_for_index(idx))

            # No selection cap: simply record the latest set for potential future use
            self._last_selected_top_keys = set(current_keys)
        except Exception:
            # If anything goes wrong, do not block user selection in a broken state
            self._suppress_tree_selection_handler = False
    
    def set_tree_root(self, folder):
        """Set the root folder for the tree view."""
        self.fs_model.setRootPath(folder)
        index = self.fs_model.index(folder)
        self.model_tree_view.setRootIndex(index)
        # Reset selection limiting state when root changes
        self._last_selected_top_keys = set()
    
    def expand_parents_of_onnx_files(self, root_folder):
        """Expand tree view items that contain ONNX or O files."""
        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                if f.endswith(".onnx"):
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
            
            # Collect ONNX files only from user-selected items in model_tree_view
            selected_paths = self.file_manager.get_selected_paths(
                self.model_tree_view, self.fs_model, root_folder
            )

            if not selected_paths:
                # Nothing selected → warn and restore UI state
                self.log_message("[Warning] No models selected. Please select model folders in the tree view.")
                self._profiling_in_progress = False
                self.profile_button.setEnabled(True)
                return

            onnx_files, o_files = self.file_manager.collect_model_files(selected_paths)

            # Special rule: for tiny-llama-chat-onnx folder, only run its model.onnx
            filtered_onnx_files = []
            for p in onnx_files:
                parts = os.path.normpath(p).split(os.sep)
                if 'tiny-llama-chat-onnx' in parts:
                    # keep only if the file name is exactly model.onnx
                    if os.path.basename(p) == 'model.onnx':
                        filtered_onnx_files.append(p)
                else:
                    filtered_onnx_files.append(p)
            onnx_files = filtered_onnx_files
            
            # Start non-blocking profiling using background threads
            self._start_cpu_profiling_async(onnx_files, root_folder)
        
        except Exception as e:
            self.log_message(f"[Error] Profiling failed: {str(e)}\n")
            # If setup failed before starting workers, re-enable UI here
            self._profiling_in_progress = False
            self.profile_button.setEnabled(True)
        finally:
            # UI re-enable will be handled when workers finish
            pass

    # ----------------------
    # Async profiling (QThread)
    # ----------------------
    def _start_cpu_profiling_async(self, onnx_files, root_folder):
        # Switch to CPU tab while filling
        if self.result_tabs and self.cpu_tab_widget:
            self.result_tabs.setCurrentWidget(self.cpu_tab_widget)
        # Set up worker and thread
        self._cpu_thread = QThread(self)
        self._cpu_worker = CPUProfileWorker(onnx_files, root_folder)
        self._cpu_worker.moveToThread(self._cpu_thread)
        # Connect signals
        self._cpu_thread.started.connect(self._cpu_worker.run)
        self._cpu_worker.result.connect(self._on_cpu_result)
        self._cpu_worker.progress.connect(self.log_message)
        self._cpu_worker.error.connect(lambda msg: self.log_message(f"[Error] {msg}"))
        self._cpu_worker.finished.connect(lambda valid_map: self._on_cpu_finished(valid_map, onnx_files, root_folder))
        self._cpu_worker.finished.connect(self._cpu_thread.quit)
        self._cpu_worker.finished.connect(self._cpu_worker.deleteLater)
        self._cpu_thread.finished.connect(self._cpu_thread.deleteLater)
        # Start thread
        self._cpu_thread.start()

    def _on_cpu_result(self, rel_path, load_ms, infer_ms):
        # Update CPU table in GUI thread
        disp_path = self._normalize_display_model(rel_path)
        tokens_per_s = self._compute_tokens_per_s(disp_path, infer_ms)
        self.ui_components.insert_result_row(self.cpu_table, disp_path, load_ms, infer_ms, tokens_per_s)
        self.log_message(f"[CPU] {disp_path}")
        if tokens_per_s is not None:
            self.log_message(f"       Load: {load_ms:.1f} ms, Inference: {infer_ms:.1f} ms, Throughput: {tokens_per_s:.2f} tokens/s\n")
        else:
            self.log_message(f"       Load: {load_ms:.1f} ms, Inference: {infer_ms:.1f} ms\n")

    def _on_cpu_finished(self, valid_model_onnx, onnx_files, root_folder):
        # After CPU finished, start GPU profiling
        self._start_gpu_profiling_async(onnx_files, root_folder, valid_model_onnx)

    def _start_gpu_profiling_async(self, onnx_files, root_folder, valid_model_onnx):
        # Switch to GPU tab while filling
        if self.result_tabs and self.gpu_tab_widget:
            self.result_tabs.setCurrentWidget(self.gpu_tab_widget)
        # Set up worker and thread
        self._gpu_thread = QThread(self)
        self._gpu_worker = GPUProfileWorker(onnx_files, root_folder)
        self._gpu_worker.moveToThread(self._gpu_thread)
        # Connect signals
        self._gpu_thread.started.connect(self._gpu_worker.run)
        self._gpu_worker.result.connect(self._on_gpu_result)
        self._gpu_worker.progress.connect(self.log_message)
        self._gpu_worker.error.connect(lambda msg: self.log_message(f"[Error] {msg}"))
        # When GPU finished, process totals and wrap up
        def _gpu_done():
            try:
                if self.total_table:
                    self._process_profiling_results(valid_model_onnx, root_folder)
            finally:
                self._profiling_in_progress = False
                self.profile_button.setEnabled(True)
                self.log_message("[Complete] Profiling finished.\n")
        self._gpu_worker.finished.connect(_gpu_done)
        self._gpu_worker.finished.connect(self._gpu_thread.quit)
        self._gpu_worker.finished.connect(self._gpu_worker.deleteLater)
        self._gpu_thread.finished.connect(self._gpu_thread.deleteLater)
        # Start thread
        self._gpu_thread.start()

    def _on_gpu_result(self, rel_path, load_ms, infer_ms):
        # Update GPU table (aliased as npu1_table) in GUI thread
        disp_path = self._normalize_display_model(rel_path)
        tokens_per_s = self._compute_tokens_per_s(disp_path, infer_ms)
        self.ui_components.insert_result_row(self.npu1_table, disp_path, load_ms, infer_ms, tokens_per_s)
        self.log_message(f"[GPU] {disp_path}")
        if tokens_per_s is not None:
            self.log_message(f"       Load: {load_ms:.1f} ms, Inference: {infer_ms:.1f} ms, Throughput: {tokens_per_s:.2f} tokens/s\n")
        else:
            self.log_message(f"       Load: {load_ms:.1f} ms, Inference: {infer_ms:.1f} ms\n")

    def _is_llm_model(self, model_path_or_rel: str) -> bool:
        n = (model_path_or_rel or "").lower()
        return ("gpt2" in n) or ("tiny-llama" in n)

    def _test_sentence_tokens(self) -> int:
        # Widely used English pangram for testing
        sent = "The quick brown fox jumps over the lazy dog."
        return max(1, len(sent.strip().split()))

    def _compute_tokens_per_s(self, rel_path: str, infer_ms: float):
        if not self._is_llm_model(rel_path):
            return None
        if not isinstance(infer_ms, (int, float)) or infer_ms <= 0:
            return None
        tokens = self._test_sentence_tokens()
        return tokens / (infer_ms / 1000.0)

    def _normalize_display_model(self, rel_path: str) -> str:
        """
        Normalize model display name for known cases. In particular, when the user
        profiles inside the tiny-llama directory, rel_path can be just 'model.onnx'
        (or even 'model'). In such a case, show folder + filename to make it clear
        and to allow LLM detection to work for tokens/s.
        """
        try:
            low = (rel_path or "").lower()
            # If it already contains tiny-llama keyword, keep as is
            if "tiny-llama" in low:
                return rel_path
            base = os.path.basename(rel_path).lower()
            if base in ("model.onnx", "model"):
                return "tiny-llama-chat-onnx/model.onnx"
            return rel_path
        except Exception:
            return rel_path
    
    def _initialize_profiling_ui(self):
        """Initialize UI for profiling."""
        if self.log_output:
            self.log_output.clear()
            self.log_message("[Start] Profiling models...\n")
        
        # Clear previous results
        self.ui_components.init_table(self.cpu_table)
        self.ui_components.init_table(self.npu1_table)
        
        # Pre/Post table removed
        
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

    def _profile_onnx_models_gpu(self, onnx_files, root_folder):
        """Profile ONNX models on Apple GPU (CoreML EP) and update GPU table."""
        for path in onnx_files:
            try:
                rel_path = os.path.relpath(path, root_folder)
                load_ms, infer_ms, _ = self.profiler.profile_model_gpu(path)
                self.ui_components.insert_result_row(self.npu1_table, rel_path, load_ms, infer_ms)
                self.log_message(f"[GPU] {rel_path}")
                self.log_message(f"       Load: {load_ms:.1f} ms, Inference: {infer_ms:.1f} ms\n")
            except Exception as e:
                self.log_message(f"[Error] GPU profiling failed for {path}: {e}\n")
    
    def _process_profiling_results(self, valid_model_onnx, root_folder):
        """Process profiling results and update total table."""
        # Initialize total table
        self.ui_components.initialize_total_table(self.total_table)
        
        # Process results using data processor
        all_models, cpu_infer_per_partition, npu1_load, npu1_infer, npu2_load, npu2_infer = \
            self.data_processor.process_profiling_results(
                valid_model_onnx, self.cpu_table, self.npu1_table, self.npu2_table
            )
        # Normalize model identifiers for Total tab so that tiny-llama shows as
        # 'tiny-llama-chat-onnx/model.onnx' instead of bare 'model' and to ensure
        # LLM detection/tokens per second calculations work.
        def _build_base_to_display(table):
            mapping = {}
            for row in range(table.rowCount()):
                name_item = table.item(row, 0)
                if not name_item:
                    continue
                disp = name_item.text() or ""
                base = os.path.splitext(os.path.basename(disp))[0]
                # Prefer a mapping that includes folder info when available
                # If multiple rows share the same base, keep the one that contains
                # a slash (more informative path) or the longest string.
                prev = mapping.get(base)
                if prev is None or ("/" in disp or "\\" in disp) or len(disp) > len(prev):
                    mapping[base] = disp
            return mapping

        # Build mapping from base model key (e.g., 'model') to display path used in tables
        base_to_disp = {}
        for t in (self.cpu_table, self.npu1_table):
            m = _build_base_to_display(t)
            base_to_disp.update(m)

        def _remap_dict_of_float(d: dict) -> dict:
            remapped = {}
            for k, v in d.items():
                disp = base_to_disp.get(k, k)
                remapped[disp] = v if isinstance(v, (int, float)) else v
            return remapped

        def _remap_dict_of_list(d: dict) -> dict:
            remapped = {}
            for k, v in d.items():
                disp = base_to_disp.get(k, k)
                remapped.setdefault(disp, [])
                remapped[disp].extend(v if isinstance(v, list) else [v])
            return remapped

        cpu_infer_per_partition = _remap_dict_of_list(cpu_infer_per_partition)
        npu1_load = _remap_dict_of_float(npu1_load)
        npu1_infer = _remap_dict_of_float(npu1_infer)
        # npu2_* kept for interface compatibility though not shown/used
        npu2_load = _remap_dict_of_float(npu2_load)
        npu2_infer = _remap_dict_of_float(npu2_infer)

        # Rebuild all_models set from remapped keys
        all_models = set()
        all_models.update(cpu_infer_per_partition.keys())
        all_models.update(npu1_load.keys())
        all_models.update(npu1_infer.keys())
        all_models.update(npu2_load.keys())
        all_models.update(npu2_infer.keys())

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
        gpu_infer_total = 0.0
        
        for row in range(self.total_table.rowCount()):
            try:
                cpu_infer_total += float(self.total_table.item(row, 1).text())
                gpu_infer_total += float(self.total_table.item(row, 2).text())
            except:
                pass
        
        # Add total row
        self.ui_components.add_total_row(
            self.total_table, cpu_infer_total, 
            gpu_infer_total
        )

    # Removed _profile_pre_post_avg_times and related functionality
    
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

        # Extract model names from paths (use filename stem; no explicit partitions)
        models = []
        for path in onnx_files:
            base = os.path.basename(path)
            model_name, _ = os.path.splitext(base)
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
            gpu_count = device_config.get("devices", {}).get("gpu", {}).get("count", 1)
            gpu_ids = device_config.get("devices", {}).get("gpu", {}).get("ids", [0])
            
            self.log_message(f"[Info] Found {cpu_count} CPU(s) and {gpu_count} GPU(s) with IDs {gpu_ids}")
        except Exception as e:
            self.log_message(f"[Error] Failed to load {self.device_settings_file}: {e}")
            return
        
        # Generate all possible combinations
        combinations = []
        
        # Helper function to generate combinations recursively
        def generate_combinations(model_idx, current_assignment):
            # Base case: all models have been assigned
            if model_idx >= len(models):
                combinations.append(current_assignment.copy())
                return
            
            model = models[model_idx]
            
            # Option 1: Assign to CPU (always possible since CPU can run multiple models)
            current_assignment[model] = "cpu"
            generate_combinations(model_idx + 1, current_assignment)
            
            # Option 2: Assign to GPU (assume single GPU device on Apple M-series)
            if gpu_count > 0:
                current_assignment[model] = "gpu"
                generate_combinations(model_idx + 1, current_assignment)
                
        # Start the recursive generation
        generate_combinations(0, {})
        
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
                # Determine default inference FPS based on model type
                infps = None
                lname = model.lower()
                if "resnet50" in lname:
                    infps = 2
                elif "yolov3" in lname:
                    infps = 30
                schedules[combination_name][model_id] = {
                    "model": model,
                    "execution": device,
                    "display": f"view{j+1}",  # Assign views in order
                    **({"infps": infps} if infps is not None else {})
                }
        
        # Write to model_schedules.yaml
        try:
            with open("../tests/model_schedules.yaml", "w") as f:
                # Add header comments
                f.write("# model_schedules.yaml\n")
                f.write("# Auto-generated configuration for model execution on CPU or GPU\n\n")
                
                # Add target device file information
                f.write(f"# Target device file: {self.device_settings_file}\n")
                f.write("# Available devices:\n")
                f.write(f"# - CPU: {cpu_count}\n")
                if gpu_count > 0:
                    f.write(f"# - GPU: {gpu_count} (IDs: {', '.join(map(str, gpu_ids))})\n")
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
        
        # Extract NPU data (Load columns removed in total_data schema)
        npu1_infer = {item["model"]: item.get("npu1_infer", 0.0) for item in sample_data.get("total_data", [])}
        npu2_infer = {item["model"]: item.get("npu2_infer", 0.0) for item in sample_data.get("total_data", [])}
        # For backward compatibility, ignore potential npu1_load/npu2_load fields if present.
        npu1_load = {}
        npu2_load = {}
        
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