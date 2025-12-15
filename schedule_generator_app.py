import os
import sys
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread, QItemSelectionModel
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QDialog,
    QTreeView, QPlainTextEdit, QTableWidget, QAction,
    QFileSystemModel, QTabWidget, QWidget,
    QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton,
    QHeaderView, QLabel, QAbstractItemView, QMessageBox,
    QTableWidgetItem, QFileDialog
)
from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor

from schedule_generator import (
    ModelProfiler, DataProcessor, UIComponents, FileManager
)

class _TokenLogHighlighter(QSyntaxHighlighter):
    """Colors lines containing token count markers in the log output."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fmt = QTextCharFormat()
        self._fmt.setForeground(QColor("purple"))

    def highlightBlock(self, text: str) -> None:
        try:
            if "[Tokens]" in text:
                self.setFormat(0, len(text), self._fmt)
        except Exception:
            # Do not raise on highlight errors
            pass

class _EmittingStream(QObject):
    """File-like stream that emits written text via Qt signal (thread-safe)."""
    text_emitted = pyqtSignal(str)

    def __init__(self, prefix: str = "", parent: QObject = None):
        super().__init__(parent)
        self._buffer = []
        self._prefix = prefix

    def write(self, text: str):
        if not isinstance(text, str):
            text = str(text)
        # Accumulate and emit full lines to keep log tidy
        self._buffer.append(text)
        joined = ''.join(self._buffer)
        lines = joined.split('\n')
        # Keep the last partial line in buffer
        self._buffer = [lines.pop()] if lines else []
        for line in lines:
            if self._prefix:
                self.text_emitted.emit(f"{self._prefix}{line}")
            else:
                self.text_emitted.emit(line)

    def flush(self):
        # Emit any remaining partial content as a line
        if self._buffer:
            remaining = ''.join(self._buffer)
            self._buffer = []
            if remaining:
                if self._prefix:
                    self.text_emitted.emit(f"{self._prefix}{remaining}")
                else:
                    self.text_emitted.emit(remaining)

class _StderrReaderThread(QThread):
    """Background thread that reads from a file descriptor and emits lines."""
    line = pyqtSignal(str)

    def __init__(self, read_fd: int, parent: QObject = None):
        super().__init__(parent)
        self._rfd = read_fd

    def run(self):
        import os as _os
        buf = b""
        try:
            while True:
                try:
                    chunk = _os.read(self._rfd, 1024)
                except Exception:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        self.line.emit(line.decode("utf-8", errors="replace"))
                    except Exception:
                        pass
        finally:
            try:
                _os.close(self._rfd)
            except Exception:
                pass

class _FDStderrRedirector(QObject):
    """Redirects POSIX stderr (fd=2) to a pipe and emits lines via signal."""
    text_emitted = pyqtSignal(str)

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._orig_fd = None
        self._thread = None

    def start(self):
        import os as _os
        try:
            self._orig_fd = _os.dup(2)
            rfd, wfd = _os.pipe()
            # Redirect process stderr to pipe's write end
            _os.dup2(wfd, 2)
            try:
                _os.close(wfd)
            except Exception:
                pass
            # Start reader thread to consume from read end
            self._thread = _StderrReaderThread(rfd)
            self._thread.line.connect(lambda s: self.text_emitted.emit(f"[stderr] {s}"))
            self._thread.start()
        except Exception:
            # Fallback: do nothing
            self._orig_fd = None
            self._thread = None

    def stop(self):
        import os as _os
        # Restore original stderr if possible
        try:
            if self._orig_fd is not None:
                _os.dup2(self._orig_fd, 2)
                _os.close(self._orig_fd)
        except Exception:
            pass
        self._orig_fd = None
        # Let thread exit after pipe EOF
        if self._thread is not None:
            try:
                self._thread.wait(1000)
            except Exception:
                pass
            self._thread = None

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

        # Install syntax highlighter to color token-count lines in purple
        try:
            if getattr(self, 'log_output', None):
                self._log_highlighter = _TokenLogHighlighter(self.log_output.document())
        except Exception:
            self._log_highlighter = None
        
        # Connect signals
        self.connect_signals()
        
        # Set up file system model
        self.setup_file_system_model()

        # Redirect stdout to the log window and capture low-level stderr (onnxruntime warnings)
        try:
            # Python-level stdout
            self._orig_stdout = sys.stdout
            self._stdout_stream = _EmittingStream(prefix="")
            self._stdout_stream.text_emitted.connect(self.log_message)
            sys.stdout = self._stdout_stream
        except Exception:
            pass

        try:
            # POSIX-level stderr redirection (captures C/C++ library warnings)
            self._fd_stderr_redirector = _FDStderrRedirector()
            self._fd_stderr_redirector.text_emitted.connect(self.log_message)
            self._fd_stderr_redirector.start()
            # Ensure restoration on exit
            QApplication.instance().aboutToQuit.connect(self._fd_stderr_redirector.stop)
        except Exception:
            pass

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

        # LLM input token counts captured from profiler logs per device
        # Keys: 'CPU' and 'GPU' → Dict[display_model_name, input_token_count]
        self._llm_input_tokens = {"CPU": {}, "GPU": {}}
    
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
            # Always append as plain text; coloring is handled by syntax highlighter
            msg = str(message)
            self.log_output.appendPlainText(msg)
            # Try to parse token count lines to store input token counts for saving
            try:
                if "[Tokens]" in msg:
                    # Expected format:
                    # [Tokens][CPU] <name>: input '...' -> batch=X, seq=Y, total=Z
                    dev = None
                    if "[CPU]" in msg:
                        dev = "CPU"
                    elif "[GPU]" in msg:
                        dev = "GPU"
                    if dev is not None:
                        # Extract model name between device tag and colon
                        # e.g., "[Tokens][CPU] gpt2.onnx: input ... total=16"
                        after = msg.split("]")[-1].strip()  # take substring after last closing bracket
                        # But safer: find first colon
                        colon_idx = after.find(":")
                        model_name = after[:colon_idx].strip() if colon_idx != -1 else after
                        # Normalize model display name to match table
                        disp = self._normalize_display_model(model_name)
                        # Extract total tokens
                        total = None
                        if "total=" in msg:
                            try:
                                part = msg.split("total=")[-1]
                                # read consecutive digits
                                num_chars = []
                                for ch in part:
                                    if ch.isdigit():
                                        num_chars.append(ch)
                                    else:
                                        break
                                if num_chars:
                                    total = int("".join(num_chars))
                            except Exception:
                                total = None
                        if total is not None and disp:
                            self._llm_input_tokens.setdefault(dev, {})[disp] = total
            except Exception:
                pass
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
        # Compute tokens/s based on the actual input token count captured in purple [Tokens] logs
        tokens_per_s = None
        try:
            tok_cnt = self._lookup_llm_input_tokens("GPU", disp_path)
            if isinstance(infer_ms, (int, float)) and infer_ms > 0:
                if isinstance(tok_cnt, int) and tok_cnt > 0:
                    tokens_per_s = tok_cnt / (infer_ms / 1000.0)
                else:
                    # Fallback: estimate using test sentence tokens for LLMs
                    tokens_per_s = self._compute_tokens_per_s(disp_path, infer_ms)
        except Exception:
            # Final fallback
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
    
    def _lookup_llm_input_tokens(self, device: str, rel_or_disp_path: str) -> int:
        """
        Lookup the input token count for an LLM from the parsed purple-log values.
        - device: 'CPU' or 'GPU'
        - rel_or_disp_path: model path as shown in tables/logs
        Returns the integer token count if found, else None.
        """
        try:
            dev = "CPU" if str(device).upper().startswith("CPU") else "GPU"
            disp = self._normalize_display_model(rel_or_disp_path or "")
            m = self._llm_input_tokens.get(dev, {})
            # Direct display-name match
            if disp in m:
                return m[disp]
            # Try by base name against stored keys
            base = os.path.splitext(os.path.basename(disp))[0]
            for k, v in m.items():
                kb = os.path.splitext(os.path.basename(k))[0]
                if kb == base:
                    return v
        except Exception:
            pass
        return None
    
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
                # Compute tokens/s using the captured input token count from purple logs,
                # with a fallback heuristic for LLMs when tokens are unavailable.
                disp_path = self._normalize_display_model(rel_path)
                tokens_per_s = None
                try:
                    tok_cnt = self._lookup_llm_input_tokens("CPU", disp_path)
                    if isinstance(infer_ms, (int, float)) and infer_ms > 0:
                        if isinstance(tok_cnt, int) and tok_cnt > 0:
                            tokens_per_s = tok_cnt / (infer_ms / 1000.0)
                        else:
                            tokens_per_s = self._compute_tokens_per_s(disp_path, infer_ms)
                except Exception:
                    tokens_per_s = self._compute_tokens_per_s(disp_path, infer_ms)
                self.ui_components.insert_result_row(self.cpu_table, rel_path, load_ms, infer_ms, tokens_per_s)
                
                self.log_message(f"[CPU] {rel_path}")
                if tokens_per_s is not None:
                    self.log_message(f"       Load: {load_ms:.1f} ms, Inference: {infer_ms:.1f} ms, Throughput: {tokens_per_s:.2f} tokens/s\n")
                else:
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
                # Compute tokens/s using the captured input token count from purple logs,
                # with a fallback heuristic for LLMs when tokens are unavailable.
                disp_path = self._normalize_display_model(rel_path)
                tokens_per_s = None
                try:
                    tok_cnt = self._lookup_llm_input_tokens("GPU", disp_path)
                    if isinstance(infer_ms, (int, float)) and infer_ms > 0:
                        if isinstance(tok_cnt, int) and tok_cnt > 0:
                            tokens_per_s = tok_cnt / (infer_ms / 1000.0)
                        else:
                            tokens_per_s = self._compute_tokens_per_s(disp_path, infer_ms)
                except Exception:
                    tokens_per_s = self._compute_tokens_per_s(disp_path, infer_ms)
                self.ui_components.insert_result_row(self.npu1_table, disp_path, load_ms, infer_ms, tokens_per_s)
                self.log_message(f"[GPU] {disp_path}")
                if tokens_per_s is not None:
                    self.log_message(f"       Load: {load_ms:.1f} ms, Inference: {infer_ms:.1f} ms, Throughput: {tokens_per_s:.2f} tokens/s\n")
                else:
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
        
        # Build Tokens/s maps from per-device tabs (mirror values exactly as shown)
        def _build_tokens_map(table):
            tok_map = {}
            agg = {}
            for row in range(table.rowCount()):
                name_item = table.item(row, 0)
                tok_item = table.item(row, 3)
                if not name_item or not tok_item:
                    continue
                name = name_item.text() or ""
                txt = tok_item.text() or ""
                try:
                    val = float(txt)
                except Exception:
                    continue
                if val <= 0:
                    continue
                agg.setdefault(name, []).append(val)
            for k, vs in agg.items():
                try:
                    tok_map[k] = sum(vs) / len(vs)
                except Exception:
                    tok_map[k] = vs[-1]
            return tok_map

        cpu_tokens_map = _build_tokens_map(self.cpu_table)
        gpu_tokens_map = _build_tokens_map(self.npu1_table)

        # Populate total table
        self.ui_components.populate_total_table(
            self.total_table, all_models, valid_model_onnx, 
            npu1_load, npu1_infer, npu2_load, npu2_infer, 
            cpu_infer_per_partition,
            cpu_tokens_map=cpu_tokens_map,
            gpu_tokens_map=gpu_tokens_map
        )
        
        # Calculate and display totals
        if self.total_table.rowCount() > 0:
            self._calculate_and_display_totals()
    
    def _calculate_and_display_totals(self):
        """Calculate and display Total row as averages over non '-' entries.
        - CPU FPS: average of numeric FPS in column 1
        - GPU FPS: average of numeric FPS in column 2
        - CPU Tokens/s: average of numeric values in column 3
        - GPU Tokens/s: average of numeric values in column 4
        """
        cpu_fps_vals = []
        gpu_fps_vals = []
        cpu_tok_vals = []
        gpu_tok_vals = []

        def _read_fps(item):
            if not item:
                return 0.0
            try:
                data = item.data(Qt.UserRole)
                if isinstance(data, (int, float)):
                    return float(data)
            except Exception:
                pass
            try:
                txt = item.text()
                return float(txt) if txt and txt != "-" else 0.0
            except Exception:
                return 0.0

        # Exclude an existing Total row if present by checking the last row's first cell
        last_index = self.total_table.rowCount() - 1
        total_label_row = None
        if last_index >= 0:
            first = self.total_table.item(last_index, 0)
            if first and (first.text() or "").strip().lower() == "total":
                total_label_row = last_index

        for row in range(self.total_table.rowCount()):
            if total_label_row is not None and row == total_label_row:
                continue
            # FPS columns
            cpu_item = self.total_table.item(row, 1)
            gpu_item = self.total_table.item(row, 2)
            cpu_fps = _read_fps(cpu_item)
            gpu_fps = _read_fps(gpu_item)
            if cpu_fps > 0:
                cpu_fps_vals.append(cpu_fps)
            if gpu_fps > 0:
                gpu_fps_vals.append(gpu_fps)
            # Tokens/s columns
            def _read_tok(item):
                if not item:
                    return 0.0
                try:
                    txt = item.text()
                    return float(txt) if txt and txt != "-" else 0.0
                except Exception:
                    return 0.0
            cpu_tok = _read_tok(self.total_table.item(row, 3))
            gpu_tok = _read_tok(self.total_table.item(row, 4))
            if cpu_tok > 0:
                cpu_tok_vals.append(cpu_tok)
            if gpu_tok > 0:
                gpu_tok_vals.append(gpu_tok)
        
        # Compute averages (non '-' entries only)
        def _avg(arr):
            return (sum(arr) / len(arr)) if arr else 0.0
        cpu_fps_avg = _avg(cpu_fps_vals)
        gpu_fps_avg = _avg(gpu_fps_vals)
        cpu_tok_avg = _avg(cpu_tok_vals)
        gpu_tok_avg = _avg(gpu_tok_vals)

        # Add total row
        self.ui_components.add_total_row(
            self.total_table,
            cpu_fps_avg,
            gpu_fps_avg,
            cpu_tok_avg,
            gpu_tok_avg
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

        # Extract model identifiers from paths
        # Rule: if the file name is 'model.onnx' (or 'model'), use parent folder name as model id
        # otherwise use the filename stem.
        models = []
        for path in onnx_files:
            try:
                base = os.path.basename(path)
                stem, ext = os.path.splitext(base)
                if base.lower() == "model.onnx" or stem.lower() == "model":
                    parent = os.path.basename(os.path.dirname(path))
                    model_name = parent if parent else stem
                else:
                    model_name = stem
                if model_name not in models:
                    models.append(model_name)
            except Exception:
                # Fallback: keep prior behavior
                base = os.path.basename(path)
                model_name, _ = os.path.splitext(base)
                if model_name not in models:
                    models.append(model_name)

        if models is None or len(models) == 0:
            self.log_message("[Warning] No models selected for assignment.")
            return
            
        # Use all selected models without limiting to 4
        # Note: total combinations grow as 2^N (CPU/GPU choices). For large N, generation may take time.
        try:
            total_combos = 2 ** len(models)
            if total_combos > 4096:
                self.log_message(f"[Warning] Large combination set detected: {total_combos} combos for {len(models)} models. This may take time.")
        except Exception:
            pass
            
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
        # 1) Build profiling maps from current tables so we can fill infps/intps and per-device times.
        def _base_name(text: str) -> str:
            """Normalize a display or relative path into a schedule model id.
            - If it ends with '/model.onnx' (or just 'model'), return the parent folder name.
            - Else return the filename stem.
            """
            try:
                t = text or ""
                base = os.path.basename(t)
                stem, ext = os.path.splitext(base)
                if base.lower() == "model.onnx" or stem.lower() == "model":
                    parent = os.path.basename(os.path.dirname(t))
                    return parent if parent else stem
                return stem
            except Exception:
                return (text or "")

        # From Total table: per-device FPS for CV models and Tokens/s for LLMs
        model_to_cpu_fps = {}
        model_to_gpu_fps = {}
        model_to_cpu_tok = {}
        model_to_gpu_tok = {}

        for row in range(self.total_table.rowCount()):
            name_item = self.total_table.item(row, 0)
            if not name_item:
                continue
            disp = name_item.text() or ""
            key = _base_name(disp)

            # FPS are stored in UserRole as floats for non-LLM rows
            cpu_item = self.total_table.item(row, 1)
            gpu_item = self.total_table.item(row, 2)
            try:
                cpu_fps = cpu_item.data(Qt.UserRole) if cpu_item else None
                gpu_fps = gpu_item.data(Qt.UserRole) if gpu_item else None
                if isinstance(cpu_fps, (int, float)) and cpu_fps > 0:
                    model_to_cpu_fps[key] = float(cpu_fps)
                if isinstance(gpu_fps, (int, float)) and gpu_fps > 0:
                    model_to_gpu_fps[key] = float(gpu_fps)
            except Exception:
                pass

            # Tokens/s shown as text for LLM rows
            def _read_tok(item):
                try:
                    txt = item.text() if item else None
                    v = float(txt) if txt and txt != "-" else None
                    return v if (v is None or v > 0) else None
                except Exception:
                    return None
            cpu_tok = _read_tok(self.total_table.item(row, 3))
            gpu_tok = _read_tok(self.total_table.item(row, 4))
            if isinstance(cpu_tok, (int, float)):
                model_to_cpu_tok[key] = float(cpu_tok)
            if isinstance(gpu_tok, (int, float)):
                model_to_gpu_tok[key] = float(gpu_tok)

        # If no profiling results are available in the UI, fall back to
        # static_results/sample_profiling_data.json
        def _fallback_from_sample_json():
            try:
                import json as _json
                static_dir = os.path.join(os.getcwd(), "static_results")
                sample_path = os.path.join(static_dir, "sample_profiling_data.json")
                if not os.path.isfile(sample_path):
                    return
                with open(sample_path, "r") as sf:
                    sample = _json.load(sf)
                total_data = sample.get("total_data", []) or []
                for item in total_data:
                    try:
                        mpath = item.get("model", "") or ""
                        base = _base_name(mpath)
                        # CV FPS
                        cfps = item.get("cpu_fps", None)
                        gfps = item.get("gpu_fps", None)
                        if isinstance(cfps, (int, float)) and cfps > 0:
                            model_to_cpu_fps.setdefault(base, float(cfps))
                        if isinstance(gfps, (int, float)) and gfps > 0:
                            model_to_gpu_fps.setdefault(base, float(gfps))
                        # LLM tokens/s: compute from input_tokens and infer(ms)
                        cinfer = item.get("cpu_infer", None)
                        ginfer = item.get("gpu_infer", None)
                        c_in_tok = item.get("cpu_input_tokens", None)
                        g_in_tok = item.get("gpu_input_tokens", None)
                        # CPU tokens/s
                        if isinstance(cinfer, (int, float)) and cinfer > 0 and isinstance(c_in_tok, (int, float)) and c_in_tok > 0:
                            tps = float(c_in_tok) / (float(cinfer) / 1000.0)
                            if tps > 0:
                                model_to_cpu_tok.setdefault(base, tps)
                        # GPU tokens/s
                        if isinstance(ginfer, (int, float)) and ginfer > 0 and isinstance(g_in_tok, (int, float)) and g_in_tok > 0:
                            tps = float(g_in_tok) / (float(ginfer) / 1000.0)
                            if tps > 0:
                                model_to_gpu_tok.setdefault(base, tps)
                    except Exception:
                        continue
                if (not model_to_cpu_fps and not model_to_gpu_fps and not model_to_cpu_tok and not model_to_gpu_tok):
                    self.log_message("[Warning] sample_profiling_data.json loaded but contained no usable totals")
                else:
                    self.log_message("[Info] No UI profiling found. Using cached results from static_results/sample_profiling_data.json")
            except Exception as _e:
                self.log_message(f"[Warning] Failed to load fallback profiling from sample_profiling_data.json: {_e}")

        if not (model_to_cpu_fps or model_to_gpu_fps or model_to_cpu_tok or model_to_gpu_tok):
            _fallback_from_sample_json()

        schedules = {}
        
        for i, combination in enumerate(combinations):
            combination_name = f"combination_{i+1}"
            schedules[combination_name] = {}

            # Assign views only to specific vision models, sequentially from view1 to view4
            # Counter resets per combination and skips non-vision models
            assigned_views = 0

            for j, (model, device) in enumerate(combination.items()):
                # Create a unique ID for this model-device pair
                model_id = f"{model}_{device}"

                # Decide if this is an LLM by presence of tokens/s in any device map
                is_llm = model in model_to_cpu_tok or model in model_to_gpu_tok

                # Pick the throughput for the assigned device
                perf_fields = {}
                if is_llm:
                    if device == "cpu":
                        tok = model_to_cpu_tok.get(model)
                    else:
                        tok = model_to_gpu_tok.get(model)
                    if isinstance(tok, (int, float)) and tok > 0:
                        # intps must be an integer value in the YAML (min 1)
                        perf_fields["intps"] = max(1, int(round(float(tok))))
                else:
                    if device == "cpu":
                        fps = model_to_cpu_fps.get(model)
                    else:
                        fps = model_to_gpu_fps.get(model)
                    if isinstance(fps, (int, float)) and fps > 0:
                        # infps must be an integer value in the YAML (min 1)
                        perf_fields["infps"] = max(1, int(round(float(fps))))

                # Assign display views only for specific vision models; others hidden
                vision_with_view = {"mnasnet", "resnet50", "resnext50", "yolov4"}
                if model in vision_with_view and assigned_views < 4:
                    display_value = f"view{assigned_views + 1}"
                    assigned_views += 1
                else:
                    display_value = "none"

                schedules[combination_name][model_id] = {
                    "model": model,
                    "execution": device,
                    "display": display_value,
                    **perf_fields
                }
        
        # Write to static_results/model_schedules_<initials>.yaml
        try:
            static_dir = os.path.join(os.getcwd(), "static_results")
            os.makedirs(static_dir, exist_ok=True)
            # Build filename suffix from the first letters of the selected model names
            try:
                # Keep duplicates if multiple models share the same first letter (e.g., resnet50 + resnext50 -> r_r)
                initials = [m.strip()[0].lower() for m in models if isinstance(m, str) and m.strip()]
                initials.sort()
            except Exception:
                initials = []
            if initials:
                initials_suffix = "_" + "_".join(initials)
            else:
                initials_suffix = ""

            base_filename = f"model_schedules{initials_suffix}.yaml"
            out_yaml = os.path.join(static_dir, base_filename)
            with open(out_yaml, "w") as f:
                # Add header comments
                f.write(f"# {base_filename}\n")
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

            self.log_message(f"[Success] Generated {len(combinations)} combinations in static_results/{base_filename}")

            # Also generate scaled variants with throughput reduced by 1/3 and by 2/3
            def _scaled_schedules(src: dict, factor: float) -> dict:
                import copy
                dst = copy.deepcopy(src)
                for comb_name, entries in dst.items():
                    if not isinstance(entries, dict):
                        continue
                    for mid, cfg in entries.items():
                        if not isinstance(cfg, dict):
                            continue
                        # Scale either intps or infps if present
                        if "intps" in cfg and isinstance(cfg["intps"], (int, float)):
                            val = int(round(float(cfg["intps"]) * factor))
                            cfg["intps"] = max(1, val)
                        if "infps" in cfg and isinstance(cfg["infps"], (int, float)):
                            val = int(round(float(cfg["infps"]) * factor))
                            cfg["infps"] = max(1, val)
                        # Ensure no legacy 'time' field sneaks in
                        if "time" in cfg:
                            try:
                                del cfg["time"]
                            except Exception:
                                pass
                return dst

            variants = [
                (f"model_schedules{initials_suffix}_x2_3.yaml", 2.0/3.0, "# This is a 2/3 throughput variant (values reduced by 1/3)\n"),
                (f"model_schedules{initials_suffix}_x1_3.yaml", 1.0/3.0, "# This is a 1/3 throughput variant (values reduced by 2/3)\n"),
            ]

            for filename, factor, note in variants:
                try:
                    scaled = _scaled_schedules(schedules, factor)
                    out_path = os.path.join(static_dir, filename)
                    with open(out_path, "w") as vf:
                        # Header
                        vf.write(f"# {filename}\n")
                        vf.write("# Auto-generated configuration for model execution on CPU or GPU\n")
                        vf.write(note + "\n")
                        vf.write(f"# Target device file: {self.device_settings_file}\n")
                        vf.write("# Available devices:\n")
                        vf.write(f"# - CPU: {cpu_count}\n")
                        if gpu_count > 0:
                            vf.write(f"# - GPU: {gpu_count} (IDs: {', '.join(map(str, gpu_ids))})\n")
                        vf.write("\n")
                        vf.write("# Available models:\n")
                        for model in models:
                            vf.write(f"# - {model}\n")
                        vf.write("\n")
                        vf.write("# Model-execution configurations with unique IDs\n")
                        vf.write(yaml.dump(scaled, default_flow_style=False).replace("combination_", "\ncombination_"))
                    self.log_message(f"[Success] Also generated scaled schedule: static_results/{filename} (factor={factor:.3f})")
                except Exception as ve:
                    self.log_message(f"[Error] Failed to write scaled schedule {filename}: {ve}")
        except Exception as e:
            self.log_message(f"[Error] Failed to write static_results/model_schedules.yaml: {e}")
            
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
        # Ask user for destination file
        default_name = os.path.join(os.getcwd(), "sample_profiling_data.json")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Sample Data",
            default_name,
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        # Ensure .json extension
        if not path.lower().endswith('.json'):
            path = f"{path}.json"

        # Capture current selection context to persist along with profiling data
        root_folder = self.folder_input.text().strip() if self.folder_input else ""
        try:
            selected_paths = self.file_manager.get_selected_paths(
                self.model_tree_view, self.fs_model, root_folder
            ) if self.model_tree_view and hasattr(self, 'fs_model') else []
        except Exception:
            selected_paths = []

        self.file_manager.save_sample_data(
            self.cpu_table, self.npu1_table, self.npu2_table, self.total_table,
            extra_meta={
                "root_folder": root_folder,
                "selected_paths": selected_paths,
            },
            llm_input_tokens=self._llm_input_tokens,
            filename=path
        )
    
    def load_sample_data(self):
        """Load sample profiling data."""
        # Ask user for source file
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Sample Data",
            os.getcwd(),
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        # Clear existing data
        self.ui_components.init_table(self.cpu_table)
        self.ui_components.init_table(self.npu1_table)
        self.ui_components.init_table(self.npu2_table)
        
        # Load sample data
        sample_data = self.file_manager.load_sample_data(filename=path)
        if not sample_data:
            return
        
        # Fill CPU/NPU tables with sample data
        # Always recompute tokens/s from inference ms to keep consistency with Total tab
        for item in sample_data.get("cpu_data", []):
            model = item.get("model", "")
            load = item.get("load", 0.0)
            infer = item.get("infer", 0.0)
            # Normalize model name like in live profiling to ensure LLM detection is consistent
            disp_model = self._normalize_display_model(model)
            tokens = self._compute_tokens_per_s(disp_model, infer)
            self.ui_components.insert_result_row(
                self.cpu_table, disp_model, load, infer, tokens
            )
        
        # Support legacy key alias: some older files may use "gpu_data"
        npu1_list = sample_data.get("npu1_data") or sample_data.get("gpu_data", [])
        for item in npu1_list:
            model = item.get("model", "")
            load = item.get("load", 0.0)
            infer = item.get("infer", 0.0)
            disp_model = self._normalize_display_model(model)
            tokens = self._compute_tokens_per_s(disp_model, infer)
            self.ui_components.insert_result_row(
                self.npu1_table, disp_model, load, infer, tokens
            )
        
        for item in sample_data.get("npu2_data", []):
            model = item.get("model", "")
            load = item.get("load", 0.0)
            infer = item.get("infer", 0.0)
            disp_model = self._normalize_display_model(model)
            tokens = self._compute_tokens_per_s(disp_model, infer)
            self.ui_components.insert_result_row(
                self.npu2_table, disp_model, load, infer, tokens
            )
        
        # Process total table
        self.ui_components.initialize_total_table(self.total_table)
        
        total_items = sample_data.get("total_data") or []
        if total_items:
            # Directly restore rows from saved total_data (display FPS in columns 1-2)
            def _is_llm(name: str) -> bool:
                n = (name or "").lower()
                return ("gpt2" in n) or ("tiny-llama" in n)

            for it in total_items:
                row = self.total_table.rowCount()
                self.total_table.insertRow(row)
                # Normalize model name for display + LLM detection
                model_name = self._normalize_display_model(str(it.get("model", "")))
                self.total_table.setItem(row, 0, QTableWidgetItem(model_name))

                # Prefer stored FPS if present; otherwise derive from ms fields
                cpu_fps = it.get("cpu_fps")
                gpu_fps = it.get("gpu_fps")
                try:
                    cpu_fps = float(cpu_fps) if cpu_fps is not None else None
                except Exception:
                    cpu_fps = None
                try:
                    gpu_fps = float(gpu_fps) if gpu_fps is not None else None
                except Exception:
                    gpu_fps = None

                if cpu_fps is None or cpu_fps <= 0:
                    cpu_infer_ms = float(it.get("cpu_infer", 0.0) or 0.0)
                    cpu_fps = (1000.0 / cpu_infer_ms) if cpu_infer_ms > 0 else 0.0
                if gpu_fps is None or gpu_fps <= 0:
                    # Do not load legacy npu1_infer; only respect gpu_infer
                    gpu_infer_ms = float((it.get("gpu_infer", None)) or 0.0)
                    gpu_fps = (1000.0 / gpu_infer_ms) if gpu_infer_ms > 0 else 0.0

                # CPU FPS cell
                cpu_item = QTableWidgetItem("-")
                if not _is_llm(model_name):
                    cpu_item.setText(f"{cpu_fps:.1f}" if cpu_fps > 0 else "-")
                cpu_item.setData(Qt.UserRole, cpu_fps if isinstance(cpu_fps, (int, float)) else 0.0)
                self.total_table.setItem(row, 1, cpu_item)

                # GPU FPS cell
                gpu_item = QTableWidgetItem("-")
                if not _is_llm(model_name):
                    gpu_item.setText(f"{gpu_fps:.1f}" if gpu_fps > 0 else "-")
                gpu_item.setData(Qt.UserRole, gpu_fps if isinstance(gpu_fps, (int, float)) else 0.0)
                self.total_table.setItem(row, 2, gpu_item)

                # Recompute tokens/s from inference ms to match result tabs
                def _tok_from_ms(name: str, infer_ms_val: float):
                    return self._compute_tokens_per_s(name, infer_ms_val)

                cpu_infer_ms = float(it.get("cpu_infer", 0.0) or 0.0)
                # Do not load legacy npu1_infer; only respect gpu_infer
                gpu_infer_ms = float((it.get("gpu_infer", None)) or 0.0)

                cpu_tok_val = _tok_from_ms(model_name, cpu_infer_ms)
                gpu_tok_val = _tok_from_ms(model_name, gpu_infer_ms)

                self.total_table.setItem(row, 3, QTableWidgetItem("-" if cpu_tok_val is None else f"{cpu_tok_val:.2f}"))
                self.total_table.setItem(row, 4, QTableWidgetItem("-" if gpu_tok_val is None else f"{gpu_tok_val:.2f}"))

            # Add total row at the end
            self._calculate_and_display_totals()
        else:
            # Backward compatibility: reconstruct via populate_total_table
            # Extract model data from CPU table to build cpu_infer_per_partition
            valid_model_onnx = {}
            cpu_infer_per_partition = {}
            models = set()
            for row in range(self.cpu_table.rowCount()):
                name_item = self.cpu_table.item(row, 0)
                model = name_item.text() if name_item else ""
                if not model:
                    continue
                infer = 0.0
                try:
                    infer = float(self.cpu_table.item(row, 2).text())
                except Exception:
                    infer = 0.0
                cpu_infer_per_partition.setdefault(model, []).append(infer)
                valid_model_onnx[model] = [0.0, infer]
                models.add(model)

            # Build npu1_infer from NPU1 table
            npu1_infer = {}
            for row in range(self.npu1_table.rowCount()):
                name_item = self.npu1_table.item(row, 0)
                model = name_item.text() if name_item else ""
                if not model:
                    continue
                try:
                    val = float(self.npu1_table.item(row, 2).text())
                except Exception:
                    val = 0.0
                npu1_infer[model] = npu1_infer.get(model, 0.0) + val
                models.add(model)

            self.ui_components.populate_total_table(
                self.total_table, models, valid_model_onnx,
                {}, npu1_infer, {}, {}, cpu_infer_per_partition
            )
            self._calculate_and_display_totals()
        
        self.log_message("[Info] Sample data loaded successfully.")

        # Restore previously selected models/folder if present
        try:
            root_folder = sample_data.get("root_folder")
            selected_paths = sample_data.get("selected_paths") or []
            if root_folder and isinstance(root_folder, str) and os.path.isdir(root_folder):
                self.folder_input.setText(root_folder)
                # Set the tree root
                try:
                    self.set_tree_root(root_folder)
                except Exception:
                    pass

            # Re-select items with visibility ensured (expand + scroll)
            if selected_paths and self.model_tree_view and hasattr(self, 'fs_model'):
                def _apply_selection():
                    try:
                        sel_model = self.model_tree_view.selectionModel()
                        if sel_model:
                            sel_model.clearSelection()
                        first_valid_idx = None
                        for p in selected_paths:
                            try:
                                # Normalize the path to increase match rate
                                np = os.path.normpath(p)
                                rp = os.path.realpath(np)
                                # Try realpath first, then original
                                idx = self.fs_model.index(rp)
                                if (not idx) or (not idx.isValid()):
                                    idx = self.fs_model.index(np)
                                if idx and idx.isValid():
                                    # Expand parents so item is visible
                                    parent = idx.parent()
                                    while parent and parent.isValid():
                                        self.model_tree_view.expand(parent)
                                        parent = parent.parent()
                                    # Select the row
                                    if sel_model:
                                        sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
                                    # Remember first valid to set focus later
                                    if first_valid_idx is None:
                                        first_valid_idx = idx
                                    # Ensure it is scrolled into view
                                    self.model_tree_view.scrollTo(idx)
                            except Exception:
                                continue
                        if first_valid_idx is not None:
                            self.model_tree_view.setCurrentIndex(first_valid_idx)
                    except Exception:
                        pass

                # Defer selection until after model/tree updates are processed
                QTimer.singleShot(0, _apply_selection)
        except Exception:
            # Non-fatal; ignore restore errors
            pass
    
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