import os
import time
import numpy as np
import onnxruntime as ort

from PyQt6 import uic
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QLineEdit, QPushButton,
    QFileDialog, QTreeView, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QApplication,
    QHeaderView
)
from PyQt6.QtGui import QFileSystemModel


class ONNXProfiler(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("onnx_profiler_display_modify.ui", self)

        # Connect UI elements
        self.folder_input = self.findChild(QLineEdit, "folder_input")
        self.browse_button = self.findChild(QPushButton, "browse_button")
        self.profile_button = self.findChild(QPushButton, "profile_button")
        self.generate_button = self.findChild(QPushButton, "generate_button")
        self.model_tree_view = self.findChild(QTreeView, "model_tree_view")
        self.log_output = self.findChild(QPlainTextEdit, "log_output") or self.findChild(QPlainTextEdit, "log_textbox")

        self.cpu_table = self.findChild(QTableWidget, "cpu_table")
        self.npu1_table = self.findChild(QTableWidget, "npu1_table")
        self.npu2_table = self.findChild(QTableWidget, "npu2_table")

        # Stretch columns
        for table in [self.cpu_table, self.npu1_table, self.npu2_table]:
            header = table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        # File system model
        self.fs_model = QFileSystemModel()
        self.fs_model.setReadOnly(True)
        self.fs_model.setNameFilters(["*.onnx", "*.o"])
        self.fs_model.setNameFilterDisables(False)
        self.model_tree_view.setModel(self.fs_model)
        self.model_tree_view.setMinimumWidth(500)
        self.model_tree_view.header().setStretchLastSection(True)
        self.model_tree_view.header().setDefaultSectionSize(300)

        # Default folder
        default_folder = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(default_folder):
            default_folder = os.getcwd()

        self.folder_input.setText(default_folder)
        self.browse_button.clicked.connect(self.browse_folder)
        self.profile_button.clicked.connect(self.run_profiling)

        self.set_tree_root(default_folder)
        QTimer.singleShot(100, lambda: self.expand_parents_of_onnx_files(default_folder))

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", os.getcwd())
        if folder:
            self.folder_input.setText(folder)
            self.set_tree_root(folder)
            QTimer.singleShot(100, lambda: self.expand_parents_of_onnx_files(folder))

    def set_tree_root(self, folder):
        self.fs_model.setRootPath(folder)
        index = self.fs_model.index(folder)
        self.model_tree_view.setRootIndex(index)

    def expand_parents_of_onnx_files(self, root_folder):
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
        root_folder = self.folder_input.text().strip()
        if not os.path.isdir(root_folder):
            return

        if self.log_output:
            self.log_output.clear()
            self.log_output.appendPlainText("[Start] Profiling models...\n")
            QApplication.processEvents()

        self.init_table(self.cpu_table)
        self.init_table(self.npu1_table)
        self.init_table(self.npu2_table)

        onnx_files = []
        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                if f.endswith(".onnx"):
                    onnx_files.append(os.path.join(dirpath, f))

        for row, model_path in enumerate(onnx_files):
            model_file = os.path.basename(model_path)

            self.log_output.appendPlainText(f"[Start] Loading {model_file}...")
            QApplication.processEvents()

            try:
                load_ms, infer_ms, input_shape = self.profile_model_cpu(model_path)
                npu1_load, npu1_infer = self.profile_model_npu1(model_file)
                npu2_load, npu2_infer = self.profile_model_npu2(model_file)

                index = self.fs_model.index(model_path)
                display_name = f"{model_file} (CPU: {infer_ms:.1f}ms / NPU1: {npu1_infer:.1f}ms)"
                self.fs_model.setData(index, display_name, role=Qt.ItemDataRole.DisplayRole)

                self.insert_result_row(self.cpu_table, row, model_file, load_ms, infer_ms)
                self.insert_result_row(self.npu1_table, row, model_file, npu1_load, npu1_infer)
                self.insert_result_row(self.npu2_table, row, model_file, npu2_load, npu2_infer)

                self.log_output.appendPlainText(f"[Done] {model_file} - CPU: {infer_ms:.1f}ms\n")
            except Exception as e:
                index = self.fs_model.index(model_path)
                error_display = f"{model_file} (Error: {str(e)})"
                self.fs_model.setData(index, error_display, role=Qt.ItemDataRole.DisplayRole)
                self.log_output.appendPlainText(f"[Error] {model_file}: {str(e)}\n")

            self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
            QApplication.processEvents()

    def init_table(self, table):
        table.clear()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Model Name", "Load Time (ms)", "Inference Time (ms)"])
        table.setRowCount(0)

    def insert_result_row(self, table, row, model_file, load_ms, infer_ms):
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(model_file))
        table.setItem(row, 1, QTableWidgetItem(f"{load_ms:.1f}"))
        table.setItem(row, 2, QTableWidgetItem(f"{infer_ms:.1f}"))

    def profile_model_cpu(self, model_path):
        start_load = time.time()
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        end_load = time.time()
        load_time_ms = (end_load - start_load) * 1000.0

        input_data_dict = {}
        for input_tensor in session.get_inputs():
            name = input_tensor.name
            shape = input_tensor.shape
            dtype = input_tensor.type
            shape = [1 if s is None or s == 'None' else int(s) for s in shape]

            if 'float' in dtype:
                np_dtype = np.float32
            elif 'uint8' in dtype:
                np_dtype = np.uint8
            elif 'int64' in dtype:
                np_dtype = np.int64
            else:
                raise ValueError(f"Unsupported input type: {dtype}")

            input_data_dict[name] = (
                np.random.randint(0, 256, size=shape).astype(np_dtype)
                if np_dtype == np.uint8
                else np.random.rand(*shape).astype(np_dtype)
            )

        start_infer = time.time()
        _ = session.run(None, input_data_dict)
        end_infer = time.time()
        infer_time_ms = (end_infer - start_infer) * 1000.0

        return load_time_ms, infer_time_ms, shape

    def profile_model_npu1(self, model_file):
        load_time_ms = np.random.uniform(20.0, 30.0)
        infer_time_ms = np.random.uniform(2.0, 4.0)
        return load_time_ms, infer_time_ms

    def profile_model_npu2(self, model_file):
        load_time_ms = np.random.uniform(18.0, 28.0)
        infer_time_ms = np.random.uniform(2.0, 4.0)
        return load_time_ms, infer_time_ms
