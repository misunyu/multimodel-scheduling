import os
import time
import numpy as np
import onnx
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

CUSTOM_OP_PREFIXES = ["com.neubla"]

class ONNXProfiler(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("onnx_profiler_display_modify.ui", self)

        self.folder_input = self.findChild(QLineEdit, "folder_input")
        self.browse_button = self.findChild(QPushButton, "browse_button")
        self.profile_button = self.findChild(QPushButton, "profile_button")
        self.generate_button = self.findChild(QPushButton, "generate_button")
        self.model_tree_view = self.findChild(QTreeView, "model_tree_view")
        self.log_output = self.findChild(QPlainTextEdit, "log_output")

        self.cpu_table = self.findChild(QTableWidget, "cpu_table")
        self.npu1_table = self.findChild(QTableWidget, "npu1_table")
        self.npu2_table = self.findChild(QTableWidget, "npu2_table")

        for table in [self.cpu_table, self.npu1_table, self.npu2_table]:
            header = table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        self.fs_model = QFileSystemModel()
        self.fs_model.setReadOnly(True)
        self.fs_model.setNameFilters(["*.onnx", "*.o"])
        self.fs_model.setNameFilterDisables(False)
        self.model_tree_view.setModel(self.fs_model)
        self.model_tree_view.setMinimumWidth(500)
        self.model_tree_view.header().setStretchLastSection(True)
        self.model_tree_view.header().setDefaultSectionSize(300)

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
        o_files = []

        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                full_path = os.path.join(dirpath, f)
                if f.endswith(".onnx"):
                    onnx_files.append(full_path)
                elif f.endswith(".o"):
                    o_files.append(full_path)

        valid_model_onnx = {}  # key: model_name, value: (load, infer)

        # CPU profiling (only record successful runs)
        for path in onnx_files:
            try:
                if self.contains_custom_op(path):
                    self.log_output.appendPlainText(f"[Skip] {path} contains custom ops\n")
                    continue

                load_ms, infer_ms, _ = self.profile_model_cpu(path)
                rel_path = os.path.relpath(path, root_folder)
                self.insert_result_row(self.cpu_table, rel_path, load_ms, infer_ms)

                # only count for */model/*.onnx
                parts = rel_path.split(os.sep)
                if len(parts) == 3 and parts[1] == "model" and parts[2].endswith(".onnx"):
                    model_key = parts[0]
                    if model_key not in valid_model_onnx:
                        valid_model_onnx[model_key] = [0.0, 0.0]
                    valid_model_onnx[model_key][0] += load_ms
                    valid_model_onnx[model_key][1] += infer_ms

            except Exception as e:
                self.log_output.appendPlainText(f"[Error] Skipping {path}: {str(e)}\n")

        # NPU profiling with dummy values
        for path in o_files:
            name = os.path.relpath(path, root_folder)
            load_npu1 = np.random.uniform(1.0, 5.0)
            infer_npu1 = np.random.uniform(10.0, 50.0)
            self.insert_result_row(self.npu1_table, name, load_npu1, infer_npu1)

            load_npu2 = load_npu1 * np.random.uniform(0.9, 1.1)
            infer_npu2 = infer_npu1 * np.random.uniform(0.9, 1.1)
            self.insert_result_row(self.npu2_table, name, load_npu2, infer_npu2)

        # --- Total Summary ---
        self.total_table = self.findChild(QTableWidget, "total_table")
        if self.total_table:
            self.total_table.clear()
            self.total_table.setColumnCount(3)
            self.total_table.setHorizontalHeaderLabels([
                "Model",
                "CPU Only Total Time (ms)",
                "CPU + NPU Combined Time (ms)"
            ])
            self.total_table.setRowCount(0)

            header = self.total_table.horizontalHeader()
            header.setStretchLastSection(True)
            for i in range(3):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

            # NPU 평균 추론 시간 계산
            def collect_npu_times(table):
                times = {}
                for row in range(table.rowCount()):
                    name_item = table.item(row, 0)
                    if not name_item:
                        continue
                    parts = name_item.text().split(os.sep)
                    if len(parts) < 1:
                        continue
                    model_name = parts[0]
                    try:
                        infer = float(table.item(row, 2).text())
                        if model_name not in times:
                            times[model_name] = []
                        times[model_name].append(infer)
                    except:
                        continue
                return times

            npu1_times = collect_npu_times(self.npu1_table)
            npu2_times = collect_npu_times(self.npu2_table)

            all_models = set(valid_model_onnx.keys()).union(npu1_times.keys()).union(npu2_times.keys())

            for model in sorted(all_models):
                cpu_total = sum(valid_model_onnx.get(model, [0.0, 0.0]))
                npu_total = np.mean(npu1_times.get(model, []) + npu2_times.get(model, [])) if (
                            npu1_times.get(model) or npu2_times.get(model)) else 0.0
                combined = cpu_total + npu_total

                row = self.total_table.rowCount()
                self.total_table.insertRow(row)
                self.total_table.setItem(row, 0, QTableWidgetItem(model))
                self.total_table.setItem(row, 1, QTableWidgetItem(f"{cpu_total:.1f}"))
                self.total_table.setItem(row, 2, QTableWidgetItem(f"{combined:.1f}"))

    def init_table(self, table):
        table.clear()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Model Name", "Load Time (ms)", "Inference Time (ms)"])
        table.setRowCount(0)

    def insert_result_row(self, table, model_file, load_ms, infer_ms):
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(model_file))
        table.setItem(row, 1, QTableWidgetItem(f"{load_ms:.1f}"))
        table.setItem(row, 2, QTableWidgetItem(f"{infer_ms:.1f}"))

    def safe_shape_value(self, s):
        try:
            return 1 if s is None or s == 'None' else int(s)
        except (ValueError, TypeError):
            return 1

    def get_dummy_input(self, input_tensor):
        name = input_tensor.name
        shape = [self.safe_shape_value(s) for s in input_tensor.shape]
        if len(shape) < 3:
            shape = [1, 3, 224, 224]

        dtype = input_tensor.type
        if 'float' in dtype:
            np_dtype = np.float32
        elif 'uint8' in dtype:
            np_dtype = np.uint8
        elif 'int64' in dtype:
            np_dtype = np.int64
        else:
            raise ValueError(f"Unsupported input type: {dtype}")

        data = (
            np.random.randint(0, 256, size=shape).astype(np_dtype)
            if np_dtype == np.uint8 else
            np.random.rand(*shape).astype(np_dtype)
        )
        return name, data

    def profile_model_cpu(self, model_path):
        start_load = time.time()
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        end_load = time.time()
        load_time_ms = (end_load - start_load) * 1000.0

        input_data_dict = {}
        for input_tensor in session.get_inputs():
            name, data = self.get_dummy_input(input_tensor)
            input_data_dict[name] = data

        start_infer = time.time()
        _ = session.run(None, input_data_dict)
        end_infer = time.time()
        infer_time_ms = (end_infer - start_infer) * 1000.0

        return load_time_ms, infer_time_ms, []

    def contains_custom_op(self, onnx_path):
        try:
            model = onnx.load(onnx_path)
            for node in model.graph.node:
                if any(node.op_type.startswith(prefix) for prefix in CUSTOM_OP_PREFIXES):
                    return True
            return False
        except Exception as e:
            self.log_output.appendPlainText(f"[Error] ONNX parse failed: {onnx_path}: {e}\n")
            return True
