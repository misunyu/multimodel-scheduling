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
# from NeublaDriver import NeublaDriver

class ONNXProfiler(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi("onnx_profiler_display_modify.ui", self)

        # UI 연결
        self.folder_input = self.findChild(QLineEdit, "folder_input")
        self.browse_button = self.findChild(QPushButton, "browse_button")
        self.profile_button = self.findChild(QPushButton, "profile_button")
        self.generate_button = self.findChild(QPushButton, "generate_button")
        self.model_tree_view = self.findChild(QTreeView, "model_tree_view")
        self.log_output = self.findChild(QPlainTextEdit, "log_output") or self.findChild(QPlainTextEdit, "log_textbox")

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

        # 초기화
        self.init_table(self.cpu_table)
        self.init_table(self.npu1_table)
        self.init_table(self.npu2_table)

        # 1. ONNX 파일 프로파일링 (CPU)
        onnx_files = []
        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                if f.endswith(".onnx"):
                    onnx_files.append(os.path.join(dirpath, f))

        for model_path in onnx_files:
            model_file = os.path.basename(model_path)
            self.log_output.appendPlainText(f"[Start] Loading {model_file}...")
            QApplication.processEvents()

            try:
                load_ms, infer_ms, input_shape = self.profile_model_cpu(model_path)
                self.insert_result_row(self.cpu_table, model_file, load_ms, infer_ms)

                index = self.fs_model.index(model_path)
                display_name = f"{model_file} (CPU: {infer_ms:.1f}ms)"
                self.fs_model.setData(index, display_name, role=Qt.ItemDataRole.DisplayRole)

                self.log_output.appendPlainText(f"[Done] {model_file} - CPU: {infer_ms:.1f}ms\n")

            except Exception as e:
                index = self.fs_model.index(model_path)
                error_display = f"{model_file} (Error: {str(e)})"
                self.fs_model.setData(index, error_display, role=Qt.ItemDataRole.DisplayRole)
                self.log_output.appendPlainText(f"[Error] {model_file}: {str(e)}\n")

            self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
            QApplication.processEvents()

        # 2. .o 파일 프로파일링 (NPU1/NPU2)
        o_files = list({os.path.join(dirpath, f)
                        for dirpath, _, filenames in os.walk(root_folder)
                        for f in filenames if f.endswith(".o")})

        for o_path in o_files:
            o_file = os.path.basename(o_path)
            self.log_output.appendPlainText(f"[Start] Loading {o_file}...")
            QApplication.processEvents()

            try:
                # NPU1
                o_file1, npu1_load, npu1_infer = self.profile_model_npu(o_path, "NPU1")
                self.insert_result_row(self.npu1_table, o_file1, npu1_load, npu1_infer)

                # NPU2
                o_file2, npu2_load, npu2_infer = self.profile_model_npu(o_path, "NPU2")
                self.insert_result_row(self.npu2_table, o_file2, npu2_load, npu2_infer)

                self.log_output.appendPlainText(
                    f"[Done] {o_file} - NPU1: {npu1_infer:.1f}ms / NPU2: {npu2_infer:.1f}ms\n"
                )
            except Exception as e:
                self.log_output.appendPlainText(f"[Error] {o_file}: {str(e)}\n")

            self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
            QApplication.processEvents()

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

    def profile_model_npu(self, o_path, label):
        o_file = os.path.basename(o_path)
        self.log_output.appendPlainText(f"[{label}] Profiling {o_file}")
        print(f"[{label}] Profiling {o_file}")

        # Simulated execution
        start_load = time.time()
        time.sleep(0.01)
        end_load = time.time()
        load_time_ms = (end_load - start_load) * 1000.0

        start_infer = time.time()
        time.sleep(0.003)
        end_infer = time.time()
        infer_time_ms = (end_infer - start_infer) * 1000.0

        # npu_num = 0
        # if label == "NPU2":
        #     npu_num = 1

        # load_time_ms, infer_time_ms = self.process_yolo_npu(npu_num, o_path)

        return o_file, load_time_ms, infer_time_ms


    # def process_yolo_npu(npu_num, o_path):
    #     # print(f"==== Start Driver #{npu_num} ====")
    #     try:
    #         driver = NeublaDriver()
    #         assert driver.Init(npu_num) == 0
    #
    #         start_load = time.time()
    #         assert driver.LoadModel(o_path) == 0
    #         end_load = time.time()
    #         load_time_ms = (end_load - start_load) * 1000.0
    #
    #         random_input = np.random.rand(3, 608, 608).astype(np.uint8)
    #         input_data = random_input.tobytes()
    #
    #         start_infer = time.time()
    #         assert driver.SendInput(input_data, 3 * 608 * 608) == 0
    #         assert driver.Launch() == 0
    #         raw_outputs = driver.ReceiveOutputs()
    #         end_infer = time.time()
    #         infer_time_ms = (end_infer - start_infer) * 1000.0
    #
    #         assert driver.Close() == 0
    #
    #     except Exception as e:
    #         assert driver.Close() == 0
    #         print(e)
    #         print("Error occured. Closed successfully.")
    #         exit()
    #
    #     return load_time_ms, infer_time_ms