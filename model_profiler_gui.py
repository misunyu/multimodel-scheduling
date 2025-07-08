import os
import time
import numpy as np
import onnx
import onnxruntime as ort
import json
from collections import defaultdict

from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QLineEdit, QPushButton,
    QFileDialog, QTreeView, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QApplication,
    QHeaderView, QVBoxLayout, QHBoxLayout, QCheckBox, QTabWidget, QWidget, QDialog,
    QSplitter, QTableWidgetItem, QAction
)
from PyQt5.QtWidgets import QFileSystemModel
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# from NeublaDriver import NeublaDriver


CUSTOM_OP_PREFIXES = ["com.neubla"]

class ONNXProfiler(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("onnx_profiler_display.ui", self)

        # Initialize profiled_times and profiled_models attributes
        self.profiled_times = []
        self.profiled_models = []

        self.enable_npu2_checkbox = self.findChild(QCheckBox, "npu2_enable_checkbox")
        self.result_tabs = self.findChild(QTabWidget, "result_tab_widget")
        self.npu2_tab = self.findChild(QWidget, "npu2_tab")

        main_layout = self.findChild(QHBoxLayout, "mainLayout")
        if main_layout:
            main_layout.setStretch(0, 3)
            main_layout.setStretch(1, 7)

        # 탭 활성/비활성 함수 정의 및 연결
        def update_npu2_tab_enabled():
            index = self.result_tabs.indexOf(self.npu2_tab)
            if index != -1:
                self.result_tabs.setTabEnabled(index, self.enable_npu2_checkbox.isChecked())

        self.enable_npu2_checkbox.stateChanged.connect(update_npu2_tab_enabled)
        update_npu2_tab_enabled()

        self.folder_input = self.findChild(QLineEdit, "folder_input")
        self.browse_button = self.findChild(QPushButton, "browse_button")
        self.profile_button = self.findChild(QPushButton, "profile_button")
        self.generate_button = self.findChild(QPushButton, "generate_button")
        self.model_tree_view = self.findChild(QTreeView, "model_tree_view")
        self.log_output = self.findChild(QPlainTextEdit, "log_output")
        self.total_table = self.findChild(QTableWidget, "total_table")

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

        self.cpu_table = self.findChild(QTableWidget, "cpu_table")
        self.npu1_table = self.findChild(QTableWidget, "npu1_table")
        self.npu2_table = self.findChild(QTableWidget, "npu2_table")

        self.npu2_table.setVisible(self.enable_npu2_checkbox.isChecked())
        self.enable_npu2_checkbox.stateChanged.connect(
            lambda: self.npu2_table.setVisible(self.enable_npu2_checkbox.isChecked())
        )

        for table in [self.cpu_table, self.npu1_table, self.npu2_table]:
            header = table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeToContents)

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

        default_folder = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(default_folder):
            default_folder = os.getcwd()

        self.folder_input.setText(default_folder)
        self.browse_button.clicked.connect(self.browse_folder)
        self.profile_button.clicked.connect(self.run_profiling)

        # self.generate_button.clicked.connect(self.highlight_deploy_results)
        self.generate_button.clicked.connect(
            lambda: self.highlight_deploy_results(self.profiled_times, self.profiled_models))

        self.set_tree_root(default_folder)
        QTimer.singleShot(100, lambda: self.expand_parents_of_onnx_files(default_folder))

        self.assignment_results = []  # (model_name, device) 목록 저장용

        self.show_assignment_button = self.findChild(QPushButton, "show_assignment_button")
        if self.show_assignment_button:
            self.show_assignment_button.clicked.connect(self.show_partition_assignment_dialog)

        # Connect Load Test Data menu action
        self.actionLoad_Test_Data = self.findChild(QAction, "actionLoad_Test_Data")
        if self.actionLoad_Test_Data:
            self.actionLoad_Test_Data.triggered.connect(self.load_sample_data)

        self.load_sample_button = self.findChild(QPushButton, "load_sample_button")
        if self.load_sample_button:
            self.load_sample_button.clicked.connect(self.load_sample_data)

        self.actionLoad_Runtime_Log = self.findChild(QAction, "actionLoad_Runtime_Log")
        if self.actionLoad_Runtime_Log:
            self.actionLoad_Runtime_Log.triggered.connect(self.update_frequencies_from_log)

        self.model_freq_table.horizontalHeader().setStretchLastSection(True)
        self.populate_model_frequencies(["resnet50", "yolov3_small", "yolov3_big"])

    def populate_model_frequencies(self, model_names: list[str]):
        if not hasattr(self, "model_freq_table"):
            return

        self.model_freq_table.setRowCount(len(model_names))
        self.model_freq_table.setColumnCount(2)
        self.model_freq_table.setHorizontalHeaderLabels(["Model", "Freq"])

        # 모델별 기본 frequency 설정
        default_freqs = {
            "resnet50": "1",
            "yolov3_small": "60",
            "yolov3_big": "40"
        }

        for row, model_name in enumerate(model_names):
            # 모델명만 추출 (디렉토리나 확장자 제거)
            base_name = model_name.split(os.sep)[0]  # 'resnet50', 'yolov3_small' 등

            # 모델명 셀
            model_item = QTableWidgetItem(model_name)
            model_item.setFlags(Qt.ItemIsEnabled)
            self.model_freq_table.setItem(row, 0, model_item)

            # frequency 셀
            freq_value = default_freqs.get(base_name, "1")
            freq_item = QTableWidgetItem(freq_value)
            freq_item.setTextAlignment(Qt.AlignCenter)
            self.model_freq_table.setItem(row, 1, freq_item)

        header = self.model_freq_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.model_freq_table.viewport().update()

        self.model_freq_table.resizeColumnsToContents()

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

        valid_model_onnx = {}

        for path in onnx_files:
            try:
                if self.contains_custom_op(path):
                    self.log_output.appendPlainText(f"[Skip] {path} contains custom ops\n")
                    continue

                load_ms, infer_ms, _ = self.profile_model_cpu(path)
                rel_path = os.path.relpath(path, root_folder)
                self.insert_result_row(self.cpu_table, rel_path, load_ms, infer_ms)

                self.log_output.appendPlainText(f"[CPU] {rel_path}")
                self.log_output.appendPlainText(f"       Load: {load_ms:.1f} ms, Inference: {infer_ms:.1f} ms\n")

                parts = rel_path.split(os.sep)
                if len(parts) == 3 and parts[1] == "model" and parts[2].endswith(".onnx"):
                    model_key = parts[0]
                    if model_key not in valid_model_onnx:
                        valid_model_onnx[model_key] = [0.0, 0.0]
                    valid_model_onnx[model_key][0] += load_ms
                    valid_model_onnx[model_key][1] += infer_ms

            except Exception as e:
                self.log_output.appendPlainText(f"[Error] Skipping {path}: {str(e)}\n")

        for path in o_files:
            name = os.path.relpath(path, root_folder)
            load_npu1, infer_npu1, _ = self.profile_model_npu(path, "NPU1")
            self.insert_result_row(self.npu1_table, name, load_npu1, infer_npu1)

            self.log_output.appendPlainText(f"[NPU1] {name}")
            self.log_output.appendPlainText(f"       Load: {load_npu1:.1f} ms, Inference: {infer_npu1:.1f} ms\n")

            if self.enable_npu2_checkbox and self.enable_npu2_checkbox.isChecked():
                load_npu2, infer_npu2, _ = self.profile_model_npu(path, "NPU2")
                self.insert_result_row(self.npu2_table, name, load_npu2, infer_npu2)

                self.log_output.appendPlainText(f"[NPU2] {name}")
                self.log_output.appendPlainText(f"       Load: {load_npu2:.1f} ms, Inference: {infer_npu2:.1f} ms\n")

        if self.total_table:
            self.total_table.clear()
            self.total_table.setColumnCount(6)
            self.total_table.setHorizontalHeaderLabels([
                "Model",
                "CPU Inf. (ms)",
                "NPU1 Load (ms)",
                "NPU1 + CPU Inf. (ms)",
                "NPU2 Load (ms)",
                "NPU2 + CPU Inf. (ms)"
            ])
            self.total_table.setRowCount(0)

            header = self.total_table.horizontalHeader()
            header.setStretchLastSection(True)
            for i in range(6):
                header.setSectionResizeMode(i, QHeaderView.Stretch)

            def collect_values(table, col_index):
                values = {}
                for row in range(table.rowCount()):
                    name_item = table.item(row, 0)
                    if not name_item:
                        continue
                    model_name = name_item.text().split(os.sep)[0]
                    try:
                        val = float(table.item(row, col_index).text())
                        values.setdefault(model_name, []).append(val)
                    except:
                        continue
                return {k: np.mean(v) for k, v in values.items()}

            cpu_infer_per_partition = {}
            for row in range(self.cpu_table.rowCount()):
                path_item = self.cpu_table.item(row, 0)
                infer_item = self.cpu_table.item(row, 2)
                if not path_item or not infer_item:
                    continue
                rel_path = path_item.text()
                infer_time = float(infer_item.text())
                model_key = rel_path.split(os.sep)[0]
                part_name = os.path.basename(rel_path)
                if "_p0" in part_name or "_p2" in part_name:
                    cpu_infer_per_partition.setdefault(model_key, []).append(infer_time)

            npu1_load = collect_values(self.npu1_table, 1)
            npu1_infer = collect_values(self.npu1_table, 2)
            npu2_load = collect_values(self.npu2_table, 1)
            npu2_infer = collect_values(self.npu2_table, 2)

            all_models = set(valid_model_onnx.keys()).union(
                npu1_load.keys(), npu1_infer.keys(), npu2_load.keys(), npu2_infer.keys()
            )

            # Prepare profiled_times and profiled_models for highlight_deploy_results
            self.profiled_times = []
            self.profiled_models = []

            for model in sorted(all_models):
                cpu_infer = valid_model_onnx.get(model, [0.0, 0.0])[1]

                load1 = npu1_load.get(model, 0.0)
                infer1_base = npu1_infer.get(model, 0.0)
                extra_cpu_infer = sum(cpu_infer_per_partition.get(model, []))  # ✅ 추가
                infer1 = infer1_base + extra_cpu_infer  # ✅ 합산

                load2 = npu2_load.get(model, 0.0)
                infer2_base = npu2_infer.get(model, 0.0)
                infer2 = infer2_base + extra_cpu_infer

                row = self.total_table.rowCount()
                self.total_table.insertRow(row)
                self.total_table.setItem(row, 0, QTableWidgetItem(model))
                self.total_table.setItem(row, 1, QTableWidgetItem(f"{cpu_infer:.1f}"))
                self.total_table.setItem(row, 2, QTableWidgetItem(f"{load1:.1f}"))
                self.total_table.setItem(row, 3, QTableWidgetItem(f"{infer1:.1f}"))  # ✅ 수정
                self.total_table.setItem(row, 4, QTableWidgetItem(f"{load2:.1f}"))
                self.total_table.setItem(row, 5, QTableWidgetItem(f"{infer2:.1f}"))

                # Store profiling data for highlight_deploy_results
                self.profiled_times.append((cpu_infer, infer1, infer2))
                self.profiled_models.append((row, model))

            # Calculate and display total values
            if self.total_table and self.total_table.rowCount() > 0:
                # Initialize totals
                cpu_infer_total = 0.0
                npu1_load_total = 0.0
                npu1_infer_total = 0.0
                npu2_load_total = 0.0
                npu2_infer_total = 0.0

                # Calculate totals
                for row in range(self.total_table.rowCount()):
                    try:
                        cpu_infer_item = self.total_table.item(row, 1)
                        npu1_load_item = self.total_table.item(row, 2)
                        npu1_infer_item = self.total_table.item(row, 3)
                        npu2_load_item = self.total_table.item(row, 4)
                        npu2_infer_item = self.total_table.item(row, 5)

                        if cpu_infer_item:
                            cpu_infer_total += float(cpu_infer_item.text())
                        if npu1_load_item:
                            npu1_load_total += float(npu1_load_item.text())
                        if npu1_infer_item:
                            npu1_infer_total += float(npu1_infer_item.text())
                        if npu2_load_item:
                            npu2_load_total += float(npu2_load_item.text())
                        if npu2_infer_item:
                            npu2_infer_total += float(npu2_infer_item.text())
                    except Exception as e:
                        if self.log_output:
                            self.log_output.appendPlainText(f"[Warning] Error calculating totals: {e}")

                # Add total row
                total_row = self.total_table.rowCount()
                self.total_table.insertRow(total_row)

                # Set total values with bold font
                bold_font = QFont()
                bold_font.setBold(True)

                total_item = QTableWidgetItem("Total")
                total_item.setFont(bold_font)
                self.total_table.setItem(total_row, 0, total_item)

                cpu_infer_total_item = QTableWidgetItem(f"{cpu_infer_total:.1f}")
                cpu_infer_total_item.setFont(bold_font)
                self.total_table.setItem(total_row, 1, cpu_infer_total_item)

                npu1_load_total_item = QTableWidgetItem(f"{npu1_load_total:.1f}")
                npu1_load_total_item.setFont(bold_font)
                self.total_table.setItem(total_row, 2, npu1_load_total_item)

                npu1_infer_total_item = QTableWidgetItem(f"{npu1_infer_total:.1f}")
                npu1_infer_total_item.setFont(bold_font)
                self.total_table.setItem(total_row, 3, npu1_infer_total_item)

                npu2_load_total_item = QTableWidgetItem(f"{npu2_load_total:.1f}")
                npu2_load_total_item.setFont(bold_font)
                self.total_table.setItem(total_row, 4, npu2_load_total_item)

                npu2_infer_total_item = QTableWidgetItem(f"{npu2_infer_total:.1f}")
                npu2_infer_total_item.setFont(bold_font)
                self.total_table.setItem(total_row, 5, npu2_infer_total_item)

                # Set background color for the total row
                for col in range(self.total_table.columnCount()):
                    item = self.total_table.item(total_row, col)
                    if item:
                        item.setBackground(QBrush(QColor(230, 230, 230)))

    def load_sample_data(self):
        # 시뮬레이션 데이터 정의
        simulated_profiles = {
            "resnet50": {
                "NPU1": (15.8, 38.6),
                "NPU2": (77.1, 38.8),
            },
            "yolov3_small": {
                "NPU1": (104.3, 60.9),
                "NPU2": (430.6, 82.7),
            },
            "yolov3_big": {
                "NPU1": (107.0, 87.4),
                "NPU2": (467.5, 110.4),
            },
        }

        # ✅ CPU 테이블: 스크린샷과 동일한 경로 및 값 사용 (.onnx 경로 포함)
        cpu_data = [
            ("yolov3_small/model/yolov3_small.onnx", 462.4, 163.4),
            ("yolov3_small/partitions/yolov3_small_neubla_p2.onnx", 6.9, 12.9),
            ("yolov3_small/partitions/yolov3_small_neubla_p1.onnx", 83.5, 58.8),
            ("yolov3_small/partitions/yolov3_small_neubla_p0.onnx", 0.6, 0.1),
            ("yolov3_big/model/yolov3_big.onnx", 365.5, 163.3),
            ("yolov3_big/partitions/yolov3_big_neubla_p0.onnx", 0.6, 0.1),
            ("yolov3_big/partitions/yolov3_big_neubla_p2.onnx", 7.4, 12.7),
            ("yolov3_big/partitions/yolov3_big_neubla_p1.onnx", 83.3, 58.5),
            ("resnet50/model/resnet50.onnx", 88.3, 10.8),
            ("resnet50/partitions/resnet50_neubla_p0.onnx", 0.5, 0.1),
        ]

        # NPU1 데이터 (기존 유지)
        npu1_data = [
            ("resnet50_neubla_p1.o", *simulated_profiles["resnet50"]["NPU1"]),
            ("yolov3_big_neubla_p1.o", *simulated_profiles["yolov3_big"]["NPU1"]),
            ("yolov3_small_neubla_p1.o", *simulated_profiles["yolov3_small"]["NPU1"]),
        ]

        # NPU2 데이터 (기존 유지)
        npu2_data = [
            ("resnet50_neubla_p1.o", *simulated_profiles["resnet50"]["NPU2"]),
            ("yolov3_big_neubla_p1.o", *simulated_profiles["yolov3_big"]["NPU2"]),
            ("yolov3_small_neubla_p1.o", *simulated_profiles["yolov3_small"]["NPU2"]),
        ]

        def fill_table(table, data):
            table.clear()
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Model", "Load (ms)", "Inf. (ms)"])
            table.setRowCount(len(data))
            for row, (model, load, inf) in enumerate(data):
                table.setItem(row, 0, QTableWidgetItem(model))
                table.setItem(row, 1, QTableWidgetItem(f"{load:.1f}"))
                table.setItem(row, 2, QTableWidgetItem(f"{inf:.1f}"))

            header = table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeToContents)

        fill_table(self.cpu_table, cpu_data)
        fill_table(self.npu1_table, npu1_data)
        if self.enable_npu2_checkbox.isChecked():
            fill_table(self.npu2_table, npu2_data)

        # Initialize total_table
        if self.total_table:
            self.total_table.clear()
            self.total_table.setColumnCount(6)
            self.total_table.setHorizontalHeaderLabels([
                "Model",
                "CPU Inf. (ms)",
                "NPU1 Load (ms)",
                "NPU1 + CPU Inf. (ms)",
                "NPU2 Load (ms)",
                "NPU2 + CPU Inf. (ms)"
            ])
            self.total_table.setRowCount(0)

            header = self.total_table.horizontalHeader()
            header.setStretchLastSection(True)
            for i in range(6):
                header.setSectionResizeMode(i, QHeaderView.Stretch)

            # Populate total_table with data from CPU and NPU tables
            valid_model_onnx = {}
            cpu_infer_per_partition = {}

            # Process CPU data
            for model_path, _, infer_time in cpu_data:
                parts = model_path.split(os.sep)
                if len(parts) == 3 and parts[1] == "model" and parts[2].endswith(".onnx"):
                    model_key = parts[0]
                    if model_key not in valid_model_onnx:
                        valid_model_onnx[model_key] = [0.0, 0.0]
                    valid_model_onnx[model_key][1] = infer_time  # CPU inference time

                model_key = model_path.split(os.sep)[0]
                part_name = os.path.basename(model_path)
                if "_p0" in part_name or "_p2" in part_name:
                    cpu_infer_per_partition.setdefault(model_key, []).append(infer_time)

            # Process NPU data
            npu1_load = {}
            npu1_infer = {}
            npu2_load = {}
            npu2_infer = {}

            # Map NPU file names to model folder names
            npu_file_to_model = {}
            for model_name in simulated_profiles.keys():
                npu_file_to_model[f"{model_name}_neubla_p1.o"] = model_name

            for model_path, load_time, infer_time in npu1_data:
                model_key = npu_file_to_model.get(model_path, model_path.split("_")[0])
                npu1_load[model_key] = load_time
                npu1_infer[model_key] = infer_time

            for model_path, load_time, infer_time in npu2_data:
                model_key = npu_file_to_model.get(model_path, model_path.split("_")[0])
                npu2_load[model_key] = load_time
                npu2_infer[model_key] = infer_time

            # Combine all models
            all_models = set(valid_model_onnx.keys()).union(
                npu1_load.keys(), npu1_infer.keys(), npu2_load.keys(), npu2_infer.keys()
            )

            # Prepare profiled_times and profiled_models for highlight_deploy_results
            self.profiled_times = []
            self.profiled_models = []

            # Add rows to total_table
            for model in sorted(all_models):
                cpu_infer = valid_model_onnx.get(model, [0.0, 0.0])[1]

                load1 = npu1_load.get(model, 0.0)
                infer1_base = npu1_infer.get(model, 0.0)
                extra_cpu_infer = sum(cpu_infer_per_partition.get(model, []))
                infer1 = infer1_base + extra_cpu_infer

                load2 = npu2_load.get(model, 0.0)
                infer2_base = npu2_infer.get(model, 0.0)
                infer2 = infer2_base + extra_cpu_infer

                row = self.total_table.rowCount()
                self.total_table.insertRow(row)
                self.total_table.setItem(row, 0, QTableWidgetItem(model))
                self.total_table.setItem(row, 1, QTableWidgetItem(f"{cpu_infer:.1f}"))
                self.total_table.setItem(row, 2, QTableWidgetItem(f"{load1:.1f}"))
                self.total_table.setItem(row, 3, QTableWidgetItem(f"{infer1:.1f}"))
                self.total_table.setItem(row, 4, QTableWidgetItem(f"{load2:.1f}"))
                self.total_table.setItem(row, 5, QTableWidgetItem(f"{infer2:.1f}"))

                # Store profiling data for highlight_deploy_results
                self.profiled_times.append((cpu_infer, infer1, infer2))
                self.profiled_models.append((row, model))

        # Apply highlighting
        self.highlight_deploy_results(self.profiled_times, self.profiled_models)

        # Calculate and display total values
        if self.total_table and self.total_table.rowCount() > 0:
            # Initialize totals
            cpu_infer_total = 0.0
            npu1_load_total = 0.0
            npu1_infer_total = 0.0
            npu2_load_total = 0.0
            npu2_infer_total = 0.0

            # Calculate totals
            for row in range(self.total_table.rowCount()):
                try:
                    cpu_infer_item = self.total_table.item(row, 1)
                    npu1_load_item = self.total_table.item(row, 2)
                    npu1_infer_item = self.total_table.item(row, 3)
                    npu2_load_item = self.total_table.item(row, 4)
                    npu2_infer_item = self.total_table.item(row, 5)

                    if cpu_infer_item:
                        cpu_infer_total += float(cpu_infer_item.text())
                    if npu1_load_item:
                        npu1_load_total += float(npu1_load_item.text())
                    if npu1_infer_item:
                        npu1_infer_total += float(npu1_infer_item.text())
                    if npu2_load_item:
                        npu2_load_total += float(npu2_load_item.text())
                    if npu2_infer_item:
                        npu2_infer_total += float(npu2_infer_item.text())
                except Exception as e:
                    if self.log_output:
                        self.log_output.appendPlainText(f"[Warning] Error calculating totals: {e}")

            # Add total row
            total_row = self.total_table.rowCount()
            self.total_table.insertRow(total_row)

            # Set total values with bold font
            bold_font = QFont()
            bold_font.setBold(True)

            total_item = QTableWidgetItem("Total")
            total_item.setFont(bold_font)
            self.total_table.setItem(total_row, 0, total_item)

            cpu_infer_total_item = QTableWidgetItem(f"{cpu_infer_total:.1f}")
            cpu_infer_total_item.setFont(bold_font)
            self.total_table.setItem(total_row, 1, cpu_infer_total_item)

            npu1_load_total_item = QTableWidgetItem(f"{npu1_load_total:.1f}")
            npu1_load_total_item.setFont(bold_font)
            self.total_table.setItem(total_row, 2, npu1_load_total_item)

            npu1_infer_total_item = QTableWidgetItem(f"{npu1_infer_total:.1f}")
            npu1_infer_total_item.setFont(bold_font)
            self.total_table.setItem(total_row, 3, npu1_infer_total_item)

            npu2_load_total_item = QTableWidgetItem(f"{npu2_load_total:.1f}")
            npu2_load_total_item.setFont(bold_font)
            self.total_table.setItem(total_row, 4, npu2_load_total_item)

            npu2_infer_total_item = QTableWidgetItem(f"{npu2_infer_total:.1f}")
            npu2_infer_total_item.setFont(bold_font)
            self.total_table.setItem(total_row, 5, npu2_infer_total_item)

            # Set background color for the total row
            for col in range(self.total_table.columnCount()):
                item = self.total_table.item(total_row, col)
                if item:
                    item.setBackground(QBrush(QColor(230, 230, 230)))

        if self.log_output:
            self.log_output.appendPlainText("[Info] 테스트 데이터가 로드되었습니다.")

    # === Table Management ===
    def init_table(self, table):
        table.clear()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Model", "Load (ms)", "Inf. (ms)"])
        table.setRowCount(0)



# === Table Management ===
    def insert_result_row(self, table, model_file, load_ms, infer_ms):
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(model_file))
        table.setItem(row, 1, QTableWidgetItem(f"{load_ms:.1f}"))
        table.setItem(row, 2, QTableWidgetItem(f"{infer_ms:.1f}"))


# === Inference Input Generation ===
    def safe_shape_value(self, s):
        try:
            return 1 if s is None or s == 'None' else int(s)
        except (ValueError, TypeError):
            return 1


# === Inference Input Generation ===
    def get_dummy_input(self, input_tensor):
        name = input_tensor.name
        shape = [self.safe_shape_value(s) for s in input_tensor.shape]

        # --- name 기반 예외 처리 (image_shape 같은 경우) ---
        if name.lower() == "image_shape" or shape == [2] or shape == [1, 2]:
            shape = [1, 2]  # 정확히 모델이 기대하는 2D 형태

        # --- fallback: 너무 작거나 정의 안 된 경우 기본값 제공 ---
        elif not shape or len(shape) < 3:
            shape = [1, 3, 224, 224]

        # --- 공간 차원 보정 (Conv 오류 방지) ---
        elif len(shape) == 4 and (shape[-1] < 4 or shape[-2] < 4):
            shape[-1] = max(shape[-1], 32)
            shape[-2] = max(shape[-2], 32)

        # --- dtype 처리 ---
        dtype = input_tensor.type
        if 'float' in dtype:
            np_dtype = np.float32
        elif 'uint8' in dtype:
            np_dtype = np.uint8
        elif 'int64' in dtype:
            np_dtype = np.int64
        else:
            raise ValueError(f"Unsupported input type: {dtype}")

        # --- dummy data 생성 ---
        if np_dtype == np.uint8:
            data = np.random.randint(0, 256, size=shape).astype(np_dtype)
        else:
            data = np.random.rand(*shape).astype(np_dtype)

        return name, data


# === Profiling ===
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

    def profile_model_npu(self, o_path, label):
        npu_num = 0 if label == "NPU1" else 1
        basename = os.path.basename(o_path)

        # # yolov3로 시작하는 경우 실제 NPU 실행
        # if os.path.basename(o_path).startswith("yolov3"):
        #     load_time_ms, infer_time_ms = self.process_yolo_npu(npu_num, o_path)
        #     return load_time_ms, infer_time_ms, []
        # elif os.path.basename(o_path).startswith("resnet50"):
        #     load_time_ms, infer_time_ms = self.process_resnet50_npu(npu_num, o_path)
        #     return load_time_ms, infer_time_ms, []
        # else:

        simulated_profiles = {
            "resnet50": {
                "NPU1": (15.8, 38.6),
                "NPU2": (77.1, 38.8),
            },
            "yolov3_small": {
                "NPU1": (104.3, 60.9),
                "NPU2": (430.6, 82.7),
            },
            "yolov3_big": {
                "NPU1": (107.0, 87.4),
                "NPU2": (467.5, 110.4),
            },
        }
        # 모델명 접두어 매칭
        matched_key = next((key for key in simulated_profiles if basename.startswith(key)), None)
        if matched_key and label in simulated_profiles[matched_key]:
            load_time_ms, infer_time_ms = simulated_profiles[matched_key][label]
            return load_time_ms, infer_time_ms, []

        # 시뮬레이션 값이 없는 경우: 기본 짧은 대기 시뮬레이션
        start_load = time.time()
        time.sleep(0.01)
        end_load = time.time()
        load_time_ms = (end_load - start_load) * 1000.0

        start_infer = time.time()
        time.sleep(0.003)
        end_infer = time.time()
        infer_time_ms = (end_infer - start_infer) * 1000.0

        return load_time_ms, infer_time_ms, []

    # def process_yolo_npu(self, npu_num, o_path):
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
    #         try:
    #             driver.Close()
    #         except:
    #             pass
    #         print(f"[Error] NPU{npu_num}: {e}")
    #         exit()
    #
    #     return load_time_ms, infer_time_ms
    #
    # def process_resnet50_npu(self, npu_num, o_path):
    #     try:
    #         driver = NeublaDriver()
    #         assert driver.Init(npu_num) == 0
    #
    #         start_load = time.time()
    #         assert driver.LoadModel(o_path) == 0
    #         end_load = time.time()
    #         load_time_ms = (end_load - start_load) * 1000.0
    #
    #         random_input = np.random.rand(3, 224, 224).astype(np.uint8)
    #         input_data = random_input.tobytes()
    #
    #         start_infer = time.time()
    #         assert driver.SendInput(input_data, 3 * 224 * 224) == 0
    #         assert driver.Launch() == 0
    #         raw_outputs = driver.ReceiveOutputs()
    #         end_infer = time.time()
    #         infer_time_ms = (end_infer - start_infer) * 1000.0
    #
    #         assert driver.Close() == 0
    #
    #     except Exception as e:
    #         try:
    #             driver.Close()
    #         except:
    #             pass
    #         print(f"[Error] NPU{npu_num}: {e}")
    #         exit()
    #
    #     return load_time_ms, infer_time_ms

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

    # === Analysis and UI Update ===
    def highlight_deploy_results(self, times, models):
        self.assignment_results = []

        # Check if times or models is empty
        if not times or not models:
            if self.log_output:
                self.log_output.appendPlainText("[Warning] No profiling data available for deployment.")
            return

        load = {"CPU": 0, "NPU1": 0, "NPU2": 0}
        assignments = []

        for idx, (cpu_t, npu1_t, npu2_t) in enumerate(times):
            model_name = models[idx][1]  # (row_index, model_name)

            # 1. Get frequency value for the model
            freq = 1
            if hasattr(self, "model_freq_table"):
                for row in range(self.model_freq_table.rowCount()):
                    name_item = self.model_freq_table.item(row, 0)
                    freq_item = self.model_freq_table.item(row, 1)
                    if name_item and name_item.text() == model_name and freq_item:
                        try:
                            freq = int(freq_item.text())
                        except ValueError:
                            freq = 1
                        break

            # 2. Determine which devices are available
            device_candidates = ["CPU", "NPU1"]
            if self.enable_npu2_checkbox and self.enable_npu2_checkbox.isChecked():
                device_candidates.append("NPU2")

            # 3. Choose the device that results in the lowest total load (considering frequency)
            best_device = min(
                device_candidates,
                key=lambda d: load[d] + freq * (
                    cpu_t if d == "CPU" else npu1_t if d == "NPU1" else npu2_t
                )
            )

            assignments.append((model_name, best_device))
            load[best_device] += freq * (
                cpu_t if best_device == "CPU" else npu1_t if best_device == "NPU1" else npu2_t
            )

        self.assignment_results = assignments

        # Update the display with the assignment results
        if hasattr(self, 'total_table') and self.total_table:
            for row in range(self.total_table.rowCount()):
                if row >= len(assignments):
                    break

                _, device = assignments[row]
                for col in range(self.total_table.columnCount()):
                    item = self.total_table.item(row, col)
                    if item:
                        # Clear any existing background
                        item.setBackground(QBrush())

                        # Set background based on device
                        if device == "CPU":
                            item.setBackground(QBrush(QColor(204, 230, 255)))  # Light blue for CPU
                        elif device == "NPU1":
                            item.setBackground(QBrush(QColor(255, 255, 204)))  # Light yellow for NPU1
                        elif device == "NPU2":
                            item.setBackground(QBrush(QColor(255, 214, 153)))  # Light orange for NPU2

    def find_partition_files(self, root_folder, model_prefix, device):
        npu_files = []
        cpu_partition_files = []

        model_base_dir = os.path.join(root_folder, model_prefix)

        if device == "CPU":
            model_dir = os.path.join(model_base_dir, "model")
            if os.path.isdir(model_dir):
                for f in sorted(os.listdir(model_dir)):
                    if f.endswith(".onnx") and f.startswith(model_prefix):
                        cpu_partition_files.append(f)

        elif device in ("NPU1", "NPU2"):
            npu_code_dir = os.path.join(model_base_dir, "npu_code")
            partitions_dir = os.path.join(model_base_dir, "partitions")

            o_file_basenames = set()

            # 1) .o 파일 수집
            if os.path.isdir(npu_code_dir):
                for f in sorted(os.listdir(npu_code_dir)):
                    if f.endswith(".o") and f.startswith(model_prefix):
                        npu_files.append(f)
                        o_file_basenames.add(os.path.splitext(f)[0])  # "yolov3_big_p1" 등

            # 2) .onnx 파일 중 .o와 같은 베이스 이름은 제외
            if os.path.isdir(partitions_dir):
                for f in sorted(os.listdir(partitions_dir)):
                    if f.endswith(".onnx") and f.startswith(model_prefix):
                        base = os.path.splitext(f)[0]
                        if base not in o_file_basenames:
                            cpu_partition_files.append(f)

        return npu_files, cpu_partition_files

    # These imports are already included at the top of the file
    # from PyQt5.QtGui import QColor, QBrush
    # from PyQt5.QtWidgets import QSplitter

    def create_inference_bar_chart(self):
        root_folder = self.folder_input.text().strip()

        # 파티션 수에 따라 적절한 높이 설정
        total_partitions = 0
        for model_prefix, device in self.assignment_results:
            npu_files, cpu_files = self.find_partition_files(root_folder, model_prefix, device)
            total_partitions += len(npu_files) + len(cpu_files)
            # 전체 모델만 있는 경우 1개 추가
            if device == "CPU" and not npu_files and not cpu_files:
                total_partitions += 1

        fig_height = max(3.5, 0.5 * total_partitions)
        fig = Figure(figsize=(6, fig_height))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        labels = []
        durations = []

        device_table_map = {
            "CPU": self.cpu_table,
            "NPU1": self.npu1_table,
            "NPU2": self.npu2_table,
        }

        # 추론 시간 가져오기: 파일명 끝이 같으면 매칭
        def get_inference_time(partition_name, primary_device):
            tables_to_search = [device_table_map.get(primary_device)]
            if primary_device != "CPU":
                tables_to_search.append(self.cpu_table)

            for table in tables_to_search:
                if not table:
                    continue
                for row in range(table.rowCount()):
                    model_item = table.item(row, 0)
                    infer_item = table.item(row, 2)
                    if model_item and infer_item:
                        model_path = model_item.text()
                        if model_path.endswith(partition_name):
                            try:
                                return float(infer_item.text())
                            except:
                                return None
            return None

        for model_prefix, device in self.assignment_results:
            npu_files, cpu_files = self.find_partition_files(root_folder, model_prefix, device)

            if device == "CPU":
                if cpu_files:
                    for f in cpu_files:
                        dur = get_inference_time(f, "CPU")
                        if dur is not None:
                            labels.append(f"{f} (CPU)")
                            durations.append(dur)
                else:
                    # 전체 모델이 CPU에만 할당되어 있는 경우 처리
                    model_file_name = f"{model_prefix}.onnx"
                    model_file_path = os.path.join(model_prefix, "model", model_file_name)
                    dur = get_inference_time(model_file_name, "CPU")
                    if dur is not None:
                        labels.append(f"{model_file_name} (CPU)")
                        durations.append(dur)
            else:
                for f in npu_files:
                    dur = get_inference_time(f, device)
                    if dur is not None:
                        labels.append(f"{f} ({device})")
                        durations.append(dur)
                for f in cpu_files:
                    dur = get_inference_time(f, "CPU")
                    if dur is not None:
                        labels.append(f"{f} (CPU)")
                        durations.append(dur)

        # 그래프 출력
        ax.barh(labels, durations, color="skyblue")
        ax.set_xlabel("Inference Time (ms)")
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=9)
        fig.tight_layout()
        return canvas

    def show_partition_assignment_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Partition Assignment Overview")

        splitter = QSplitter(Qt.Vertical)

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Model", "Device", "Partition"])

        all_rows = []
        root_folder = self.folder_input.text().strip()

        row_colors = [
            QColor(240, 248, 255),
            QColor(255, 250, 205),
            QColor(224, 255, 255),
            QColor(255, 228, 225),
            QColor(245, 245, 220),
            QColor(230, 230, 250),
        ]
        color_map = {}
        color_index = 0

        for model_prefix, device in self.assignment_results:
            if model_prefix not in color_map:
                color_map[model_prefix] = row_colors[color_index % len(row_colors)]
                color_index += 1

            model_color = QBrush(color_map[model_prefix])
            npu_files, cpu_files = self.find_partition_files(root_folder, model_prefix, device)

            model_written = False
            if device == "CPU":
                target_files = cpu_files
            else:
                target_files = npu_files

            if not target_files:
                all_rows.append((model_prefix, device, "(None)", model_color))
            else:
                for i, f in enumerate(sorted(target_files)):
                    model_col = model_prefix if not model_written else ""
                    all_rows.append((model_col, device if i == 0 else "", f, model_color))
                    model_written = True

            if cpu_files and device != "CPU":
                for i, f in enumerate(sorted(cpu_files)):
                    model_col = model_prefix if not model_written else ""
                    all_rows.append((model_col, "CPU" if i == 0 else "", f, model_color))
                    model_written = True

        table.setRowCount(len(all_rows))
        for i, (model, dev, part, brush) in enumerate(all_rows):
            model_item = QTableWidgetItem(model)
            model_item.setBackground(brush)
            if model:
                font = QFont()
                font.setBold(True)
                model_item.setFont(font)
            table.setItem(i, 0, model_item)

            dev_item = QTableWidgetItem(dev)
            dev_item.setBackground(brush)
            table.setItem(i, 1, dev_item)

            part_item = QTableWidgetItem(part)
            part_item.setBackground(brush)
            table.setItem(i, 2, part_item)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        splitter.addWidget(table)
        splitter.addWidget(self.create_inference_bar_chart())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(dialog)
        layout.addWidget(splitter)
        dialog.setLayout(layout)
        dialog.resize(720, 600)
        dialog.show()

    def update_frequencies_from_log(self):
        log_dir = "./logs"  # ← log 디렉토리 경로를 명시적으로 지정
        model_freqs = defaultdict(list)

        for filename in os.listdir(log_dir):
            if not filename.endswith("_log.json"):
                continue

            model_name = filename.replace("_log.json", "")
            log_path = os.path.join(log_dir, filename)

            try:
                with open(log_path, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            fps = entry.get("average_fps", None)
                            if fps is not None:
                                model_freqs[model_name].append(fps)
                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                print(f"[Error] {log_path} 읽기 실패: {e}")
                continue

        updated_count = 0
        for row in range(self.model_freq_table.rowCount()):
            model_item = self.model_freq_table.item(row, 0)
            freq_item = self.model_freq_table.item(row, 1)
            if model_item is None or freq_item is None:
                continue

            model_name = model_item.text()
            if model_name in model_freqs and model_freqs[model_name]:
                avg_freq = round(sum(model_freqs[model_name]) / len(model_freqs[model_name]))
                freq_item.setText(str(avg_freq))
                updated_count += 1

        print(f"[Info] Frequency 업데이트 완료: {updated_count}개 모델")


