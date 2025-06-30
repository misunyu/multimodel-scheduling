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
    QHeaderView, QVBoxLayout, QCheckBox, QTabWidget, QWidget,
    QLabel, QHBoxLayout
)
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import QLabel, QHBoxLayout
from NeublaDriver import NeublaDriver


CUSTOM_OP_PREFIXES = ["com.neubla"]

class ONNXProfiler(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("onnx_profiler_display_modify.ui", self)

        self.enable_npu2_checkbox = self.findChild(QCheckBox, "npu2_enable_checkbox")
        self.result_tabs = self.findChild(QTabWidget, "result_tab_widget")  # <-- 여기가 핵심!
        self.npu2_tab = self.findChild(QWidget, "npu2_tab")

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

        self.legend_label = QLabel()
        self.legend_label.setText(
            "<span style='background-color:#cce6ff;'>&nbsp;&nbsp;&nbsp;</span> CPU &nbsp;&nbsp;"
            "<span style='background-color:#ffffcc;'>&nbsp;&nbsp;&nbsp;</span> NPU1 &nbsp;&nbsp;"
            "<span style='background-color:#ffd699;'>&nbsp;&nbsp;&nbsp;</span> NPU2"
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
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

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
        self.generate_button.clicked.connect(self.highlight_deploy_results)

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

        self.total_table = self.findChild(QTableWidget, "total_table")
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
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

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

            for model in sorted(all_models):
                cpu_infer = valid_model_onnx.get(model, [0.0, 0.0])[1]

                load1 = npu1_load.get(model, 0.0)
                infer1_base = npu1_infer.get(model, 0.0)
                extra_cpu_infer = sum(cpu_infer_per_partition.get(model, []))  # ✅ 추가
                infer1 = infer1_base + extra_cpu_infer  # ✅ 합산

                load2 = npu2_load.get(model, 0.0)
                infer2_base = npu2_infer.get(model, 0.0)
                # extra_cpu_infer2 = sum(cpu_infer_per_partition.get(model, []))  # ✅ _p0, _p2 CPU 시간 더함
                infer2 = infer2_base + extra_cpu_infer

                row = self.total_table.rowCount()
                self.total_table.insertRow(row)
                self.total_table.setItem(row, 0, QTableWidgetItem(model))
                self.total_table.setItem(row, 1, QTableWidgetItem(f"{cpu_infer:.1f}"))
                self.total_table.setItem(row, 2, QTableWidgetItem(f"{load1:.1f}"))
                self.total_table.setItem(row, 3, QTableWidgetItem(f"{infer1:.1f}"))  # ✅ 수정
                self.total_table.setItem(row, 4, QTableWidgetItem(f"{load2:.1f}"))
                self.total_table.setItem(row, 5, QTableWidgetItem(f"{infer2:.1f}"))


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

        # yolov3로 시작하는 경우 실제 NPU 실행
        if os.path.basename(o_path).startswith("yolov3"):
            load_time_ms, infer_time_ms = self.process_yolo_npu(npu_num, o_path)
            return load_time_ms, infer_time_ms, []
        elif os.path.basename(o_path).startswith("resnet50"):
            load_time_ms, infer_time_ms = self.process_resnet50_npu(npu_num, o_path)
            return load_time_ms, infer_time_ms, []
        else:

            if basename.startswith("resnet50"):
                if label == "NPU1":
                    load_time_ms = 15.8
                    infer_time_ms = 38.6
                else:  # NPU2
                    load_time_ms = 77.1
                    infer_time_ms = 38.8
                return load_time_ms, infer_time_ms, []

                # yolov3_small 시뮬레이션 값
            elif basename.startswith("yolov3_small"):
                if label == "NPU1":
                    load_time_ms = 104.3
                    infer_time_ms = 60.9
                else:  # NPU2
                    load_time_ms = 430.6
                    infer_time_ms = 82.7
                return load_time_ms, infer_time_ms, []

                # yolov3_big 시뮬레이션 값
            elif basename.startswith("yolov3_big"):
                if label == "NPU1":
                    load_time_ms = 107.0
                    infer_time_ms = 87.4
                else:  # NPU2
                    load_time_ms = 467.5
                    infer_time_ms = 110.4
                return load_time_ms, infer_time_ms, []

                # 그 외의 경우 일반 시뮬레이션
            start_load = time.time()
            time.sleep(0.01)
            end_load = time.time()
            load_time_ms = (end_load - start_load) * 1000.0

            start_infer = time.time()
            time.sleep(0.003)
            end_infer = time.time()
            infer_time_ms = (end_infer - start_infer) * 1000.0


        return load_time_ms, infer_time_ms, []

    def process_yolo_npu(self, npu_num, o_path):
        try:
            driver = NeublaDriver()
            assert driver.Init(npu_num) == 0

            start_load = time.time()
            assert driver.LoadModel(o_path) == 0
            end_load = time.time()
            load_time_ms = (end_load - start_load) * 1000.0

            random_input = np.random.rand(3, 608, 608).astype(np.uint8)
            input_data = random_input.tobytes()

            start_infer = time.time()
            assert driver.SendInput(input_data, 3 * 608 * 608) == 0
            assert driver.Launch() == 0
            raw_outputs = driver.ReceiveOutputs()
            end_infer = time.time()
            infer_time_ms = (end_infer - start_infer) * 1000.0

            assert driver.Close() == 0

        except Exception as e:
            try:
                driver.Close()
            except:
                pass
            print(f"[Error] NPU{npu_num}: {e}")
            exit()

        return load_time_ms, infer_time_ms

    def process_resnet50_npu(self, npu_num, o_path):
        try:
            driver = NeublaDriver()
            assert driver.Init(npu_num) == 0

            start_load = time.time()
            assert driver.LoadModel(o_path) == 0
            end_load = time.time()
            load_time_ms = (end_load - start_load) * 1000.0

            random_input = np.random.rand(3, 224, 224).astype(np.uint8)
            input_data = random_input.tobytes()

            start_infer = time.time()
            assert driver.SendInput(input_data, 3 * 224 * 224) == 0
            assert driver.Launch() == 0
            raw_outputs = driver.ReceiveOutputs()
            end_infer = time.time()
            infer_time_ms = (end_infer - start_infer) * 1000.0

            assert driver.Close() == 0

        except Exception as e:
            try:
                driver.Close()
            except:
                pass
            print(f"[Error] NPU{npu_num}: {e}")
            exit()

        return load_time_ms, infer_time_ms

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
    def highlight_deploy_results(self):
        if not self.total_table:
            return

        models = []
        times = []  # [(cpu, npu1, npu2)]

        for row in range(self.total_table.rowCount()):
            try:
                model_item = self.total_table.item(row, 0)
                cpu_infer_item = self.total_table.item(row, 1)
                npu1_infer_item = self.total_table.item(row, 3)
                npu2_infer_item = self.total_table.item(row, 5)

                if not model_item or not cpu_infer_item or not npu1_infer_item:
                    continue

                model = model_item.text()
                cpu_infer = float(cpu_infer_item.text())
                npu1_infer = float(npu1_infer_item.text())

                # NPU2 체크 여부 확인
                if self.enable_npu2_checkbox and self.enable_npu2_checkbox.isChecked() and npu2_infer_item:
                    npu2_infer = float(npu2_infer_item.text())
                else:
                    npu2_infer = float('inf')  # 배치에서 제외

                models.append((row, model))
                times.append((cpu_infer, npu1_infer, npu2_infer))

            except Exception as e:
                print(f"[Warning] Skipping row {row}: {e}")

        load = {"CPU": 0.0, "NPU1": 0.0, "NPU2": 0.0}
        assignments = []

        for idx, (cpu_t, npu1_t, npu2_t) in enumerate(times):
            device_candidates = ["CPU", "NPU1"]
            if self.enable_npu2_checkbox and self.enable_npu2_checkbox.isChecked():
                device_candidates.append("NPU2")

            best_device = min(
                device_candidates,
                key=lambda d: load[d] + (cpu_t if d == "CPU" else npu1_t if d == "NPU1" else npu2_t)
            )
            assignments.append(best_device)
            load[best_device] += cpu_t if best_device == "CPU" else npu1_t if best_device == "NPU1" else npu2_t

        brush_map = {
            "CPU": QBrush(QColor(200, 230, 255)),
            "NPU1": QBrush(QColor(255, 255, 204)),
            "NPU2": QBrush(QColor(255, 214, 153)),
        }

        for i, (row_idx, _) in enumerate(models):
            brush = brush_map.get(assignments[i])
            if brush:
                for col in range(self.total_table.columnCount()):
                    item = self.total_table.item(row_idx, col)
                    if item:
                        item.setBackground(brush)

