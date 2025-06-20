import sys
import os
import time
import numpy as np
import onnx
import onnxruntime as ort

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QListWidget, QTableWidget,
    QTableWidgetItem, QFileDialog, QSplitter, QSizePolicy
)
from PyQt6.QtCore import Qt


class ONNXProfiler(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ONNX 모델 성능 측정기")
        self.resize(900, 600)

        self.folder_input = QLineEdit(os.getcwd())
        self.folder_input.setPlaceholderText("ONNX 모델이 있는 폴더 경로를 입력하거나 폴더 선택 버튼을 누르세요")

        self.browse_button = QPushButton("폴더 선택..")
        self.browse_button.clicked.connect(self.browse_folder)

        self.profile_button = QPushButton("Profile")
        self.profile_button.clicked.connect(self.run_profiling)

        self.model_list = QListWidget()
        self.model_list.setSizePolicy(self.model_list.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)

        self.execution_log = QTableWidget()
        self.execution_log.setColumnCount(2)
        self.execution_log.setHorizontalHeaderLabels(["모델 이름", "처리 유닛"])
        self.execution_log.horizontalHeader().setStretchLastSection(True)
        self.execution_log.setSizePolicy(self.execution_log.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        self.execution_log.setRowCount(0)

        self.generate_button = QPushButton("모델 배치 코드 생성")
        # self.generate_button.clicked.connect(self.generate_code)

        # 왼쪽 레이아웃
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.folder_input)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.browse_button)
        button_layout.addWidget(self.profile_button)
        left_layout.addLayout(button_layout)

        left_layout.addWidget(QLabel("ONNX 모델 리스트"))
        left_layout.addWidget(self.model_list)

        left_layout.addWidget(QLabel("실행된 모델 이름"))
        left_layout.addWidget(self.execution_log)
        left_layout.addWidget(self.generate_button)

        # 오른쪽: CPU/NPU 결과 테이블
        self.cpu_result_table = QTableWidget()
        self.cpu_result_table.setColumnCount(4)
        self.cpu_result_table.setHorizontalHeaderLabels(["모델 이름", "입력 Shape", "로딩 시간 (ms)", "추론 시간 (ms)"])
        self.cpu_result_table.horizontalHeader().setStretchLastSection(True)

        self.npu_result_table = QTableWidget()
        self.npu_result_table.setColumnCount(4)
        self.npu_result_table.setHorizontalHeaderLabels(["모델 이름", "입력 Shape", "로딩 시간 (ms)", "추론 시간 (ms)"])
        self.npu_result_table.horizontalHeader().setStretchLastSection(True)

        cpu_label = QLabel("CPU 실행 결과")
        cpu_layout = QVBoxLayout()
        cpu_layout.addWidget(cpu_label)
        cpu_layout.addWidget(self.cpu_result_table)

        npu_label = QLabel("NPU 실행 결과 (임의 값)")
        npu_layout = QVBoxLayout()
        npu_layout.addWidget(npu_label)
        npu_layout.addWidget(self.npu_result_table)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        cpu_widget = QWidget()
        cpu_widget.setLayout(cpu_layout)
        npu_widget = QWidget()
        npu_widget.setLayout(npu_layout)
        right_splitter.addWidget(cpu_widget)
        right_splitter.addWidget(npu_widget)
        right_splitter.setSizes([300, 300])

        # 전체 레이아웃
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addWidget(right_splitter, 3)
        self.setLayout(main_layout)

        # 초기 onnx 파일 로딩
        self.load_onnx_files(os.getcwd())

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if folder:
            self.folder_input.setText(folder)
            self.load_onnx_files(folder)

    def load_onnx_files(self, folder):
        self.model_list.clear()
        onnx_files = [f for f in os.listdir(folder) if f.endswith(".onnx")]
        for f in onnx_files:
            self.model_list.addItem(f)

    def run_profiling(self):
        folder = self.folder_input.text().strip()
        if not os.path.isdir(folder):
            self.cpu_result_table.setRowCount(0)
            self.npu_result_table.setRowCount(0)
            self.execution_log.setRowCount(0)
            return

        onnx_files = [f for f in os.listdir(folder) if f.endswith(".onnx")]
        self.cpu_result_table.setRowCount(0)
        self.npu_result_table.setRowCount(0)
        self.execution_log.setRowCount(0)

        for row, model_file in enumerate(onnx_files):
            model_path = os.path.join(folder, model_file)
            row_pos = self.execution_log.rowCount()
            self.execution_log.insertRow(row_pos)

            try:
                load_ms, infer_ms, input_shape = self.profile_model(model_path)

                self.execution_log.setItem(row_pos, 0, QTableWidgetItem(model_file))
                self.execution_log.setItem(row_pos, 1, QTableWidgetItem("CPU"))

                self.cpu_result_table.insertRow(row)
                self.cpu_result_table.setItem(row, 0, QTableWidgetItem(model_file))
                self.cpu_result_table.setItem(row, 1, QTableWidgetItem(str(input_shape)))
                self.cpu_result_table.setItem(row, 2, QTableWidgetItem(f"{load_ms:.1f}"))
                self.cpu_result_table.setItem(row, 3, QTableWidgetItem(f"{infer_ms:.1f}"))

                self.npu_result_table.insertRow(row)
                self.npu_result_table.setItem(row, 0, QTableWidgetItem(model_file))
                self.npu_result_table.setItem(row, 1, QTableWidgetItem(str(input_shape)))
                self.npu_result_table.setItem(row, 2, QTableWidgetItem("42.0"))
                self.npu_result_table.setItem(row, 3, QTableWidgetItem("7.7"))

            except Exception as e:
                self.execution_log.setItem(row_pos, 0, QTableWidgetItem(model_file))
                self.execution_log.setItem(row_pos, 1, QTableWidgetItem("Error"))

                self.cpu_result_table.insertRow(row)
                self.cpu_result_table.setItem(row, 0, QTableWidgetItem(model_file))
                self.cpu_result_table.setItem(row, 1, QTableWidgetItem("Error"))
                self.cpu_result_table.setItem(row, 2, QTableWidgetItem("-"))
                self.cpu_result_table.setItem(row, 3, QTableWidgetItem(str(e)))

                self.npu_result_table.insertRow(row)
                self.npu_result_table.setItem(row, 0, QTableWidgetItem(model_file))
                self.npu_result_table.setItem(row, 1, QTableWidgetItem("Error"))
                self.npu_result_table.setItem(row, 2, QTableWidgetItem("-"))
                self.npu_result_table.setItem(row, 3, QTableWidgetItem("-"))

    def profile_model(self, model_path):
        start_load = time.time()
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        end_load = time.time()
        load_time_ms = (end_load - start_load) * 1000.0

        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_shape = [1 if s is None or s == 'None' else int(s) for s in input_shape]

        input_data = np.random.rand(*input_shape).astype(np.float32)

        start_infer = time.time()
        _ = session.run(None, {input_name: input_data})
        end_infer = time.time()
        infer_time_ms = (end_infer - start_infer) * 1000.0

        return load_time_ms, infer_time_ms, input_shape


if __name__ == "__main__":
    app = QApplication(sys.argv)
    profiler = ONNXProfiler()
    profiler.show()
    sys.exit(app.exec())
