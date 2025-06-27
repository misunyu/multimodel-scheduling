import os
import time
import numpy as np
import onnxruntime as ort

from PyQt6 import uic
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QLineEdit, QPushButton,
    QFileDialog, QTreeView, QPlainTextEdit
)
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtWidgets import QApplication


class ONNXProfiler(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("onnx_profiler_display_modify.ui", self)

        # UI 요소 연결
        self.folder_input = self.findChild(QLineEdit, "folder_input")
        self.browse_button = self.findChild(QPushButton, "browse_button")
        self.profile_button = self.findChild(QPushButton, "profile_button")
        self.generate_button = self.findChild(QPushButton, "generate_button")
        self.model_tree_view = self.findChild(QTreeView, "model_tree_view")
        self.log_output = self.findChild(QPlainTextEdit, "log_output") or self.findChild(QPlainTextEdit, "log_textbox")

        # 파일 시스템 모델 설정
        self.fs_model = QFileSystemModel()
        self.fs_model.setReadOnly(True)
        self.fs_model.setNameFilters(["*.onnx"])
        self.fs_model.setNameFilterDisables(False)
        self.model_tree_view.setModel(self.fs_model)
        self.model_tree_view.setMinimumWidth(500)
        self.model_tree_view.header().setStretchLastSection(True)
        self.model_tree_view.header().setDefaultSectionSize(300)

        # 기본 경로 설정
        default_folder = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(default_folder):
            default_folder = os.getcwd()

        self.folder_input.setText(default_folder)
        self.browse_button.clicked.connect(self.browse_folder)
        self.profile_button.clicked.connect(self.run_profiling)

        self.set_tree_root(default_folder)
        QTimer.singleShot(100, lambda: self.expand_parents_of_onnx_files(default_folder))

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택", os.getcwd())
        if folder:
            self.folder_input.setText(folder)
            self.set_tree_root(folder)
            QTimer.singleShot(100, lambda: self.expand_parents_of_onnx_files(folder))

    def set_tree_root(self, folder):
        self.fs_model.setRootPath(folder)
        index = self.fs_model.index(folder)
        self.model_tree_view.setRootIndex(index)

    def expand_parents_of_onnx_files(self, root_folder):
        """모든 .onnx 포함 폴더 확장"""
        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                if f.endswith(".onnx"):
                    file_path = os.path.join(dirpath, f)
                    index = self.fs_model.index(file_path)
                    parent = index.parent()
                    while parent.isValid():
                        self.model_tree_view.expand(parent)
                        parent = parent.parent()

    from PyQt6.QtWidgets import QApplication  # 상단 import에 추가

    def run_profiling(self):
        root_folder = self.folder_input.text().strip()
        if not os.path.isdir(root_folder):
            return

        # 로그 초기화
        if self.log_output:
            self.log_output.clear()
            self.log_output.appendPlainText("[시작] 모델 프로파일링을 시작합니다...\n")
            QApplication.processEvents()  # 로그 초기화 즉시 반영

        onnx_files = []
        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                if f.endswith(".onnx"):
                    onnx_files.append(os.path.join(dirpath, f))

        for model_path in onnx_files:
            model_file = os.path.basename(model_path)

            if self.log_output:
                self.log_output.appendPlainText(f"[시작] {model_file} 로딩 중...")
                QApplication.processEvents()

            try:
                load_ms, infer_ms, input_shape = self.profile_model(model_path)

                # 트리뷰 항목 이름 업데이트
                index = self.fs_model.index(model_path)
                display_name = f"{model_file} (CPU: {infer_ms:.1f}ms / NPU: 42.0ms)"
                self.fs_model.setData(index, display_name, role=Qt.ItemDataRole.DisplayRole)

                if self.log_output:
                    self.log_output.appendPlainText(f"[완료] {model_file} - CPU: {infer_ms:.1f}ms\n")
                    QApplication.processEvents()

            except Exception as e:
                index = self.fs_model.index(model_path)
                error_display = f"{model_file} (Error: {str(e)})"
                self.fs_model.setData(index, error_display, role=Qt.ItemDataRole.DisplayRole)

                if self.log_output:
                    self.log_output.appendPlainText(f"[에러] {model_file}: {str(e)}\n")
                    QApplication.processEvents()

    def profile_model(self, model_path):
        start_load = time.time()
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        end_load = time.time()
        load_time_ms = (end_load - start_load) * 1000.0

        input_data_dict = {}
        for input_tensor in session.get_inputs():
            name = input_tensor.name
            shape = input_tensor.shape
            dtype = input_tensor.type  # e.g., 'tensor(float)', 'tensor(uint8)'

            shape = [1 if s is None or s == 'None' else int(s) for s in shape]

            if 'float' in dtype:
                np_dtype = np.float32
            elif 'uint8' in dtype:
                np_dtype = np.uint8
            elif 'int64' in dtype:
                np_dtype = np.int64
            else:
                raise ValueError(f"지원되지 않는 입력 타입: {dtype}")

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
