import sys
import os
from PyQt6.QtWidgets import QApplication
from model_profiler_gui import ONNXProfiler  # 파일명이 onnx_profiler_gui.py였다면 이름만 변경하세요

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # .ui 파일이 실행 경로에 있는지 확인 (필요시 디버깅용)
    ui_path = os.path.join(os.path.dirname(__file__), "onnx_profiler_display_modify.ui")
    if not os.path.exists(ui_path):
        raise FileNotFoundError(f"No .ui file: {ui_path}")

    window = ONNXProfiler()
    window.show()

    sys.exit(app.exec())
