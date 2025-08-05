# multimodel_main.py
import sys
import os
from PyQt5.QtWidgets import QApplication
from multimodel_gui import UnifiedViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = UnifiedViewer()
    viewer.show()

    try:
        exit_code = app.exec_()
    except Exception as e:
        print(f"[Main] QApplication error: {e}")
        exit_code = 1

    print("[Main] QApplication loop exited.")
    os._exit(exit_code)
