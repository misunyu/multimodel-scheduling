import sys
import os
from PyQt5.QtWidgets import QApplication
from model_profiler_gui import ONNXProfiler

def main():
    app = QApplication(sys.argv)
    window = ONNXProfiler()
    window.show()

    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())