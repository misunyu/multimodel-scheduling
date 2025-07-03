# multimodel_main.py
import sys
from PyQt5.QtWidgets import QApplication
from multimodel_gui import UnifiedViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = UnifiedViewer()
    viewer.show()
    sys.exit(app.exec_())
