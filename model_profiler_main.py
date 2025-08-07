#!/usr/bin/env python3
"""
Model Profiler Main Script
This script launches the Model Profiler GUI application.
"""

import sys
from PyQt5.QtWidgets import QApplication
from model_profiler_app import ONNXProfilerApp

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = ONNXProfilerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()