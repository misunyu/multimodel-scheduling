"""
Main entry point for the multimodel scheduling application.
"""
import sys
from PyQt5.QtWidgets import QApplication
from unified_viewer import UnifiedViewer

def main():
    """Main function to start the application."""
    app = QApplication(sys.argv)
    viewer = UnifiedViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()