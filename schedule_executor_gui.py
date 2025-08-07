"""
Schedule Executor GUI application - Main file (legacy version)

This file is kept for backward compatibility.
For the modularized version, see:
- main.py: Entry point
- unified_viewer.py: Main viewer class
- view_handlers.py: View handling components
- model_processors.py: Model processing functions
- image_processing.py: Image processing functions
- utils.py: Utility functions
"""

import sys
import argparse
from PyQt5.QtWidgets import QApplication
from unified_viewer import UnifiedViewer

def main():
    """Main function to start the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Schedule Executor GUI application')
    parser.add_argument('--schedule', '-s', type=str, default='model_schedules.yaml',
                        help='Path to the model scheduling information file (default: model_schedules.yaml)')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    viewer = UnifiedViewer(schedule_file=args.schedule)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()