#!/usr/bin/env python3
"""
Schedule Generator Main Script
This script launches the Schedule Generator GUI application.

Usage:
    python schedule_generator_main.py [--target-device TARGET_DEVICE_FILE]
"""

import sys
import argparse
from PyQt5.QtWidgets import QApplication
from schedule_generator_app import ONNXProfilerApp

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Schedule Generator Application')
    parser.add_argument('--target-device', type=str, help='Path to target device information file (e.g., target_device.yaml)')
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    app = QApplication(sys.argv)
    window = ONNXProfilerApp(target_device_file=args.target_device)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()