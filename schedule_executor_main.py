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
import os
import argparse
import yaml
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from unified_viewer import UnifiedViewer

def main():
    """Main function to execute all schedules sequentially and save throughput results."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Schedule Executor GUI application')
    parser.add_argument('--schedule', '-s', type=str, default='model_schedules.yaml',
                        help='Path to the model scheduling information file (default: model_schedules.yaml)')
    parser.add_argument('--duration', '-d', type=int, default=6,
                        help='Execution duration per schedule in seconds (default: 6)')
    args = parser.parse_args()

    # Remove existing result_throughput.json before execution if exists
    results_path = os.path.join(os.getcwd(), 'result_throughput.json')
    if os.path.exists(results_path):
        try:
            os.remove(results_path)
            print('[Main] Existing result_throughput.json removed.')
        except Exception as e:
            print(f"[Main] Warning: failed to remove existing result_throughput.json: {e}")

    # Load schedule combinations in order
    try:
        with open(args.schedule, 'r') as f:
            schedules = yaml.safe_load(f) or {}
        combination_keys = list(schedules.keys())
    except Exception as e:
        print(f"[Main] ERROR: Failed to read schedules from {args.schedule}: {e}")
        combination_keys = []

    if not combination_keys:
        print('[Main] No combinations found in schedule file. Exiting.')
        return os._exit(1)

    app = QApplication(sys.argv)

    state = {'index': 0, 'viewer': None}

    def run_next():
        # If previous viewer exists, ensure it is cleaned up
        if state['viewer'] is not None:
            try:
                # Ensure stop_execution has been called and viewer is hidden/deleted
                state['viewer'].stop_execution()
            except Exception:
                pass
            # Hide and delete the previous info window safely (do NOT close to avoid os._exit)
            try:
                if hasattr(state['viewer'], 'info_window') and state['viewer'].info_window is not None:
                    state['viewer'].info_window.hide()
                    state['viewer'].info_window.deleteLater()
            except Exception:
                pass
            try:
                state['viewer'].hide()
            except Exception:
                pass
            try:
                state['viewer'].deleteLater()
            except Exception:
                pass
            state['viewer'] = None

        if state['index'] >= len(combination_keys):
            print('[Main] All combinations executed. Quitting application.')
            app.quit()
            return

        combo = combination_keys[state['index']]
        print(f"[Main] Starting schedule: {combo}")
        viewer = UnifiedViewer(schedule_file=args.schedule, combination_name=combo)
        state['viewer'] = viewer
        # Ensure the info window displays the current schedule name and is visible on top
        try:
            viewer.info_window.update_schedule_name(f"Current Schedule: {combo}")
            viewer.info_window.show()
            viewer.info_window.raise_()
            viewer.info_window.activateWindow()
        except Exception:
            pass
        viewer.show()
        # Use specified duration per schedule
        viewer.start_execution(args.duration)
        # Schedule moving to the next combination after duration + small buffer (ms)
        buffer_ms = 1000
        QTimer.singleShot((args.duration * 1000) + buffer_ms, lambda: _after_stop())

    def _after_stop():
        # Ensure throughput is saved and proceed to next
        try:
            if state['viewer'] is not None:
                state['viewer'].stop_execution()
        except Exception as e:
            print(f"[Main] Warning: stop_execution error: {e}")
        state['index'] += 1
        # Short delay to allow file writes flush
        QTimer.singleShot(300, run_next)

    # Kick off the first run shortly after app starts
    QTimer.singleShot(0, run_next)

    try:
        exit_code = app.exec_()
    except Exception as e:
        print(f"[Main] QApplication error: {e}")
        exit_code = 1

    print('[Main] QApplication loop exited.')
    os._exit(exit_code)

if __name__ == "__main__":
    main()