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
import json
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from unified_viewer import UnifiedViewer, InfoWindow

def main():
    """Main function to execute all schedules sequentially and save throughput results."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Schedule Executor GUI application')
    parser.add_argument('--schedule', '-s', type=str, default='model_schedules.yaml',
                        help='Path to the model scheduling information file (default: model_schedules.yaml)')
    parser.add_argument('--duration', '-d', type=int, default=6,
                        help='Execution duration per schedule in seconds (default: 60)')
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

    # Create a persistent InfoWindow that will remain open across runs
    persistent_info = InfoWindow(parent=None)
    persistent_info.show()

    state = {'index': 0, 'viewer': None, 'info_window': persistent_info, 'duration': args.duration, 'running': False}

    def _write_best_header():
        """After all runs, rewrite result_throughput.json into the required object format.
        Output format:
        {
            "best deployment": "combination_X",
            "data": [ ... ]
        }
        """
        results_path = os.path.join(os.getcwd(), 'result_throughput.json')
        if not os.path.exists(results_path):
            print('[Main] No results file to annotate with best deployment.')
            return
        try:
            with open(results_path, 'r', encoding='utf-8') as rf:
                content = rf.read()
            # Try to parse JSON directly; ignore any legacy comment lines starting with '#'
            try:
                data_json = json.loads(content)
            except json.JSONDecodeError:
                # Remove comment lines and try again (backward compatibility)
                cleaned = '\n'.join(line for line in content.splitlines() if not line.strip().startswith('#'))
                data_json = json.loads(cleaned)

            # Determine if already wrapped or a plain list
            if isinstance(data_json, dict) and isinstance(data_json.get('data'), list):
                entries = data_json.get('data', [])
            elif isinstance(data_json, list):
                entries = data_json
            else:
                print('[Main] Results format not recognized; skipping annotation.')
                return

            if not entries:
                print('[Main] No entries found in results; skipping annotation.')
                return

            # Select item with max total throughput
            def _total(d):
                try:
                    return float(d.get('total', {}).get('total_throughput_fps', 0) or 0)
                except Exception:
                    return 0.0

            best = max(entries, key=_total)
            best_combo = best.get('combination') or 'unknown'

            final_obj = {
                "best deployment": best_combo,
                "data": entries
            }
            with open(results_path, 'w', encoding='utf-8') as wf:
                json.dump(final_obj, wf, indent=4, ensure_ascii=False)
            print(f"[Main] Wrote results in required format with best deployment: {best_combo}")
        except Exception as e:
            print(f"[Main] Warning: failed to write required results format: {e}")

    def run_next():
        # If we've executed all combinations, leave windows open and stop scheduling further actions
        if state['index'] >= len(combination_keys):
            print('[Main] All combinations executed. Leaving windows open.')
            # Add best deployment header at the end of all runs
            _write_best_header()
            # Re-enable start button to allow re-running from the beginning
            try:
                state['info_window'].start_button.setEnabled(True)
            except Exception:
                pass
            # Reset state to allow starting from the first schedule again
            state['running'] = False
            state['index'] = 0
            return

        # If previous viewer exists, ensure it is cleaned up
        if state['viewer'] is not None:
            try:
                # Ensure stop_execution has been called and viewer is hidden/deleted
                state['viewer'].stop_execution()
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

        combo = combination_keys[state['index']]
        print(f"[Main] Starting schedule: {combo}")
        viewer = UnifiedViewer(schedule_file=args.schedule, combination_name=combo, info_window=state['info_window'])
        state['viewer'] = viewer
        # Ensure the InfoWindow keeps the external controller as parent (UnifiedViewer may set itself)
        try:
            if 'controller' in state and state['controller'] is not None:
                state['info_window'].parent = state['controller']
        except Exception:
            pass
        # Ensure the info window displays the current schedule name and is visible on top
        try:
            state['info_window'].update_schedule_name(f"Current Schedule: {combo}")
            state['info_window'].show()
            state['info_window'].raise_()
            state['info_window'].activateWindow()
        except Exception:
            pass
        viewer.show()
        # Use specified duration per schedule (from state, which can be set by InfoWindow Start)
        duration = state.get('duration', args.duration)
        viewer.start_execution(duration)
        # Schedule moving to the next combination after duration + small buffer (ms)
        buffer_ms = 1000
        QTimer.singleShot((duration * 1000) + buffer_ms, lambda: _after_stop())

    def _after_stop():
        # If user stopped execution, do not advance further
        if not state.get('running'):
            return
        # Ensure throughput is saved and proceed to next
        try:
            if state['viewer'] is not None:
                state['viewer'].stop_execution()
        except Exception as e:
            print(f"[Main] Warning: stop_execution error: {e}")
        state['index'] += 1
        # Short delay to allow file writes flush
        QTimer.singleShot(300, run_next)

    # Wire InfoWindow buttons via a lightweight controller to start/stop execution on demand
    class _Controller:
        def __init__(self, state_ref):
            self._state = state_ref
        def start_execution(self, duration):
            # Ignore if already running
            if self._state.get('running'):
                print('[Main] Start requested but execution is already running.')
                return
            # Remove and recreate results file at the moment Start is pressed
            results_path = os.path.join(os.getcwd(), 'result_throughput.json')
            try:
                if os.path.exists(results_path):
                    os.remove(results_path)
                    print('[Main] Existing result_throughput.json removed on Start.')
                # Recreate as empty JSON array for consistent appending during this run
                with open(results_path, 'w', encoding='utf-8') as wf:
                    wf.write('[]')
                print('[Main] New empty result_throughput.json created.')
            except Exception as e:
                print(f"[Main] Warning: could not reset result_throughput.json on Start: {e}")
            # Set duration from InfoWindow and disable start button to prevent duplicates
            try:
                self._state['info_window'].start_button.setEnabled(False)
            except Exception:
                pass
            self._state['duration'] = max(1, int(duration))
            # Always start from the first schedule per requirement
            self._state['index'] = 0
            self._state['running'] = True
            print('[Main] Starting execution from the first schedule via Start button.')
            run_next()
        def stop_execution(self):
            # Stop current viewer and mark as not running; do not advance automatically
            viewer = self._state.get('viewer')
            if viewer is not None:
                try:
                    viewer.stop_execution()
                except Exception:
                    pass
                try:
                    viewer.hide()
                except Exception:
                    pass
                try:
                    viewer.deleteLater()
                except Exception:
                    pass
                self._state['viewer'] = None
            self._state['running'] = False
            try:
                self._state['info_window'].start_button.setEnabled(True)
            except Exception:
                pass
            print('[Main] Execution stopped by user.')

    controller = _Controller(state)
    state['controller'] = controller
    # Assign controller as the parent so InfoWindow's built-in handlers call our methods
    try:
        persistent_info.parent = controller
    except Exception:
        pass

    try:
        exit_code = app.exec_()
    except Exception as e:
        print(f"[Main] QApplication error: {e}")
        exit_code = 1

    print('[Main] QApplication loop exited.')
    os._exit(exit_code)

if __name__ == "__main__":
    main()