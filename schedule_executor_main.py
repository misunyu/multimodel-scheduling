"""
Schedule Executor GUI application (modularized)

This module provides a cleaned-up and modular implementation of the legacy
schedule execution flow. The previous monolithic version embedded nested
functions and state dictionaries. Here, we encapsulate behavior inside a
ScheduleExecutor class and expose a small Controller that InfoWindow can
bind to. Functionality and CLI remain compatible.
"""

import sys
import os
import argparse
import yaml
import json
from typing import List
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from unified_viewer import UnifiedViewer, InfoWindow


class ScheduleExecutor:
    """Encapsulates state and behavior for running schedule combinations sequentially."""

    def __init__(self, schedule_file: str, duration: int, info_window: InfoWindow):
        self.schedule_file = schedule_file
        self.default_duration = max(1, int(duration))
        self.info_window = info_window

        self._viewer: UnifiedViewer = None
        self._index: int = 0
        self._running: bool = False
        self._combinations: List[str] = self._load_combinations(schedule_file)

        if not self._combinations:
            print('[Executor] No combinations found in schedule file. Exiting.')
            os._exit(1)

    # ------------------------------ Public API ------------------------------ #

    def start(self, duration: int = None):
        if self._running:
            print('[Executor] Start requested but execution is already running.')
            return
        self._running = True
        self._index = 0
        if duration is not None:
            self.default_duration = max(1, int(duration))
        self._reset_results_file()
        self._set_start_button_enabled(False)
        print('[Executor] Starting execution from the first schedule.')
        self._run_next()

    def stop(self):
        self._running = False
        self._cleanup_viewer()
        self._set_start_button_enabled(True)
        print('[Executor] Execution stopped by user.')

    # ----------------------------- Internal API ---------------------------- #

    def _load_combinations(self, schedule_file: str) -> List[str]:
        try:
            with open(schedule_file, 'r') as f:
                schedules = yaml.safe_load(f) or {}
            return list(schedules.keys())
        except Exception as e:
            print(f"[Executor] ERROR: Failed to read schedules from {schedule_file}: {e}")
            return []

    def _reset_results_file(self):
        results_path = os.path.join(os.getcwd(), 'result_throughput.json')
        try:
            if os.path.exists(results_path):
                os.remove(results_path)
                print('[Executor] Existing result_throughput.json removed on Start.')
            with open(results_path, 'w', encoding='utf-8') as wf:
                wf.write('[]')  # Empty JSON array for consistent appending
            print('[Executor] New empty result_throughput.json created.')
        except Exception as e:
            print(f"[Executor] Warning: could not reset result_throughput.json on Start: {e}")

    def _set_start_button_enabled(self, enabled: bool):
        try:
            self.info_window.start_button.setEnabled(enabled)
        except Exception:
            pass

    def _cleanup_viewer(self):
        viewer = self._viewer
        self._viewer = None
        if viewer is None:
            return
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

    def _run_next(self):
        if self._index >= len(self._combinations):
            print('[Executor] All combinations executed. Leaving windows open.')
            self._write_best_header()
            self._running = False
            self._index = 0
            self._set_start_button_enabled(True)
            return

        # Clean previous viewer if exists
        self._cleanup_viewer()

        combo = self._combinations[self._index]
        print(f"[Executor] Starting schedule: {combo}")

        # Create and show viewer
        self._viewer = UnifiedViewer(
            schedule_file=self.schedule_file,
            combination_name=combo,
            info_window=self.info_window,
        )

        # Keep info window up-to-date and on top
        try:
            self.info_window.update_schedule_name(f"Current Schedule: {combo}")
            self.info_window.show()
            self.info_window.raise_()
            self.info_window.activateWindow()
        except Exception:
            pass

        self._viewer.show()

        duration = self.default_duration
        self._viewer.start_execution(duration)

        # Schedule moving to the next combination after duration + small buffer (ms)
        buffer_ms = 1000
        QTimer.singleShot((duration * 1000) + buffer_ms, self._after_stop)

    def _after_stop(self):
        if not self._running:
            return
        try:
            if self._viewer is not None:
                self._viewer.stop_execution()
        except Exception as e:
            print(f"[Executor] Warning: stop_execution error: {e}")
        self._index += 1
        QTimer.singleShot(300, self._run_next)  # short delay to flush file writes

    def _write_best_header(self):
        """Rewrite result_throughput.json into required object format with best deployment."""
        results_path = os.path.join(os.getcwd(), 'result_throughput.json')
        if not os.path.exists(results_path):
            print('[Executor] No results file to annotate with best deployment.')
            return
        try:
            with open(results_path, 'r', encoding='utf-8') as rf:
                content = rf.read()
            try:
                data_json = json.loads(content)
            except json.JSONDecodeError:
                cleaned = '\n'.join(line for line in content.splitlines() if not line.strip().startswith('#'))
                data_json = json.loads(cleaned)

            if isinstance(data_json, dict) and isinstance(data_json.get('data'), list):
                entries = data_json.get('data', [])
            elif isinstance(data_json, list):
                entries = data_json
            else:
                print('[Executor] Results format not recognized; skipping annotation.')
                return

            if not entries:
                print('[Executor] No entries found in results; skipping annotation.')
                return

            def _total(d):
                try:
                    return float(d.get('total', {}).get('total_throughput_fps', 0) or 0)
                except Exception:
                    return 0.0

            best = max(entries, key=_total)
            best_combo = best.get('combination') or 'unknown'

            final_obj = {"best deployment": best_combo, "data": entries}
            with open(results_path, 'w', encoding='utf-8') as wf:
                json.dump(final_obj, wf, indent=4, ensure_ascii=False)
            print(f"[Executor] Wrote results with best deployment: {best_combo}")
        except Exception as e:
            print(f"[Executor] Warning: failed to write required results format: {e}")


class Controller:
    """Minimal controller that InfoWindow can bind its Start/Stop buttons to."""

    def __init__(self, executor: ScheduleExecutor):
        self._executor = executor

    def start_execution(self, duration):
        self._executor.start(duration)

    def stop_execution(self):
        self._executor.stop()


def main():
    """Entry point: parse args, create app/windows, and run event loop."""
    parser = argparse.ArgumentParser(description='Schedule Executor GUI application')
    parser.add_argument('--schedule', '-s', type=str, default='model_schedules.yaml',
                        help='Path to the model scheduling information file (default: model_schedules.yaml)')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Execution duration per schedule in seconds (default: 60)')
    args = parser.parse_args()

    # Legacy pre-clean of results file on launch (kept for compatibility)
    results_path = os.path.join(os.getcwd(), 'result_throughput.json')
    if os.path.exists(results_path):
        try:
            os.remove(results_path)
            print('[Main] Existing result_throughput.json removed.')
        except Exception as e:
            print(f"[Main] Warning: failed to remove existing result_throughput.json: {e}")

    app = QApplication(sys.argv)

    info = InfoWindow(parent=None)
    info.show()

    executor = ScheduleExecutor(schedule_file=args.schedule, duration=args.duration, info_window=info)
    controller = Controller(executor)

    # Assign controller as the parent so InfoWindow's built-in handlers call our methods
    try:
        info.parent = controller
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