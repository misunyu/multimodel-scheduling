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
from PyQt5.QtCore import QTimer, Qt
from unified_viewer import UnifiedViewer, InfoWindow


class ScheduleExecutor:
    """Encapsulates state and behavior for running schedule combinations sequentially."""

    def __init__(self, schedule_file: str, duration: int, info_window: InfoWindow, selected_combo: str = None):
        self.schedule_file = schedule_file
        self.default_duration = max(1, int(duration))
        self.info_window = info_window
        self._selected_combo = selected_combo

        self._viewer: UnifiedViewer = None
        self._index: int = 0
        self._running: bool = False
        self._combinations: List[str] = self._load_combinations(schedule_file)

        # If a specific combination is requested, filter list to that single name
        if self._selected_combo:
            if self._selected_combo in self._combinations:
                self._combinations = [self._selected_combo]
            else:
                print(f"[Executor] ERROR: requested combination '{self._selected_combo}' not found in {os.path.basename(schedule_file)}")
                os._exit(2)

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
        # Prepare a unique results file path for this run
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._results_path = os.path.join(results_dir, f'performance_{ts}.json')
        self._reset_results_file()
        self._set_start_button_enabled(False)
        print(f"[Executor] Starting execution from the first schedule. Results will be saved to {self._results_path}")
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
        # Ensure results directory exists and initialize the run file with an empty array
        try:
            base_dir = os.path.dirname(getattr(self, '_results_path', '')) or os.path.join(os.getcwd(), 'results')
            os.makedirs(base_dir, exist_ok=True)
            with open(self._results_path, 'w', encoding='utf-8') as wf:
                wf.write('[]')  # Start with empty JSON array for appending
            print(f"[Executor] Initialized results file: {self._results_path}")
        except Exception as e:
            print(f"[Executor] Warning: could not initialize results file on Start: {e}")

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
            # If a specific combination was requested (executor-only mode), loop indefinitely until app exit
            if getattr(self, '_selected_combo', None):
                self._index = 0
                QTimer.singleShot(300, self._run_next)
                return
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
        # Mark viewer as executor-only if running a specific selected combo
        try:
            self._viewer.executor_only = bool(getattr(self, '_selected_combo', None))
        except Exception:
            pass
        # Pass shared results path to the viewer so all combinations append to the same run file
        try:
            self._viewer.results_path = self._results_path
        except Exception:
            pass

        # Keep info window up-to-date; in schedule_name mode keep it behind
        try:
            self.info_window.update_schedule_name(f"Current Schedule: {combo}")
            self.info_window.show()
            if getattr(self, '_selected_combo', None):
                try:
                    self.info_window.lower()
                except Exception:
                    pass
            else:
                try:
                    self.info_window.raise_()
                    self.info_window.activateWindow()
                except Exception:
                    pass
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
        """Rewrite the run results file into required object format with best deployment."""
        results_path = getattr(self, '_results_path', None)
        if not results_path or not os.path.exists(results_path):
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

            # ScheduleExecutor._write_best_header() 내부
            def _metrics(d):
                # 1) Total FPS
                try:
                    total_fps = float(d.get('total', {}).get('total_throughput_fps', 0) or 0)
                except Exception:
                    total_fps = 0.0

                # 2) 드롭 개수 합산
                drop = 0
                try:
                    models = d.get('models', {}) or {}
                    for mv in models.values():
                        drop += int(mv.get('dropped_frames_due_to_full_queue', 0) or 0)
                        # 다른 드롭 원인도 있으면 같이 더합니다(옵션)
                        drop += int(mv.get('dropped_frames_due_to_deadline', 0) or 0)
                        drop += int(mv.get('dropped_frames_cancelled', 0) or 0)
                except Exception:
                    drop = 0

                # 3) window_sec으로 나눠 drops/s로 변환
                window = float(d.get('window_sec', 1.0) or 1.0)
                drop_rate = drop / window

                # (선택) 투명성 위해 필드 추가
                d.setdefault('derived', {})['drop_rate_fps'] = round(drop_rate, 4)
                d['derived']['drop_count'] = int(drop)
                d['derived']['window_sec'] = window

                return total_fps, drop_rate

            # 점수 계산부
            for ent in entries:
                total_fps, drop_rate = _metrics(ent)
                score = total_fps - 0.2 * drop_rate
                ent['score'] = round(score, 4)

            # Determine best by highest score
            def _score(d):
                try:
                    return float(d.get('score', 0) or 0)
                except Exception:
                    return 0.0

            best = max(entries, key=_score)
            best_combo = best.get('combination') or 'unknown'

            final_obj = {"best deployment": best_combo, "schedule file": os.path.basename(self.schedule_file), "data": entries}
            with open(results_path, 'w', encoding='utf-8') as wf:
                json.dump(final_obj, wf, indent=4, ensure_ascii=False)
            print(f"[Executor] Wrote results with best deployment: {best_combo} (schedule: {os.path.basename(self.schedule_file)})")
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
    parser.add_argument('--schedule_name', '--schedule-name', type=str, default=None,
                        help='When set, run only the specified combination name from the schedule file in executor-only mode (no controller).')
    parser.add_argument('--auto_start_all', action='store_true',
                        help='Automatically start running all combinations and quit the app when done (no Start button needed).')
    args = parser.parse_args()

    # Resolve schedule path: if given path doesn't exist, try tests/<basename>
    schedule_path = args.schedule
    try:
        if not os.path.isabs(schedule_path) and not os.path.exists(schedule_path):
            tests_candidate = os.path.join(os.path.dirname(__file__), 'tests', os.path.basename(schedule_path))
            if os.path.exists(tests_candidate):
                schedule_path = tests_candidate
    except Exception:
        pass

    # No legacy pre-clean: results are now saved per-run under results/performance_*.json

    app = QApplication(sys.argv)

    # Create the InfoWindow instance
    info = InfoWindow(parent=None)

    # If schedule_name is provided, run executor-only mode and send InfoWindow to back
    if args.schedule_name:
        try:
            # Clear always-on-top and ensure the window is behind
            info.setWindowFlag(Qt.WindowStaysOnTopHint, False)
            info.show()
            try:
                info.lower()
            except Exception:
                pass
        except Exception:
            pass
        executor = ScheduleExecutor(schedule_file=schedule_path, duration=args.duration, info_window=info, selected_combo=args.schedule_name)
        # Disable Start button since we auto-run and no controller
        try:
            info.start_button.setEnabled(False)
            info.stop_button.setEnabled(True)
        except Exception:
            pass
        # Start immediately and keep cycling the same combination until user closes the app or presses Stop
        executor.start(args.duration)
        try:
            exit_code = app.exec_()
        except Exception as e:
            print(f"[Main] QApplication error: {e}")
            exit_code = 1
        print('[Main] QApplication loop exited.')
        os._exit(exit_code)

    # Otherwise, show InfoWindow and use full GUI mode
    info.show()

    # Default GUI mode with controller
    executor = ScheduleExecutor(schedule_file=schedule_path, duration=args.duration, info_window=info)
    controller = Controller(executor)

    # Assign controller as the parent so InfoWindow's built-in handlers call our methods
    try:
        info.parent = controller
    except Exception:
        pass

    # If auto_start_all requested, start immediately and quit when all combinations finish
    if args.auto_start_all:
        try:
            info.start_button.setEnabled(False)
            info.stop_button.setEnabled(True)
        except Exception:
            pass

        # Hook to quit the app once all combos are done
        def _on_all_done_quit():
            try:
                # When not running and start button is enabled again, we consider it done
                if not executor._running and executor._index == 0:
                    print('[Main] Auto mode: all combinations finished. Quitting application...')
                    QTimer.singleShot(50, app.quit)
                    return
            except Exception:
                pass
            # Re-check soon until done
            QTimer.singleShot(200, _on_all_done_quit)

        # Start now with requested duration
        controller.start_execution(args.duration)
        # Begin monitoring for completion
        QTimer.singleShot(200, _on_all_done_quit)

    try:
        exit_code = app.exec_()
    except Exception as e:
        print(f"[Main] QApplication error: {e}")
        exit_code = 1

    print('[Main] QApplication loop exited.')
    os._exit(exit_code)


if __name__ == "__main__":
    main()