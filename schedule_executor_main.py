"""
Schedule Executor GUI application

Runs schedule combinations one after another in the Schedule Executor Display
(UnifiedViewer). The display is the only window: there is no separate control
window, and execution starts as soon as the process does.

When launched with --control_stdin the process also takes one-line JSON commands
on stdin and reports events on stdout, which is how the main window switches the
display mode while a run is going:

    stdin   {"cmd": "set_show_all", "value": true}
    stdout  @@CTRL {"evt": "show_all", "value": true, ...}

stdin/stdout rather than a socket or a flag file: the main window already owns
both ends of this process's pipes (it launches it under QProcess and already reads
its stdout), so the channel needs no port, no path, no polling and no cleanup, and
a command is applied on the display's next event-loop turn.
"""

import sys
import os
import signal
import argparse
import yaml
import json
import threading
from typing import List
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from unified_viewer import UnifiedViewer

CTRL_PREFIX = "@@CTRL "


def emit_event(evt: str, **fields):
    """Report an event to whoever launched us (one JSON line on stdout)."""
    try:
        print(CTRL_PREFIX + json.dumps({"evt": evt, **fields}), flush=True)
    except Exception:
        pass


class _PipeSafeStream:
    """stdout/stderr that survive the reader going away.

    Our output is a pipe to the launcher. If the launcher dies, the next print raises
    BrokenPipeError -- and the first print in the shutdown path is exactly where that
    used to land, aborting the process before it had stopped its workers and leaving
    them holding the NPU. Losing the log must never cost the cleanup.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        try:
            return self._stream.write(text)
        except (BrokenPipeError, OSError, ValueError):
            return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


class ControlReader(QObject):
    """Reads command lines from stdin on a thread and hands them to the GUI thread.

    The signals are emitted from the reader thread and delivered queued, so the slots
    run inside the Qt event loop -- a command never touches a widget off-thread.
    """

    line_received = pyqtSignal(str)
    closed = pyqtSignal()       # stdin hit EOF: whoever launched us is gone

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            while True:
                line = sys.stdin.readline()
                if not line:        # EOF: the launcher went away
                    self.closed.emit()
                    return
                line = line.strip()
                if line:
                    self.line_received.emit(line)
        except Exception as e:
            print(f"[Control] stdin reader stopped: {e}", flush=True)


class ScheduleExecutor:
    """Encapsulates state and behavior for running schedule combinations sequentially."""

    def __init__(self, schedule_file: str, duration: int, selected_combo: str = None,
                 show_all: bool = False, geometry=None):
        self.schedule_file = schedule_file
        self.default_duration = max(1, int(duration))
        self._selected_combo = selected_combo
        # Display state that outlives any one viewer: each combination (and each cycle
        # of a looping one) gets a fresh UnifiedViewer, which must come up in the same
        # mode, at the same place and size, as the one it replaces.
        self.show_all = bool(show_all)
        self._geometry = geometry          # (x, y, w, h) or None

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
        print(f"[Executor] Starting execution from the first schedule. Results will be saved to {self._results_path}")
        self._run_next()

    def handle_control(self, line: str):
        """Apply one command from the launcher. A bad command is logged, never fatal."""
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            viewer = self._viewer
            if cmd == "set_show_all":
                self.show_all = bool(msg.get("value"))
                if viewer is not None:
                    viewer.set_show_all(self.show_all, source="main", t_sent=msg.get("t_sent"))
                else:
                    emit_event("show_all", value=self.show_all, source="main")
            elif cmd == "click_show_all_toggle":
                # Test hook: presses the display's own toggle, i.e. the user's path.
                if viewer is not None:
                    viewer.show_all_toggle.click()
            elif cmd == "click_stop_all":
                # Test hook: presses the display's Stop All button.
                if viewer is not None and viewer.stop_all_button is not None:
                    viewer.stop_all_button.click()
            elif cmd == "grab":
                # Test hook: save what the display window is showing right now.
                ok = bool(viewer is not None and viewer.grab().save(str(msg.get("path"))))
                emit_event("grabbed", path=msg.get("path"), ok=ok)
            else:
                print(f"[Control] Unknown command ignored: {line}", flush=True)
        except Exception as e:
            print(f"[Control] Command failed ({type(e).__name__}: {e}): {line}", flush=True)

    def on_launcher_gone(self):
        """The control channel closed: the main window that launched us no longer exists.

        Nothing can stop or replace this run from there any more, and a main window
        started afresh would not know it is here -- it would load a second set on top
        of this one. So leave the way a close does: stop the workers, dispose the
        models, then exit.
        """
        print("[Control] Launcher went away (control channel closed); shutting down cleanly.",
              flush=True)
        self._running = False
        viewer = self._viewer
        if viewer is not None:
            viewer.shutdown_all()
        else:
            QApplication.instance().quit()

    def _on_viewer_event(self, evt: str, **fields):
        """Events from the display: remember what must survive it, and pass them on."""
        if evt == "show_all":
            self.show_all = bool(fields.get("value"))
        elif evt == "geometry":
            self._geometry = (fields.get("x"), fields.get("y"), fields.get("w"), fields.get("h"))
        elif evt == "stopped":
            # Stop All: stay stopped. Without this the cycle timer armed at start would
            # replace the frozen display with a fresh run a little later.
            self._running = False
            print('[Executor] Execution stopped by user.')
        emit_event(evt, **fields)

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
            return

        # Clean previous viewer if exists
        self._cleanup_viewer()

        combo = self._combinations[self._index]
        print(f"[Executor] Starting schedule: {combo}")

        # Create and show viewer
        self._viewer = UnifiedViewer(
            schedule_file=self.schedule_file,
            combination_name=combo,
            show_all=self.show_all,
        )
        self._viewer.notify = self._on_viewer_event
        if self._geometry and all(v is not None for v in self._geometry):
            x, y, w, h = (int(v) for v in self._geometry)
            self._viewer.resize(w, h)
            self._viewer.move(x, y)
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

        self._viewer.show()
        emit_event("show_all", value=self._viewer.show_all, source="start")

        # Run for duration + warmup; the viewer starts measuring after the warmup.
        from unified_viewer import WARMUP_SECONDS
        measured_duration = self.default_duration
        warmup_seconds = WARMUP_SECONDS
        run_duration = int(measured_duration + warmup_seconds)
        self._viewer.start_execution(run_duration)

        # Schedule moving to the next combination after run_duration + small buffer (ms)
        buffer_ms = 1000
        QTimer.singleShot((run_duration * 1000) + buffer_ms, self._after_stop)

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

                # 2) Deadline-miss rate (0..1). Prefer the window-level value; else
                #    aggregate per-model misses / offered.
                total_obj = d.get('total', {}) or {}
                miss_rate = total_obj.get('deadline_miss_rate', None)
                miss = offered = 0
                try:
                    models = d.get('models', {}) or {}
                    for mv in models.values():
                        miss += int(mv.get('deadline_miss_count', 0) or 0)
                        offered += int(mv.get('frames_total', 0) or 0)
                except Exception:
                    pass
                if miss_rate is None:
                    miss_rate = (miss / offered) if offered > 0 else 0.0

                window = float(d.get('window_sec', 1.0) or 1.0)
                d.setdefault('derived', {})['deadline_miss_rate'] = round(float(miss_rate), 4)
                d['derived']['deadline_miss_count'] = int(miss)
                d['derived']['frames_total'] = int(offered)
                d['derived']['window_sec'] = window

                return total_fps, float(miss_rate)

            # 점수 계산부: FPS - PENALTY * deadline_miss_rate (miss_rate in [0,1])
            PENALTY = 100.0
            for ent in entries:
                total_fps, miss_rate = _metrics(ent)
                score = total_fps - PENALTY * miss_rate
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


def main():
    """Entry point: parse args, create app/windows, and run event loop."""
    # Use 'spawn' so per-view worker processes initialize CUDA / Mobilint NPU
    # cleanly (fork after Qt/driver init deadlocks and leaves NPU workers idle).
    import multiprocessing as _mp
    try:
        _mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='Schedule Executor GUI application')
    parser.add_argument('--schedule', '-s', type=str, default='model_schedules.yaml',
                        help='Path to the model scheduling information file (default: model_schedules.yaml)')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Execution duration per schedule in seconds (default: 60)')
    parser.add_argument('--schedule_name', '--schedule-name', type=str, default=None,
                        help='When set, run only the specified combination name from the schedule file in executor-only mode (no controller).')
    parser.add_argument('--auto_start_all', action='store_true',
                        help='Quit the app once every combination has run (without it the display stays open).')
    parser.add_argument('--show-all', '--show_all', dest='show_all', action='store_true',
                        help='Start the display in exhibition mode (render every running model). '
                             'Later changes arrive on stdin; see --control_stdin.')
    parser.add_argument('--control_stdin', action='store_true',
                        help='Accept one-line JSON commands on stdin, e.g. {"cmd":"set_show_all","value":true}.')
    # Not "--geometry": Qt claims that option for itself (X11 WxH+X+Y syntax) and would
    # apply its own reading of the value to the window as well.
    parser.add_argument('--window_rect', type=str, default=None,
                        help='Initial display window position and size as x,y,w,h.')
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

    sys.stdout, sys.stderr = _PipeSafeStream(sys.stdout), _PipeSafeStream(sys.stderr)

    app = QApplication(sys.argv)

    geometry = None
    if args.window_rect:
        try:
            geometry = tuple(int(v) for v in args.window_rect.split(','))
            if len(geometry) != 4:
                raise ValueError("expected four integers")
        except Exception as e:
            print(f"[Main] Ignoring --window_rect '{args.window_rect}': {e}")
            geometry = None

    # SIGTERM is how the parent GUI asks us to stop (QProcess.terminate). Python's
    # default handler would kill us outright, leaving the view workers holding the
    # NPU, and the next run would fail to allocate. Route it through the viewer's
    # normal cleanup instead, so the models get disposed.
    def _graceful_signal(signum, _frame):
        print(f"[Main] Signal {signum} received; shutting down cleanly.", flush=True)
        viewer = getattr(_signal_target.get('executor'), '_viewer', None)
        if viewer is not None:
            try:
                # shutdown_all = cleanup (dispose models) -> quit() -> forced backstop.
                # Going through it rather than quit()-ing ourselves matters: Qt does not
                # always leave exec_() when torch/Mobilint leave non-daemon threads
                # behind, and a process that lingers gets SIGKILLed by the parent.
                viewer.shutdown_all()
                return
            except Exception as e:
                print(f"[Main] Cleanup during signal failed (exiting anyway): {e}", flush=True)
        try:
            QApplication.instance().quit()
        except Exception:
            pass
        QTimer.singleShot(3000, lambda: os._exit(0))

    _signal_target = {}
    signal.signal(signal.SIGTERM, _graceful_signal)
    signal.signal(signal.SIGINT, _graceful_signal)

    # Python only runs signal handlers between bytecodes, and Qt's event loop sits in
    # C the whole time -- so without something that periodically hands control back to
    # the interpreter, the handler above never runs and SIGTERM is effectively ignored
    # until the parent gives up and SIGKILLs us (which orphans the view workers with
    # the models still on the NPU). This idle timer is that something.
    _sig_pump = QTimer()
    _sig_pump.start(200)
    _sig_pump.timeout.connect(lambda: None)

    executor = ScheduleExecutor(schedule_file=schedule_path, duration=args.duration,
                                selected_combo=args.schedule_name,
                                show_all=args.show_all, geometry=geometry)
    _signal_target['executor'] = executor

    # Runtime control channel (the main window's "Show all models" checkbox).
    if args.control_stdin:
        control = ControlReader()
        control.line_received.connect(executor.handle_control)
        control.closed.connect(executor.on_launcher_gone)
        control.start()

    # No Start button any more: the run begins with the process. With --schedule_name
    # the one combination cycles until the display is closed; otherwise every
    # combination runs once and, with --auto_start_all, the app then quits.
    executor.start(args.duration)

    if args.auto_start_all and not args.schedule_name:
        def _on_all_done_quit():
            try:
                if not executor._running and executor._index == 0:
                    print('[Main] Auto mode: all combinations finished. Quitting application...')
                    QTimer.singleShot(50, app.quit)
                    return
            except Exception:
                pass
            QTimer.singleShot(200, _on_all_done_quit)

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
