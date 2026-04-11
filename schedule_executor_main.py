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
import signal
import time
from typing import List
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from unified_viewer import UnifiedViewer, InfoWindow


class ScheduleExecutor:
    """Encapsulates state and behavior for running schedule combinations sequentially."""

    def __init__(self, schedule_file: str, duration: int, info_window: InfoWindow,
                 selected_combo: str = None, adaptive_mode: int = 0, metrics_csv: str = None,
                 combo_durations: dict = None):
        self.schedule_file = schedule_file
        self.default_duration = max(1, int(duration))
        self.info_window = info_window
        self._selected_combo = selected_combo
        self.adaptive_mode = adaptive_mode  # 0=off, 1=adaptive hot-swap, 2=reactive (BoundGuard), 3=static, 4=stop-and-restart + rollback
        self.metrics_csv = metrics_csv      # Path for per-second CSV metrics recording
        # Optional per-combo duration override: {combo_name: int_seconds}
        # When a combo is not in this map, default_duration is used.
        self.combo_durations = dict(combo_durations or {})

        self._viewer: UnifiedViewer = None
        self._index: int = 0
        self._running: bool = False
        self._end_time = None  # wall-clock end time for capped runs (epoch seconds)
        self._combinations: List[str] = self._load_combinations(schedule_file)

        # If a specific combination is requested, filter list to that single name
        if self._selected_combo:
            if self._selected_combo in self._combinations:
                self._combinations = [self._selected_combo]
            else:
                print(f"[Executor] ERROR: requested combination '{self._selected_combo}' not found in {os.path.basename(schedule_file)}")
                # Raised exception instead of os._exit(2)
                raise ValueError(f"Combination '{self._selected_combo}' not found.")

        if not self._combinations:
            print('[Executor] No combinations found in schedule file.')
            raise ValueError("No combinations found in schedule file.")

    # ------------------------------ Public API ------------------------------ #

    def start(self, duration: int = None):
        if self._running:
            print('[Executor] Start requested but execution is already running.')
            return
        self._running = True
        self._index = 0
        if duration is not None:
            self.default_duration = max(1, int(duration))
        # Cap total continuous execution time to 1 hour when running a single selected combination
        if getattr(self, '_selected_combo', None):
            self._end_time = time.time() + 3600.0
        else:
            self._end_time = None
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
            # Cache the parsed schedule so _placement_signature() can compare
            # combos without re-reading the YAML on every transition.
            self._schedules_cache = schedules
            return list(schedules.keys())
        except Exception as e:
            print(f"[Executor] ERROR: Failed to read schedules from {schedule_file}: {e}")
            self._schedules_cache = {}
            return []

    def _next_transition_is_inplace(self) -> bool:
        """Return True if the *next* mode 0 phase transition (current -> next
        combo) will be applied in-place because the placement is unchanged.

        Used to decide whether to:
          - schedule _after_stop EARLY (before viewer's timed_shutdown), and
          - cancel timed_shutdown in _after_stop to keep the viewer alive.

        Only meaningful for mode 0; the adaptive/reactive/static modes already
        keep the viewer alive across phase transitions by design.
        """
        if self.adaptive_mode != 0:
            return False
        if self._index + 1 >= len(self._combinations):
            return False
        cur = self._combinations[self._index]
        nxt = self._combinations[self._index + 1]
        return (self._placement_signature(cur) ==
                self._placement_signature(nxt))

    def _placement_signature(self, combo_name: str):
        """Return a hashable, infps-independent signature of a combo's placement.

        Two combos with the same set of (display, model, execution) tuples
        share a signature. Differences in `infps` are ignored on purpose, so
        a transition that only changes input rate (without moving any model
        between devices) is detected as "no real placement change" and can
        be applied in-place via apply_static_phase instead of restarting
        workers — which would otherwise inject a spurious cold-start spike
        into the V(t) trace (see qos_recovery_validation.py).
        """
        schedules = getattr(self, '_schedules_cache', None) or {}
        cfg = schedules.get(combo_name) or {}
        sig = []
        for entry in cfg.values():
            if not isinstance(entry, dict):
                continue
            sig.append((
                str(entry.get('display', '')).strip().lower(),
                str(entry.get('model', '')),
                str(entry.get('execution', '')).strip().lower(),
            ))
        return tuple(sorted(sig))

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
        if viewer is None:
            return
        try:
            viewer.stop_execution()
        except Exception:
            pass

    def _run_next(self):
        # Enforce max continuous runtime (1 hour) in selected-combo mode
        if getattr(self, '_end_time', None) is not None:
            remaining = int(self._end_time - time.time())
            if remaining <= 0:
                print('[Executor] Reached 1-hour cap for selected combination. Stopping execution.')
                self.stop()
                return

        if self._index >= len(self._combinations):
            # If a specific combination was requested (executor-only mode), loop until cap or stop
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

        combo = self._combinations[self._index]
        print(f"[Executor] Starting schedule: {combo}")

        adaptive = (self.adaptive_mode == 1)
        reactive = (self.adaptive_mode == 2)
        static = (self.adaptive_mode == 3)
        restart_rollback = (self.adaptive_mode == 4)

        # Mode 3 (static): keep the first combination running. From the second
        # combo onwards, only update infps + the CSV phase label — no worker
        # restart, no hot-swap. Used by qos_recovery_validation.py to plot the
        # "static" baseline against "stop-and-restart".
        if static and self._viewer is not None and getattr(self._viewer, '_run_active', False):
            try:
                self._viewer.apply_static_phase(self.schedule_file, combo)
            except Exception as e:
                print(f"[Executor] Static phase update failed: {e}")
            # Schedule the next phase advancement and return without touching
            # workers / start_execution.
            measured_duration = int(self.combo_durations.get(combo, self.default_duration))
            if measured_duration < 1:
                measured_duration = 1
            QTimer.singleShot(measured_duration * 1000, self._after_stop)
            return

        # Mode 0 (stop-and-restart) shortcut for same-placement transitions:
        # if the new combo only changes infps (the model->device assignment is
        # identical to the currently-running combo), we MUST NOT tear down and
        # restart workers — that would inject a spurious cold-start spike into
        # V(t). Instead, do an in-place infps update like static mode. The
        # "real" mode 0 restart is reserved for transitions that actually move
        # at least one model between devices (e.g. CPU->GPU offload), which is
        # the only kind of "stop-and-start" the paper figure cares about.
        if (not adaptive and not reactive and not static
                and self._viewer is not None
                and getattr(self._viewer, '_run_active', False)):
            prev_combo = getattr(self, '_prev_running_combo', None)
            if prev_combo and prev_combo != combo:
                if (self._placement_signature(prev_combo) ==
                        self._placement_signature(combo)):
                    print(f"[Executor] Mode 0: same placement {prev_combo} -> "
                          f"{combo}; applying infps in-place (no restart)")
                    try:
                        self._viewer.apply_static_phase(self.schedule_file, combo)
                    except Exception as e:
                        print(f"[Executor] In-place phase update failed: {e}")
                    measured_duration = int(self.combo_durations.get(
                        combo, self.default_duration))
                    if measured_duration < 1:
                        measured_duration = 1
                    self._prev_running_combo = combo
                    QTimer.singleShot(measured_duration * 1000, self._after_stop)
                    return

        # Reuse existing viewer if it exists
        if self._viewer is not None:
            print(f"[Executor] Reusing existing viewer for schedule: {combo} "
                  f"(adaptive={adaptive}, reactive={reactive})")
            try:
                # Collect previous state for reactive mode (rollback/fallback).
                # `prev_vscore` must be the *terminal* V(t) of the combo we
                # are transitioning away from (sampled live here), not the
                # cached value from when the previous combo started -- that
                # cached value is always ~0 because the workers had not yet
                # produced data when it was captured, and a 0 baseline makes
                # mode 2's rollback validation fire on every fallback.
                prev_combo = getattr(self, '_prev_running_combo', None)
                prev_model_set = getattr(self, '_prev_model_set', None)
                try:
                    from reactive_deploy import _collect_vscore as _live_vscore
                    if getattr(self._viewer, '_run_active', False):
                        prev_vscore = _live_vscore(self._viewer)
                    else:
                        prev_vscore = getattr(self, '_prev_stable_vscore', None)
                except Exception:
                    prev_vscore = getattr(self, '_prev_stable_vscore', None)

                self._viewer.update_combination(
                    self.schedule_file, combo,
                    adaptive=adaptive, reactive=reactive,
                    prev_combo_name=prev_combo,
                    prev_model_set=prev_model_set,
                    prev_vscore=prev_vscore,
                )
            except Exception as e:
                print(f"[Executor] Error updating combination: {e}")
                self._cleanup_viewer()
                self._viewer = None

        if self._viewer is None:
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

        # Pass metrics CSV path for per-second recording
        try:
            if self.metrics_csv:
                self._viewer.metrics_csv_path = self.metrics_csv
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

        # Bring window to front but avoid repositioning if it was already visible
        try:
            if not self._viewer.isVisible():
                self._viewer.show()
            else:
                self._viewer.raise_()
                self._viewer.activateWindow()
        except Exception:
            pass

        # Apply 1-second warmup: run for duration+1, but measurement starts after 1s inside viewer
        measured_duration = int(self.combo_durations.get(combo, self.default_duration))
        if measured_duration < 1:
            measured_duration = 1
        # If capped, ensure we don't exceed remaining time (include 1s warm-up)
        if getattr(self, '_end_time', None) is not None:
            remaining = max(0, int(self._end_time - time.time()))
            # Reserve 1 second for warm-up; run for at least 1 second if remaining is small
            run_duration = max(1, min(measured_duration + 1, remaining))
        else:
            run_duration = measured_duration + 1

        # In adaptive/reactive mode with a still-running viewer, skip start_execution —
        # the hot-swap manager already transitioned workers while the run continued.
        # Mode 4 (stop-and-restart + rollback) does call start_execution because
        # the forward transition itself is a stop-and-restart cycle.
        if (adaptive or reactive) and getattr(self._viewer, '_run_active', False):
            print(f"[Executor] {'Reactive' if reactive else 'Adaptive'}: execution continues, hot-swap in progress")
        else:
            self._viewer.start_execution(run_duration)

        # Capture prev state BEFORE we overwrite it, so the mode-2/mode-4
        # managers can use it as the rollback baseline.
        prev_combo_for_mgr = getattr(self, '_prev_running_combo', None)
        prev_model_set_for_mgr = getattr(self, '_prev_model_set', None)
        # `prev_vscore` must reflect the *terminal* V(t) of the combo we
        # are transitioning away from -- not the V(t) at the moment that
        # combo started, which would always be ~0 because the workers had
        # not yet generated any data. Sample V(t) live here so the
        # rollback delta is computed against the correct baseline.
        try:
            from reactive_deploy import _collect_vscore as _live_vscore
            if self._viewer is not None and getattr(self._viewer, '_run_active', False):
                prev_vscore_for_mgr = _live_vscore(self._viewer)
            else:
                prev_vscore_for_mgr = getattr(self, '_prev_stable_vscore', None)
        except Exception:
            prev_vscore_for_mgr = getattr(self, '_prev_stable_vscore', None)

        # Save current state for reactive mode's rollback/fallback tracking
        self._prev_running_combo = combo
        try:
            ms = getattr(self._viewer, 'model_settings', {}) or {}
            self._prev_model_set = set(
                cfg.get("model", "") for cfg in ms.values() if cfg.get("model", "")
            )
        except Exception:
            self._prev_model_set = None
        # Stable V(t) will be collected by ReactiveDeployManager after stabilisation;
        # store the last known value from the previous combo here.
        try:
            from reactive_deploy import _collect_vscore
            if self._viewer and getattr(self._viewer, '_run_active', False):
                self._prev_stable_vscore = _collect_vscore(self._viewer)
        except Exception:
            self._prev_stable_vscore = None

        # Mode 4: spawn the validation+rollback monitor on top of the just-
        # completed stop-and-restart transition.
        if restart_rollback and self._viewer is not None:
            try:
                from restart_rollback_deploy import RestartRollbackDeployManager
                mgr = RestartRollbackDeployManager(self._viewer)
                mgr.execute(
                    self.schedule_file, combo,
                    prev_combo_name=prev_combo_for_mgr,
                    prev_model_set=prev_model_set_for_mgr,
                    prev_vscore=prev_vscore_for_mgr,
                    run_duration=run_duration,
                )
            except Exception as e:
                print(f"[Executor] Mode 4: failed to spawn restart-rollback monitor: {e}")

        # Schedule moving to the next combination.
        # For adaptive/reactive/static: fire BEFORE timed_shutdown so we can
        # cancel it and keep the viewer alive across phase transitions.
        # For mode 0 with a same-placement next transition (in-place infps
        # update): same — fire early and cancel timed_shutdown so workers
        # survive into the in-place phase.
        # For mode 0 with a real placement change: fire after the
        # timed_shutdown with a buffer (the viewer will be torn down).
        if adaptive or reactive or static or self._next_transition_is_inplace():
            # Fire 1 second before timed_shutdown (which fires at run_duration*1000)
            after_ms = max(1000, (run_duration - 1) * 1000)
            QTimer.singleShot(after_ms, self._after_stop)
        else:
            buffer_ms = 1000
            QTimer.singleShot((run_duration * 1000) + buffer_ms, self._after_stop)

    def _after_stop(self):
        if not self._running:
            return

        adaptive = (self.adaptive_mode == 1)
        reactive = (self.adaptive_mode == 2)
        static = (self.adaptive_mode == 3)
        restart_rollback = (self.adaptive_mode == 4)
        # Mode 0 only: is the next transition a same-placement (in-place)
        # infps update? If so, keep the viewer alive across the boundary.
        mode0_inplace_next = self._next_transition_is_inplace()
        is_last = (self._index + 1 >= len(self._combinations))

        if (adaptive or reactive or static or mode0_inplace_next) and not is_last:
            # Adaptive/reactive/static OR mode-0-with-in-place-next: skip
            # stop_execution between combos so the viewer stays alive and
            # update_combination / apply_static_phase can take effect.
            # Also cancel the pending timed_shutdown from the previous
            # start_execution, otherwise it will fire independently and
            # kill the running workers.
            if self._viewer is not None:
                self._viewer.cancel_timed_shutdown()
        elif (reactive or restart_rollback) and is_last:
            # Reactive (mode 2) and restart-rollback (mode 4) on the last combo:
            # delay stop to allow the rollback decision to execute and produce
            # measurable results. Both monitors need 5s stabilisation +
            # transition time + recovery observation.
            if self._viewer is not None:
                self._viewer.cancel_timed_shutdown()
            label = "Reactive" if reactive else "Restart+Rollback"
            print(f"[Executor] {label}: last combo — extending run for potential rollback")
            extra_sec = 15  # extra seconds to observe rollback recovery
            QTimer.singleShot(extra_sec * 1000, self._reactive_final_stop)
            return  # skip normal _run_next advancement
        else:
            try:
                if self._viewer is not None:
                    self._viewer.stop_execution()
            except Exception as e:
                print(f"[Executor] Warning: stop_execution error: {e}")

        self._index += 1
        QTimer.singleShot(300, self._run_next)

    def _reactive_final_stop(self):
        """Final stop for reactive mode after rollback observation window."""
        print("[Executor] Reactive: rollback observation window ended. Stopping execution.")
        if not self._running:
            return
        try:
            if self._viewer is not None:
                self._viewer.stop_execution()
        except Exception as e:
            print(f"[Executor] Warning: stop_execution error: {e}")
        self._running = False
        self._index = 0
        self._set_start_button_enabled(True)
        self._write_best_header()
        # Signal auto-mode exit if applicable
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QTimer as _Qt
            app = QApplication.instance()
            if app:
                _Qt.singleShot(500, app.quit)
        except Exception:
            pass

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
    """Controller that InfoWindow binds to. Supports single or multi schedule execution."""

    def __init__(self, executor: ScheduleExecutor):
        self._executor = executor
        self._multi_files = []
        self._multi_idx = 0
        self._multi_running = False
        self._duration = executor.default_duration

    def _info(self) -> InfoWindow:
        return getattr(self._executor, 'info_window', None)

    def start_execution(self, duration):
        self._duration = max(1, int(duration))
        info = self._info()
        use_multi = False
        folder = None
        try:
            use_multi = bool(info.multiple_schedule_files.isChecked())
            folder = info.schedule_folder_input.text().strip()
        except Exception:
            use_multi = False

        if use_multi and folder:
            # Gather YAML files in folder
            try:
                if not os.path.isabs(folder):
                    folder = os.path.abspath(os.path.join(os.getcwd(), folder))
                files = []
                for name in os.listdir(folder):
                    if name.lower().endswith(('.yaml', '.yml')):
                        files.append(os.path.join(folder, name))
                files.sort()
            except Exception as e:
                print(f"[Controller] Failed to list schedule folder '{folder}': {e}")
                files = []

            if not files:
                print(f"[Controller] No YAML files found in folder: {folder}. Falling back to single schedule: {self._executor.schedule_file}")
                self._executor.start(self._duration)
                return

            # Start multi-file sequential execution
            self._multi_files = files
            self._multi_idx = 0
            self._multi_running = True
            try:
                info.start_button.setEnabled(False)
                info.stop_button.setEnabled(True)
            except Exception:
                pass
            QTimer.singleShot(10, self._start_next_file)
        else:
            # Single file
            self._executor.start(self._duration)

    def _start_next_file(self):
        if not self._multi_running:
            return
        if self._multi_idx >= len(self._multi_files):
            # Done
            self._multi_running = False
            try:
                self._info().start_button.setEnabled(True)
            except Exception:
                pass
            print('[Controller] All schedule files executed.')
            return
        current_schedule = self._multi_files[self._multi_idx]
        print(f"[Controller] Running schedule file {self._multi_idx+1}/{len(self._multi_files)}: {os.path.basename(current_schedule)}")
        # Create a fresh executor for this schedule file
        info = self._info()
        self._executor = ScheduleExecutor(schedule_file=current_schedule, duration=self._duration, info_window=info)
        # Ensure InfoWindow parent points here
        try:
            info.parent = self
        except Exception:
            pass
        self._executor.start(self._duration)
        # Start monitoring this executor for completion
        QTimer.singleShot(200, self._monitor_current_done)

    def _monitor_current_done(self):
        if not self._multi_running:
            return
        try:
            ex = self._executor
            if not ex._running and ex._index == 0:
                # Finished this file
                self._multi_idx += 1
                QTimer.singleShot(200, self._start_next_file)
                return
        except Exception:
            pass
        # Keep polling
        QTimer.singleShot(300, self._monitor_current_done)

    def stop_execution(self):
        # Stop current execution; if in multi mode, stop and cancel the rest
        self._multi_running = False
        try:
            self._info().start_button.setEnabled(True)
        except Exception:
            pass
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
    parser.add_argument('--adaptive-mode', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='Adaptive deploy mode: 0=off (stop-and-restart between combos), '
                             '1=adaptive hot-swap, 2=reactive / BoundGuard '
                             '(hot-swap + validation/rollback/fallback), '
                             '3=static (no deploy change between combos — only sweeps infps; '
                             'used by qos_recovery_validation as the static baseline), '
                             '4=stop-and-restart + rollback (mode 0 transitions with mode 2 '
                             'style validation/rollback; the "Stop-and-restart (+Rollback)" '
                             'baseline used in the four-strategy comparison) '
                             '(default: 0)')
    parser.add_argument('--metrics-csv', type=str, default=None,
                        help='Path to CSV file for per-second metrics recording')
    parser.add_argument('--combo-duration', action='append', default=[],
                        metavar='COMBO=SECONDS',
                        help='Per-combination duration override. Repeatable. '
                             'Combos not listed here use --duration. '
                             'Example: --combo-duration combination_failure=5')
    args = parser.parse_args()

    # Parse --combo-duration overrides into a {combo_name: seconds} dict
    combo_durations: dict = {}
    for spec in (args.combo_duration or []):
        if "=" not in spec:
            print(f"[Main] WARNING: ignoring malformed --combo-duration '{spec}' (need COMBO=SECONDS)")
            continue
        name, _, val = spec.partition("=")
        name = name.strip()
        try:
            combo_durations[name] = max(1, int(val.strip()))
        except ValueError:
            print(f"[Main] WARNING: ignoring --combo-duration '{spec}' (seconds must be int)")

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

    app = QApplication.instance() or QApplication(sys.argv)

    # Create the InfoWindow instance
    info = InfoWindow(parent=None)

    # Shared graceful shutdown handler that mimics pressing Stop in InfoWindow
    def _graceful_shutdown(signum=None, frame=None):
        try:
            print(f"[Main] Received signal {signum}; initiating graceful shutdown...")
        except Exception:
            pass
        try:
            # Prefer stopping executor (mirrors Stop Execution)
            nonlocal_executor = getattr(_graceful_shutdown, '_executor', None)
            if nonlocal_executor is not None:
                try:
                    nonlocal_executor.stop()
                except Exception:
                    pass
        except Exception:
            pass
        # Hide InfoWindow and quit the app event loop
        try:
            info.hide()
        except Exception:
            pass
        try:
            QTimer.singleShot(50, app.quit)
        except Exception:
            try:
                app.quit()
            except Exception:
                pass

    # If schedule_name is provided, run executor-only mode without showing InfoWindow
    if args.schedule_name:
        try:
            # Ensure InfoWindow stays hidden in combination execution mode
            info.hide()
            info.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        except Exception:
            pass
        
        try:
            executor = ScheduleExecutor(schedule_file=schedule_path, duration=args.duration, info_window=info,
                                        selected_combo=args.schedule_name,
                                        adaptive_mode=getattr(args, 'adaptive_mode', 0),
                                        metrics_csv=getattr(args, 'metrics_csv', None),
                                        combo_durations=combo_durations)
        except ValueError as e:
            print(f"[Main] ERROR: {e}")
            return 1

        # Link executor for shutdown handler
        try:
            setattr(_graceful_shutdown, '_executor', executor)
        except Exception:
            pass
        # Install signal handlers to perform graceful stop on SIGTERM/SIGINT
        try:
            signal.signal(signal.SIGTERM, _graceful_shutdown)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGINT, _graceful_shutdown)
        except Exception:
            pass
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
        return exit_code

    # Otherwise, show InfoWindow and use full GUI mode
    info.show()

    # Default GUI mode with controller
    try:
        executor = ScheduleExecutor(schedule_file=schedule_path, duration=args.duration, info_window=info,
                                    adaptive_mode=getattr(args, 'adaptive_mode', 0),
                                    metrics_csv=getattr(args, 'metrics_csv', None),
                                    combo_durations=combo_durations)
    except ValueError as e:
        print(f"[Main] ERROR: {e}")
        return 1
    
    controller = Controller(executor)

    # Assign controller as the parent so InfoWindow's built-in handlers call our methods
    try:
        info.parent = controller
    except Exception:
        pass

    # Link executor for shutdown handler and install signals
    try:
        setattr(_graceful_shutdown, '_executor', executor)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _graceful_shutdown)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGINT, _graceful_shutdown)
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
                multi_running = False
                try:
                    multi_running = bool(getattr(controller, '_multi_running', False))
                except Exception:
                    multi_running = False
                if not executor._running and executor._index == 0 and not multi_running:
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
    return exit_code

if __name__ == "__main__":
    sys.exit(main())