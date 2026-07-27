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


# D5: single mode->method mapping for THIS entry point (schedule_executor_main).
# best_deploy_finder_executor.py uses a different, smaller numbering (see its header
# and the note on ScheduleExecutor.adaptive_mode). Reused by docs/runbooks.
MODE_METHOD_MAP = {
    0: "Stop-and-restart",
    1: "Adaptive hot-swap",
    2: "BoundGuard (reactive: validation window + rollback/fallback)",
    3: "Static",
    4: "Stop-and-restart + rollback (baseline)",
}


class ScheduleExecutor:
    """Encapsulates state and behavior for running schedule combinations sequentially."""

    def __init__(self, schedule_file: str, duration: int, info_window: InfoWindow,
                 selected_combo: str = None, adaptive_mode: int = 0, metrics_csv: str = None,
                 combo_durations: dict = None,
                 combo_triggers: dict = None,
                 qos_epsilon: float = 1.0,
                 qos_tv: float = 3.0,
                 qos_window_T: int = 3,
                 qos_slope_threshold: float = 0.5,
                 background: bool = False,
                 background_max_new_tokens: int = 64):
        self.schedule_file = schedule_file
        self.default_duration = max(1, int(duration))
        self.info_window = info_window
        self._selected_combo = selected_combo
        # adaptive_mode -> strategy (single source of truth; see MODE_METHOD_MAP below).
        #   0 = Stop-and-restart      (default / the implicit else path)
        #   1 = Adaptive hot-swap
        #   2 = reactive == BoundGuard (validation window + rollback/fallback)
        #   3 = Static
        #   4 = Stop-and-restart + rollback (baseline)
        # NOTE (D5): this numbering is NOT shared by best_deploy_finder_executor.py,
        # whose GUI --adaptive-mode uses {0,1,2} with 1=Adaptive, 2=reactive and no
        # static/mode-4. Some analysis scripts therefore drive "BoundGuard" as mode 1
        # (the hot-swap manager) rather than mode 2. Always read the mapping for the
        # entry point you are in.
        self.adaptive_mode = adaptive_mode
        self.metrics_csv = metrics_csv      # Path for per-second CSV metrics recording
        # Optional per-combo duration override: {combo_name: int_seconds}
        # When a combo is not in this map, default_duration is used.
        self.combo_durations = dict(combo_durations or {})
        # QoS-driven advancement: {combo_name: "v-above" | "validate"}.
        #   v-above  : advance as soon as V(t) > epsilon (or combo_durations cap)
        #   validate : wait T_v seconds after entering combo, then check V(t);
        #              if > epsilon advance, otherwise stay (up to combo_durations cap)
        self.combo_triggers = dict(combo_triggers or {})
        self.qos_epsilon = float(qos_epsilon)
        self.qos_tv = float(qos_tv)
        self.qos_window_T = int(qos_window_T)
        # Trend-based validation: commit a candidate whose backlog is draining
        # faster than this (frames/sec) even if V(t) is still above eps. Flat
        # (mu~lambda) or rising backlog -> advance. See check_validate.
        self.qos_slope_threshold = float(qos_slope_threshold)
        # Runtime state for the polling logic.
        self._qos_poll_timer = None
        self._qos_advance_fired = False

        # Background generative load (Q3/Q5 mixed set). Default OFF so every
        # background-off result reproduces exactly; when ON, generative combo
        # entries (llama1b/qwen2_vl) run as isolated child processes and their
        # placement is a coarse switch target. Vision dispatch / QoS untouched.
        from runtime.background_llm import BackgroundManager
        self._bg = BackgroundManager(enabled=bool(background),
                                     max_new_tokens=int(background_max_new_tokens),
                                     log=print)

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
        # Per-episode record of each *validated* candidate's observed service
        # rate (frames completed / s over the tail of its T_v window). Used by
        # _revert_to_best when the candidate budget is exhausted: instead of
        # default-committing to whatever candidate happened to be last in the
        # predictor ranking (which has no reason to be good -- in Q4 it was the
        # worst of five), BoundGuard reverts to the best placement it actually
        # observed. Auto-advanced candidates (== the violating placement) are
        # deliberately NOT recorded, so we never revert to the known-bad combo.
        self._cand_obs = {}
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
        try:
            self._bg.shutdown()
        except Exception as e:
            print(f"[Executor] background shutdown error: {e}")
        self._set_start_button_enabled(True)
        print('[Executor] Execution stopped by user.')

    def _fire_window_end(self):
        # Fixed observation window (work 5) elapsed. Stop the run cleanly even if a
        # combo committed/reverted and stopped advancing. Mirror the natural
        # schedule-completion path (dump accounting, reset _index=0) so the auto-quit
        # poller fires and the process exits at exactly the window length.
        if not getattr(self, '_running', False):
            return
        print('[Executor] Fixed observation window elapsed. Stopping execution.')
        try:
            self._dump_accounting()
        except Exception:
            pass
        try:
            self._write_best_header()
        except Exception:
            pass
        self.stop()
        self._index = 0

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

    def _generative_entries(self, combo_name: str):
        """Return [(model, execution), ...] for the generative (llm/vlm) entries
        of a combo. Used to drive the background manager's coarse placement
        switch. Vision entries are excluded (the viewer handles those)."""
        import model_registry as reg
        schedules = getattr(self, '_schedules_cache', None) or {}
        cfg = schedules.get(combo_name) or {}
        entries = []
        for entry in cfg.values():
            if not isinstance(entry, dict):
                continue
            model = str(entry.get('model', '') or '')
            try:
                if model and reg.is_llm_like(model):
                    entries.append((model, str(entry.get('execution', 'cpu') or 'cpu')))
            except Exception:
                continue
        return entries

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
        # Enforce the hard end-time: the 1-hour cap in selected-combo mode, or the
        # fixed observation window (work 5) anchored at the first candidate. Mirror
        # the natural schedule-completion path (dump accounting, reset _index=0) so
        # the auto-quit poller (_on_all_done_quit, which requires _index==0) fires
        # and the process exits cleanly instead of hanging until the outer timeout.
        if getattr(self, '_end_time', None) is not None:
            remaining = int(self._end_time - time.time())
            if remaining <= 0:
                print('[Executor] Hard end-time reached. Stopping execution.')
                try:
                    self._dump_accounting()
                except Exception:
                    pass
                try:
                    self._write_best_header()
                except Exception:
                    pass
                self.stop()
                self._index = 0
                return

        if self._index >= len(self._combinations):
            # If a specific combination was requested (executor-only mode), loop until cap or stop
            if getattr(self, '_selected_combo', None):
                self._index = 0
                QTimer.singleShot(300, self._run_next)
                return
            print('[Executor] All combinations executed. Leaving windows open.')
            self._dump_accounting()
            self._write_best_header()
            try:
                self._bg.shutdown()
            except Exception as e:
                print(f"[Executor] background shutdown error: {e}")
            self._running = False
            self._index = 0
            self._set_start_button_enabled(True)
            return

        combo = self._combinations[self._index]
        print(f"[Executor] Starting schedule: {combo}")

        # Fixed observation window (work 5): anchor the hard end-time at the first
        # recovery candidate so every variant runs an identical wall-clock window
        # from overload, regardless of how the candidate walk advances or where it
        # lands. Without this the run length equals the *final combo's* nominal
        # duration, which differs by variant (Adaptive holds cand_1 for its full
        # duration; walk variants land on cand_5's shorter tail; no-dwell reverts
        # at cand_4), making the strict-t_r recovery verdict depend on run length.
        # The overload (burst) combo has no trigger, so cand_1 begins at a fixed
        # offset after overload injection for every variant -> dur is fixed.
        _hw = os.environ.get("FSRR_HARD_WINDOW_S")
        if _hw and combo.startswith("cand_") and getattr(self, '_end_time', None) is None:
            self._end_time = time.time() + float(_hw)
            print(f"[Executor] Fixed window armed: hard end {float(_hw):.0f}s from {combo}")
            # A committed/reverted final combo stops advancing, so _run_next (and its
            # end-time check) is never re-entered -- the run would hold until the
            # outer timeout. Schedule an independent one-shot that fires at _end_time
            # regardless of commit/hold, so the run length is exactly the window.
            QTimer.singleShot(int(float(_hw) * 1000), self._fire_window_end)

        # Reconcile background generative placement to this combo (coarse switch:
        # kill old-device child, start new-device child). No-op when disabled or
        # when the combo has no generative entries. Kept ahead of every early
        # return below so LLM placement tracks each combo transition.
        try:
            self._bg.sync(self._generative_entries(combo))
        except Exception as e:
            print(f"[Executor] background sync error: {e}")

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
            self._maybe_start_qos_trigger(combo)
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
                    self._maybe_start_qos_trigger(combo)
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

        # Placement signature of the combo that was running BEFORE this one, so
        # the validation trigger can detect a candidate whose placement is
        # identical to the current (already-violating) one -- there is no
        # transition to evaluate, so it must auto-advance (see
        # _maybe_start_qos_trigger).
        self._prev_placement_sig = (
            self._placement_signature(prev_combo_for_mgr)
            if prev_combo_for_mgr else None)
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
        # Cancel any previous duration timer before scheduling a new one.
        if getattr(self, '_duration_timer', None) is not None:
            try:
                self._duration_timer.stop()
            except Exception:
                pass
            self._duration_timer = None

        if adaptive or reactive or static or self._next_transition_is_inplace():
            # Fire 1 second before timed_shutdown (which fires at run_duration*1000)
            after_ms = max(1000, (run_duration - 1) * 1000)
        else:
            after_ms = (run_duration * 1000) + 1000

        self._duration_timer = QTimer(self._viewer if self._viewer else None)
        self._duration_timer.setSingleShot(True)
        self._duration_timer.timeout.connect(self._after_stop)
        self._duration_timer.start(after_ms)

        # QoS-driven advancement: optionally replace/complement the timer above
        # with a V(t) poll that shortens the phase when a violation is detected.
        self._maybe_start_qos_trigger(combo)

    # ------------------------------------------------------------------
    # QoS-driven advancement
    # ------------------------------------------------------------------
    def _stop_qos_poll(self):
        if self._qos_poll_timer is not None:
            try:
                self._qos_poll_timer.stop()
            except Exception:
                pass
            self._qos_poll_timer = None

    def _current_vscore(self):
        if self._viewer is None:
            return None
        try:
            from reactive_deploy import _collect_vscore
            return _collect_vscore(self._viewer, window_T=self.qos_window_T)
        except Exception:
            return None

    def _handler_snapshot(self):
        """Snapshot each active view handler's cumulative accumulators
        (infer/wait sums and counts) at combo-entry time. The validation V is
        then computed from the DELTA since this snapshot, so it reflects only
        post-transition frames -- the cumulative avg_infer/avg_wait the handler
        exposes are lifetime averages (total/count, never reset on hot-swap) and
        are therefore contaminated by the previous (burst) phase.
        Returns {vname: (total_infer, infer_count, total_wait, wait_count)}."""
        snap = {}
        if self._viewer is None:
            return snap
        for vname in ("view1", "view2", "view3", "view4"):
            h = getattr(self._viewer, f"{vname}_handler", None)
            if h is None:
                continue
            snap[vname] = (
                float(getattr(h, 'total_infer_time', 0.0) or 0.0),
                int(getattr(h, 'infer_count', 0) or 0),
                float(getattr(h, 'total_wait_ms', 0.0) or 0.0),
                int(getattr(h, 'wait_count', 0) or 0),
            )
        return snap

    def _postswap_vscore(self, snap):
        """V(t) computed from post-transition frames only (delta since `snap`).
        Same definition as reactive_deploy._collect_vscore but with per-view
        latency = (Δtotal_infer/Δinfer_count) + (Δtotal_wait/Δwait_count) so the
        pre-transition (burst) history does not leak into the validation."""
        if self._viewer is None or not snap:
            return None
        views_without = getattr(self._viewer, 'views_without_model', set())
        v_sum = 0.0
        n_active = 0
        for vname in ("view1", "view2", "view3", "view4"):
            if vname in views_without or vname not in snap:
                continue
            h = getattr(self._viewer, f"{vname}_handler", None)
            if h is None:
                continue
            ti0, ic0, tw0, wc0 = snap[vname]
            ti = float(getattr(h, 'total_infer_time', 0.0) or 0.0)
            ic = int(getattr(h, 'infer_count', 0) or 0)
            tw = float(getattr(h, 'total_wait_ms', 0.0) or 0.0)
            wc = int(getattr(h, 'wait_count', 0) or 0)
            d_ic, d_wc = ic - ic0, wc - wc0
            if d_ic <= 0:
                continue  # no post-transition frames yet -> skip (not diluted)
            infer_ms = (ti - ti0) / d_ic
            wait_ms = ((tw - tw0) / d_wc) if d_wc > 0 else 0.0
            li = infer_ms + wait_ms
            if li <= 0:
                continue
            ms = getattr(h, 'model_settings', None) or {}
            infps = float((ms.get(vname) or {}).get('infps', 10.0) or 10.0)
            l_slo = 1000.0 / infps if infps > 0 else 100.0
            v_sum += max(0.0, (li / l_slo) - 1.0)
            n_active += 1
        if n_active == 0:
            return None
        return v_sum / n_active

    def _current_backlog(self):
        """Total instantaneous frame-queue backlog Q(k) across the four views
        (same qsize() sum the metrics loop logs). None if unavailable.

        This is the fluid-model backlog: its slope over the T_v window is
        -(mu*-lambda) for a draining (feasible) candidate and >=0 for an
        infeasible one, independent of backlog inherited from prior candidates.
        """
        if self._viewer is None:
            return None
        try:
            total = 0
            for vn in ("view1", "view2", "view3", "view4"):
                fq = getattr(self._viewer, f"{vn}_frame_queue", None)
                if fq is not None:
                    total += int(fq.qsize())
            return total
        except Exception:
            return None

    def _current_drops(self):
        """Per-view cumulative dropped-frame counts (queue-full drops), summed
        across both feeders. {view: total_drops}. Used by the saturation guard:
        a queue that is dropping cannot let its backlog grow, so the observed
        slope no longer estimates lambda-mu -- ongoing drops are direct evidence
        that arrival > service (infeasible), independent of slope/V."""
        out = {v: 0 for v in ("view1", "view2", "view3", "view4")}
        if self._viewer is None:
            return out
        for fname in ("video_feeder", "resnet_feeder"):
            f = getattr(self._viewer, fname, None)
            dc = getattr(f, "drop_counts", None) if f is not None else None
            if not dc:
                continue
            for v in out:
                try:
                    out[v] += int(dc.get(v, 0) or 0)
                except Exception:
                    pass
        return out

    def _current_enqueues(self):
        """Per-view cumulative successfully-enqueued frame counts, summed across
        both feeders. {view: total_enqueued}. Used only by the transition
        frame-accounting audit (enqueued == completed + dropped + residual)."""
        out = {v: 0 for v in ("view1", "view2", "view3", "view4")}
        if self._viewer is None:
            return out
        for fname in ("video_feeder", "resnet_feeder"):
            f = getattr(self._viewer, fname, None)
            ec = getattr(f, "enqueue_counts", None) if f is not None else None
            if not ec:
                continue
            for v in out:
                try:
                    out[v] += int(ec.get(v, 0) or 0)
                except Exception:
                    pass
        return out

    def _dump_accounting(self):
        """Emit the final frame-accounting line for the transition-losslessness
        audit. Gated by FSRR_ACCT so default runs are unchanged. Per view:
        enqueued / completed(handler infer_count) / dropped / residual(qsize)."""
        if not os.environ.get("FSRR_ACCT"):
            return
        try:
            enq = self._current_enqueues()
            drp = self._current_drops()
            for v in ("view1", "view2", "view3", "view4"):
                h = getattr(self._viewer, f"{v}_handler", None)
                comp = int(getattr(h, 'infer_count', 0) or 0) if h is not None else 0
                fq = getattr(self._viewer, f"{v}_frame_queue", None)
                resid = int(fq.qsize()) if fq is not None else 0
                e = enq.get(v, 0); d = drp.get(v, 0)
                bal = e - (comp + d + resid)
                print(f"[ACCT] {v}: enqueued={e} completed={comp} dropped={d} "
                      f"residual={resid} balance(enq-comp-drop-resid)={bal}",
                      flush=True)
        except Exception as ex:
            print(f"[ACCT] error: {ex}", flush=True)

    def _total_infer_count(self):
        """Sum of completed-frame counters across foreground view handlers.
        The delta of this over a window is the aggregate service rate mu_i
        (frames/s actually completed). Unlike backlog slope, this stays
        meaningful under saturation: a queue clipped at cap still tells us how
        many frames drained, so it discriminates candidates in exactly the
        regime (Q4) where slope and V(t) do not."""
        if self._viewer is None:
            return None
        vw = getattr(self._viewer, 'views_without_model', set())
        total = 0
        any_h = False
        for vname in ("view1", "view2", "view3", "view4"):
            if vname in vw:
                continue
            h = getattr(self._viewer, f"{vname}_handler", None)
            if h is None:
                continue
            total += int(getattr(h, 'infer_count', 0) or 0)
            any_h = True
        return total if any_h else None

    @staticmethod
    def _slope(samples):
        """Least-squares slope (units/sec) of (t, y) samples; None if <2 or
        degenerate."""
        n = len(samples)
        if n < 2:
            return None
        sx = sum(t for t, _ in samples)
        sy = sum(y for _, y in samples)
        sxx = sum(t * t for t, _ in samples)
        sxy = sum(t * y for t, y in samples)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-9:
            return None
        return (n * sxy - sx * sy) / denom

    def _qos_advance(self, reason):
        if self._qos_advance_fired:
            return
        self._qos_advance_fired = True
        self._stop_qos_poll()
        # Cancel the duration-based timer so it doesn't double-fire _after_stop.
        if getattr(self, '_duration_timer', None) is not None:
            try:
                self._duration_timer.stop()
            except Exception:
                pass
            self._duration_timer = None
        print(f"[Executor] QoS-triggered advance: {reason}")
        # Immediately move to the next combo.
        QTimer.singleShot(0, self._after_stop)

    def _commit_and_stay(self, combo):
        """Validation passed -> commit-and-stay: skip the remaining candidate
        phases and hold the committed placement for the rest of the scheduled
        time. This matches the paper's algorithm (a recovery episode completes
        on commit) rather than the previous visit-all behaviour (which cycled
        through later, worse candidates and injected spurious V(t) spikes).

        Total run time is preserved -- the committed placement absorbs the
        skipped candidates' durations -- so BoundGuard stays time-matched to the
        baselines (which also hold their chosen placement for the whole run).

        V(t) monitoring continues via the per-second metrics CSV. A re-violation
        would be a NEW recovery episode (fresh detection window + a new predictor
        ranking under the then-current conditions), not a resume of this episode;
        the pre-listed schedule cannot express that, and here the committed
        placement stays feasible so no re-violation occurs (confirmed from the
        CSV trace). Background (LLM/VLM) placement is left as-is at the committed
        combo, since no further combo transition -> no bg.sync (intended)."""
        self._qos_advance_fired = True
        self._stop_qos_poll()
        combos = self._combinations
        idx = self._index
        skipped = [combos[j] for j in range(idx + 1, len(combos))]
        skipped_secs = sum(max(1, int(self.combo_durations.get(c, self.default_duration)))
                           for c in skipped)
        # committed combo's own not-yet-consumed cap (validate fires ~T_v in)
        own_remaining = max(0, int(self.combo_durations.get(combo, self.default_duration))
                            - int(self.qos_tv))
        hold_secs = max(1, own_remaining + skipped_secs)
        # Cancel the per-candidate duration timer; hold the committed placement.
        if getattr(self, '_duration_timer', None) is not None:
            try:
                self._duration_timer.stop()
            except Exception:
                pass
            self._duration_timer = None
        # Jump the index to the last phase so the next _after_stop ends the run;
        # the viewer keeps running the committed placement in the meantime.
        self._index = len(combos) - 1
        self._duration_timer = QTimer(self._viewer if self._viewer else None)
        self._duration_timer.setSingleShot(True)
        self._duration_timer.timeout.connect(self._after_stop)
        self._duration_timer.start(hold_secs * 1000)
        if skipped:
            print(f"[Executor] commit-and-stay: committed '{combo}'; skipping "
                  f"{skipped[0]}..{skipped[-1]} ({len(skipped)} phases); holding "
                  f"committed placement for {hold_secs}s; V(t) monitoring continues.")
        else:
            print(f"[Executor] commit-and-stay: committed '{combo}' (last phase; "
                  f"nothing to skip).")

    def _revert_to_best(self, current_combo):
        """Candidate budget exhausted (all N_cand validated, none committed).

        Instead of default-committing to `current_combo` (the last candidate in
        the predictor ranking -- no reason to be good, and in Q4 it was the
        worst of five), revert to the candidate with the highest *observed*
        service rate. Reverting is itself one more transition, so the search
        bound becomes T + N_cand(T_v + delta) + delta. If the best observed
        placement is already the current one, we hold it (no extra delta).

        Only *validated* candidates are in self._cand_obs; the auto-advanced
        candidate (the violating placement itself) is excluded by construction,
        so we never revert to the known-bad combo."""
        obs = {k: v for k, v in self._cand_obs.items()
               if v.get("rate") is not None}
        obs_str = ", ".join(f"{k}={v['rate']:.1f}fps" for k, v in obs.items()) or "none"
        if not obs:
            print(f"[Executor] exhausted: no candidate service-rate observations; "
                  f"holding current placement {current_combo}.")
            self._commit_and_stay(current_combo)
            return
        best_combo, best = max(obs.items(), key=lambda kv: kv[1]["rate"])
        cur_sig = self._placement_signature(current_combo)
        print(f"[Executor] exhausted -> revert to best (cand={best_combo}, "
              f"service_rate={best['rate']:.1f} fps); observations: {obs_str}")
        if best["sig"] == cur_sig:
            print(f"[Executor] best placement == current ({current_combo}); "
                  f"no transition needed (saving one delta).")
            self._commit_and_stay(current_combo)
            return
        # Transition to the best placement (adaptive hot-swap w/ backlog
        # preservation), then hold it for the remaining scheduled time.
        self._qos_advance_fired = True
        self._stop_qos_poll()
        hold_secs = max(1, int(self.combo_durations.get(
            current_combo, self.default_duration)) - int(self.qos_tv))
        if getattr(self, '_duration_timer', None) is not None:
            try:
                self._duration_timer.stop()
            except Exception:
                pass
            self._duration_timer = None
        try:
            # Same dispatch as _run_next: mode 1 -> adaptive hot-swap (backlog
            # preserved); mode 2 -> reactive. Falling back to the default
            # stop+restart path would cold-start the workers and discard backlog.
            self._viewer.update_combination(
                self.schedule_file, best_combo,
                adaptive=(self.adaptive_mode == 1),
                reactive=(self.adaptive_mode == 2))
            self._prev_running_combo = best_combo
            self._prev_placement_sig = best["sig"]
            print(f"[Executor] reverted to best placement '{best_combo}'; "
                  f"holding for {hold_secs}s; V(t) monitoring continues.")
        except Exception as e:
            print(f"[Executor] revert transition error: {e}; holding current.")
        self._index = len(self._combinations) - 1
        self._duration_timer = QTimer(self._viewer if self._viewer else None)
        self._duration_timer.setSingleShot(True)
        self._duration_timer.timeout.connect(self._after_stop)
        self._duration_timer.start(hold_secs * 1000)

    def _maybe_start_qos_trigger(self, combo):
        self._qos_advance_fired = False
        self._stop_qos_poll()
        policy = self.combo_triggers.get(combo)
        if not policy:
            return
        # Validation evaluates a *state change*. A candidate whose placement is
        # identical to the combo we just left is not a transition -- it is the
        # placement that already triggered the violation, so there is nothing new
        # to validate and measuring it only captures the leftover drain transient
        # (which nearly mis-committed cand_1 = top-1 = burst). Auto-advance it.
        # Safe-by-construction: advancing a candidate costs a bounded search step,
        # whereas committing an already-failing placement means an unbounded
        # violation.
        # The auto-advance guard (skip a candidate whose placement equals the
        # current violating one) is itself a PROGRESS mechanism. The C2 hybrid
        # ablation (dwell kept, progress removed) must run WITHOUT it, otherwise
        # its repeat-top1 schedule is skipped instantly instead of being dwelt on
        # T_v each time. Gated by FSRR_NO_AUTOADVANCE so BoundGuard/A/B are
        # unaffected (default keeps the guard on).
        if policy == "validate" and not os.environ.get("FSRR_NO_AUTOADVANCE"):
            prev_sig = getattr(self, '_prev_placement_sig', None)
            if prev_sig is not None and self._placement_signature(combo) == prev_sig:
                self._qos_advance(
                    f"combo={combo} placement identical to the current "
                    f"(violating) placement; no transition to validate, "
                    f"auto-advancing")
                return
        # Clear the rolling V(t) history on combo entry so carried-over
        # samples from the previous phase don't pre-trigger the advance.
        try:
            if self._viewer is not None:
                self._viewer._v_history_mode2 = []
        except Exception:
            pass
        poll_ms = 200
        eps = self.qos_epsilon
        entered_at = time.time()
        poll_count = [0]
        # Completed-frame count at combo entry (for the no-dwell service-rate obs).
        entry_count = [self._total_infer_count()]
        # Backlog samples (elapsed_s, Q) collected across the T_v window for the
        # trend-based feasibility test (see check_validate).
        bl_samples = []
        # Drop samples (elapsed_s, {view: cumulative_drops}) for the saturation
        # guard: if any view keeps dropping through the last third of the window,
        # its queue is clipped at cap and the slope no longer estimates lambda-mu,
        # so the candidate is infeasible regardless of slope/V.
        drop_samples = []
        # Completed-frame counter samples (elapsed_s, total_infer_count) for the
        # best-so-far service-rate criterion used by _revert_to_best.
        count_samples = []
        slope_th = self.qos_slope_threshold
        # Snapshot handler accumulators NOW so the validation V is computed from
        # post-transition frames only (the exposed avg_infer/avg_wait are
        # lifetime cumulative averages, contaminated by the previous phase).
        hsnap = self._handler_snapshot()

        def check_v_above():
            if self._qos_advance_fired or not self._running:
                self._stop_qos_poll()
                return
            v = self._current_vscore()
            poll_count[0] += 1
            # Log every 10th poll (2s) so we can see progress without spam
            if poll_count[0] % 10 == 1:
                elapsed = time.time() - entered_at
                print(f"[Executor] QoS-poll combo={combo} elapsed={elapsed:.1f}s "
                      f"V(t)={'None' if v is None else f'{v:.3f}'} eps={eps}")
            if v is None:
                return
            if v > eps:
                # no-dwell (reactive ablation A/B): advance the INSTANT V>eps, with
                # NO T_v observation. Record a service-rate observation over the
                # (tiny) elapsed window so best-so-far is kept -- the whole point
                # is that with no dwell this measurement is near-meaningless, so
                # best-so-far may pick a bad placement. Same best-so-far / revert
                # termination as validate: on the last candidate, revert instead
                # of running off the end.
                elapsed_nd = time.time() - entered_at
                svc = None
                ic_now = self._total_infer_count()
                if ic_now is not None and entry_count[0] is not None and elapsed_nd > 0:
                    svc = (ic_now - entry_count[0]) / elapsed_nd
                self._cand_obs[combo] = {"rate": svc,
                                         "sig": self._placement_signature(combo)}
                is_last_nd = (self._index >= len(self._combinations) - 1)
                reason = (f"no-dwell: V(t)={v:.3f} > eps={eps} in combo={combo} "
                          f"(svc_rate={'None' if svc is None else f'{svc:.1f}fps'}, "
                          f"no T_v observation)")
                if is_last_nd:
                    print(f"[Executor] QoS-triggered advance: {reason} "
                          f"(last candidate -> revert-to-best)")
                    self._revert_to_best(combo)
                else:
                    self._qos_advance(reason)

        def check_validate():
            if self._qos_advance_fired or not self._running:
                self._stop_qos_poll()
                return
            elapsed = time.time() - entered_at
            # Sample the backlog Q(k) every poll across the whole T_v window so
            # we can fit its slope at the end.
            bl = self._current_backlog()
            if bl is not None:
                bl_samples.append((elapsed, bl))
            drop_samples.append((elapsed, self._current_drops()))
            _ic = self._total_infer_count()
            if _ic is not None:
                count_samples.append((elapsed, _ic))
            if elapsed < self.qos_tv:
                return  # still inside the T_v validation window (collecting)
            # Validation V uses post-transition frames only (see _postswap_vscore);
            # v_report is the lifetime-cumulative value kept only for the log so we
            # can see the contamination gap.
            # Validation V uses the paper metric: ell_i over the last Delta
            # seconds (post-swap by construction) + the T-window (_v_history
            # cleared on combo entry). The old _postswap snapshot-delta workaround
            # is no longer needed now that ell_i itself is Delta-windowed.
            v = self._current_vscore()
            v_report = self._postswap_vscore(hsnap)  # kept only for log comparison
            # Fit the slope over the *last third* of the T_v window only. The
            # early part after a hot-swap is a transition transient (restarted
            # workers momentarily burn down the inherited backlog, then it
            # refills to the ceiling if the candidate is infeasible). Measuring
            # only the settled tail distinguishes a genuinely draining (feasible)
            # candidate from one that just spiked-then-saturated. (Same-placement
            # candidates, whose transient dominates the whole window, are already
            # auto-advanced above.)
            tail_start = self.qos_tv * (2.0 / 3.0)
            tail = [(t, y) for (t, y) in bl_samples if t >= tail_start]
            slope = self._slope(tail if len(tail) >= 3 else bl_samples)
            slope_str = 'None' if slope is None else f'{slope:.2f}'
            # Saturation guard: per-view drops accrued over the last third of the
            # window. If any view is still dropping, its queue is clipped at cap
            # and the slope cannot estimate lambda-mu -> infeasible.
            tail_drops = [(t, d) for (t, d) in drop_samples if t >= tail_start]
            drop_delta = {_vw: 0 for _vw in ("view1", "view2", "view3", "view4")}
            if len(tail_drops) >= 2:
                d0 = tail_drops[0][1]
                d1 = tail_drops[-1][1]
                for _vw in drop_delta:
                    drop_delta[_vw] = int(d1.get(_vw, 0)) - int(d0.get(_vw, 0))
            saturated_views = [_vw for _vw, dd in drop_delta.items() if dd > 0]
            _drops_str = (','.join(f'{_vw}:{drop_delta[_vw]}'
                                   for _vw in drop_delta if drop_delta[_vw] > 0) or 'none')
            # Service rate over the settled tail (frames completed / s). Recorded
            # for EVERY validated candidate, whatever the commit decision, so
            # _revert_to_best can pick the best-observed placement on exhaustion.
            tail_counts = [(t, c) for (t, c) in count_samples if t >= tail_start]
            svc_rate = None
            if len(tail_counts) >= 2:
                (ct0, cc0), (ct1, cc1) = tail_counts[0], tail_counts[-1]
                if ct1 > ct0:
                    svc_rate = (cc1 - cc0) / (ct1 - ct0)
            self._cand_obs[combo] = {"rate": svc_rate,
                                     "sig": self._placement_signature(combo)}
            elapsed = time.time() - entered_at
            print(f"[Executor] QoS-validate combo={combo} elapsed={elapsed:.1f}s "
                  f"V_postswap={'None' if v is None else f'{v:.3f}'} "
                  f"V_cumulative={'None' if v_report is None else f'{v_report:.3f}'} "
                  f"eps={eps} backlog_slope={slope_str}/s "
                  f"tail_drops={{{_drops_str}}} "
                  f"service_rate={'None' if svc_rate is None else f'{svc_rate:.1f}fps'} "
                  f"(post-T_v)")
            # Is this the last candidate in the search? If a validated candidate
            # fails AND there is nothing after it, the budget is exhausted:
            # revert to the best observed placement instead of default-committing
            # to this (last-ranked) one.
            is_last_cand = (self._index >= len(self._combinations) - 1)
            # Trend-based feasibility test. A high V(t) alone does NOT mean the
            # candidate is infeasible: it may just be draining backlog inherited
            # from earlier (rejected) candidates. We therefore commit when either
            #   (a) V(t) <= eps  -> already drained / feasible, or
            #   (b) backlog is draining (slope < -slope_th) -> mu* > lambda, the
            #       candidate is feasible and will reach V<=eps given time.
            # Otherwise (slope >= -slope_th: rising or flat, i.e. mu~lambda which
            # the fluid model shows diverges) we advance -- conservative.
            #
            # (0) Saturation guard FIRST. Ongoing drops mean the queue is clipped
            # at cap: the slope no longer estimates lambda-mu (growth is truncated
            # to slope~0 + noise), AND V(t) is survivorship-biased (dropped slow
            # frames never complete, so ell_i averages only the fast survivors and
            # can look low). Neither slope nor V is trustworthy -> advance.
            if saturated_views:
                reason = (
                    f"saturated: views {saturated_views} still dropping in the "
                    f"last {self.qos_tv/3.0:.1f}s (queue clipped at cap -> "
                    f"arrival>service); slope/V unreliable, advancing in combo={combo}")
                if is_last_cand:
                    print(f"[Executor] QoS-triggered advance: {reason} "
                          f"(last candidate -> revert-to-best)")
                    self._revert_to_best(combo)
                else:
                    self._qos_advance(reason)
            elif v is not None and v <= eps:
                print(f"[Executor] QoS-trigger: combo={combo} validated "
                      f"(V(t)={v:.3f} <= eps={eps}); committing.")
                self._commit_and_stay(combo)
            elif slope is not None and slope < -slope_th:
                print(f"[Executor] QoS-trigger: combo={combo} validated "
                      f"(backlog draining slope={slope:.2f}/s < -{slope_th}; "
                      f"feasible, V(t)={'None' if v is None else f'{v:.3f}'} still "
                      f"draining); committing.")
                self._commit_and_stay(combo)
            else:
                reason = (
                    f"backlog not draining (slope={slope_str}/s >= -{slope_th}) "
                    f"and V(t)={'None' if v is None else f'{v:.3f}'} > eps={eps} "
                    f"after T_v={self.qos_tv}s in combo={combo}")
                if is_last_cand:
                    print(f"[Executor] QoS-triggered advance: {reason} "
                          f"(last candidate -> revert-to-best)")
                    self._revert_to_best(combo)
                else:
                    self._qos_advance(reason)

        if policy == "v-above":
            check_fn = check_v_above
        elif policy == "validate":
            check_fn = check_validate
        else:
            print(f"[Executor] Unknown combo-trigger policy '{policy}' "
                  f"for combo={combo}; ignoring.")
            return

        self._qos_poll_timer = QTimer(self._viewer if self._viewer else None)
        self._qos_poll_timer.setInterval(poll_ms)
        self._qos_poll_timer.timeout.connect(check_fn)
        self._qos_poll_timer.start()
        print(f"[Executor] QoS-trigger active: combo={combo} policy={policy} "
              f"eps={eps} T_v={self.qos_tv}s")

    def _after_stop(self):
        if not self._running:
            return
        # Stop any QoS poll leftover from the previous combo.
        self._stop_qos_poll()
        self._qos_advance_fired = False

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
    parser.add_argument('--combo-trigger', action='append', default=[],
                        metavar='COMBO=POLICY',
                        help='QoS-driven advancement policy per combo. '
                             'Policies: "v-above" (advance when V(t) > epsilon), '
                             '"validate" (wait T_v, then advance if V(t) > epsilon; '
                             'otherwise commit). Repeatable.')
    parser.add_argument('--qos-trigger-epsilon', type=float, default=1.0,
                        help='Epsilon for QoS-triggered advancement (default 1.0).')
    parser.add_argument('--qos-trigger-tv', type=float, default=3.0,
                        help='Validation window T_v for the "validate" policy '
                             '(default 3.0 s).')
    parser.add_argument('--qos-window-t', type=int, default=3,
                        help='V(t) sliding window (default 3 s).')
    parser.add_argument('--stop-after', type=str, default=None,
                        help='Stop the executor after this combo completes '
                             '(combos after it in the yaml are skipped).')
    parser.add_argument('--qos-slope-threshold', type=float, default=0.5,
                        help='Trend validation: commit a candidate whose backlog '
                             'drains faster than this (frames/s) even if V(t)>eps. '
                             'Flat/rising backlog advances. Default 0.5.')
    parser.add_argument('--background', action='store_true',
                        help='Run generative (llm/vlm) combo entries as isolated '
                             'background processes creating accelerator contention '
                             '(Q3/Q5 mixed set). Default OFF reproduces prior runs.')
    parser.add_argument('--background-max-new-tokens', type=int, default=64,
                        help='max_new_tokens per background generation (default 64).')
    args = parser.parse_args()
    # Env fallback so headless runners can toggle without editing argv.
    _bg_on = bool(getattr(args, 'background', False)) or \
        os.environ.get('FSRR_BACKGROUND', '0') not in ('0', '', 'false', 'False')

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

    # Parse --combo-trigger into {combo_name: policy}
    combo_triggers: dict = {}
    for spec in (args.combo_trigger or []):
        if "=" not in spec:
            print(f"[Main] WARNING: ignoring malformed --combo-trigger '{spec}'")
            continue
        name, _, pol = spec.partition("=")
        pol = pol.strip()
        if pol not in ("v-above", "validate"):
            print(f"[Main] WARNING: unknown --combo-trigger policy '{pol}' "
                  f"for combo '{name}'; must be one of: v-above, validate")
            continue
        combo_triggers[name.strip()] = pol

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
                                        combo_durations=combo_durations,
                                        combo_triggers=combo_triggers,
                                        qos_epsilon=args.qos_trigger_epsilon,
                                        qos_tv=args.qos_trigger_tv,
                                        qos_window_T=args.qos_window_t,
                                        qos_slope_threshold=args.qos_slope_threshold,
                                        background=_bg_on,
                                        background_max_new_tokens=args.background_max_new_tokens)
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
                                    combo_durations=combo_durations,
                                    combo_triggers=combo_triggers,
                                    qos_epsilon=args.qos_trigger_epsilon,
                                    qos_tv=args.qos_trigger_tv,
                                    qos_window_T=args.qos_window_t,
                                    qos_slope_threshold=args.qos_slope_threshold,
                                    background=_bg_on,
                                    background_max_new_tokens=args.background_max_new_tokens)
        # Truncate combos to stop after a specified name, if requested.
        try:
            if args.stop_after and args.stop_after in executor._combinations:
                idx = executor._combinations.index(args.stop_after)
                executor._combinations = executor._combinations[:idx + 1]
                print(f"[Main] --stop-after {args.stop_after}: "
                      f"truncated combos to {executor._combinations}")
        except Exception as e:
            print(f"[Main] --stop-after failed: {e}")
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