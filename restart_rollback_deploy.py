"""
Restart-rollback deployment manager (Mode 4).

Combines two existing pieces:
  - Mode 0 (stop-and-restart): every placement transition tears down the
    current workers and reloads the new placement from scratch.
  - Mode 2 (reactive: validation + rollback): after a transition the
    manager monitors V(t); if the new placement fails to improve QoS, the
    previous placement is restored.

Mode 2 (BoundGuard) does the rollback via an *adaptive hot-swap* so
in-flight requests can complete on the previous placement. This module is
the stop-and-restart counterpart: the rollback also goes through a full
worker stop+reload cycle, which is the worst-case behaviour we use as a
baseline in the four-strategy comparison ("Stop-and-restart (+Rollback)").

Public entry point:
    RestartRollbackDeployManager(viewer).execute(
        schedule_file, combination_name,
        prev_combo_name=..., prev_model_set=..., prev_vscore=...,
        run_duration=...,
    )

The manager assumes that the *forward* transition into ``combination_name``
has already been performed by the executor's standard mode-0 path
(``UnifiedViewer.update_combination`` followed by ``start_execution``).
This module only adds the post-transition validation thread and the
rollback restart that may follow.
"""
import os
import time
from threading import Thread

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal


class _RollbackDispatcher(QObject):
    """Helper QObject that lets the monitor thread schedule the rollback
    restart on the Qt main thread via Qt's signal/slot mechanism.

    PyQt5's ``QTimer.singleShot`` does not reliably dispatch a callable to
    the main thread when called from a worker thread (the timer is created
    in the calling thread which has no event loop). The canonical solution
    is to emit a signal whose slot is connected with ``Qt.QueuedConnection``
    so the slot is queued for execution in the thread that owns the
    QObject — which here is the main thread, since this dispatcher is
    instantiated from the executor.
    """

    rollback_signal = pyqtSignal(str, str, int)

    def __init__(self, mgr, parent=None):
        super().__init__(parent)
        self._mgr = mgr
        self.rollback_signal.connect(self._handle, Qt.QueuedConnection)

    def _handle(self, schedule_file, prev_combo, run_duration):
        # Runs on the main thread.
        self._mgr._do_rollback_restart(schedule_file, prev_combo, run_duration)

# Reuse the per-tick v(t) accumulator that already lives in reactive_deploy.
# We deliberately keep its sliding-window state in a *separate* attribute on
# the viewer (``_v_history_mode4``) so a mode-4 run does not pollute or get
# polluted by a mode-2 run that might share the same process.


def _collect_vscore_mode4(viewer, window_T: int = 5) -> float:
    """V(t) sliding-window collector for mode 4.

    Mirrors ``reactive_deploy._collect_vscore`` exactly but stores its
    history under ``viewer._v_history_mode4`` so the two managers can run
    independently in the same process if anyone ever tries to combine
    them.
    """
    views_without = getattr(viewer, "views_without_model", set())
    sched = [v for v in ("view1", "view2", "view3", "view4") if v not in views_without]

    v_sum = 0.0
    n_active = 0
    for vname in sched:
        handler = getattr(viewer, f"{vname}_handler", None)
        if handler is None:
            continue
        infer_ms = float(getattr(handler, "avg_infer_time", 0.0) or 0.0)
        wait_ms = float(getattr(handler, "avg_wait_ms", 0.0) or 0.0)
        li = infer_ms + wait_ms
        if li <= 0:
            # Cold-starting view (no measurement yet) — skip so v(t) is not
            # diluted toward zero by views that have not produced data.
            continue
        ms = getattr(handler, "model_settings", None) or {}
        infps = float((ms.get(vname) or {}).get("infps", 10.0) or 10.0)
        l_slo = 1000.0 / infps if infps > 0 else 100.0
        v_sum += max(0.0, (li / l_slo) - 1.0)
        n_active += 1

    v_t = (v_sum / n_active) if n_active > 0 else 0.0

    hist = getattr(viewer, "_v_history_mode4", [])
    hist.append(v_t)
    if len(hist) > window_T:
        hist[:] = hist[-window_T:]
    viewer._v_history_mode4 = hist

    return sum(hist) / len(hist) if hist else 0.0


class RestartRollbackDeployManager:
    """Mode 4: stop-and-restart transitions + validation/rollback.

    Compared to ReactiveDeployManager (mode 2):
      - Forward transition is *not* a hot-swap; the executor stops the
        current workers and starts new ones (= mode 0 path). The manager
        is invoked AFTER ``start_execution`` so it can monitor the
        already-restarted workers.
      - Rollback also goes through a full ``stop_execution`` →
        ``initialize_model_settings`` → ``initialize_state_variables`` →
        ``start_execution`` cycle, dispatched onto the Qt main thread via
        ``QTimer.singleShot``.

    The manager only triggers a rollback when:
      (a) the model set is unchanged from the previous combo (so an exact
          revert to ``prev_combo_name`` is meaningful), AND
      (b) the stable post-transition V(t) is at least ``ROLLBACK_DELTA``
          higher than the recorded ``prev_vscore``.

    If the model set changed, the manager logs and exits without action —
    fallback heuristics are out of scope for this baseline (the user
    explicitly asked for a stop-and-restart counterpart of mode 2's
    rollback path, not its fallback heuristic).
    """

    STABILISATION_SEC = 5     # seconds to wait before measuring V(t)
    WINDOW_T = 5              # length of the V(t) sliding window (seconds)
    ROLLBACK_DELTA = 5.0      # V(t) increase threshold for rollback
    RESTART_GAP_MS = 250      # delay between stop and start_execution on rollback

    def __init__(self, viewer):
        self.viewer = viewer
        # Created on the Qt main thread (the executor calls this from
        # _run_next which runs in the main thread). Used by the monitor
        # thread to safely schedule the rollback restart on the main thread.
        self._dispatcher = _RollbackDispatcher(self)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def execute(self, schedule_file, combination_name,
                prev_combo_name=None, prev_model_set=None, prev_vscore=None,
                run_duration=None):
        """Spawn a background validation thread for the just-restarted combo.

        IMPORTANT: the executor must have already done a stop-and-restart
        for ``combination_name`` before calling this. The manager does NOT
        perform the forward transition itself.
        """
        v = self.viewer
        v._v_history_mode4 = []   # fresh sliding window for V(t)

        # Determine current model set from viewer settings
        new_model_set = set()
        for vname in ("view1", "view2", "view3", "view4"):
            m = (v.model_settings or {}).get(vname, {}).get("model", "")
            if m:
                new_model_set.add(m)

        same_models = (prev_model_set is not None and new_model_set == prev_model_set)

        print(f"[RestartRollbackDeployManager] Forward transition complete: {combination_name} "
              f"(same_models={same_models}, prev_combo={prev_combo_name}, prev_vscore={prev_vscore})")

        monitor = Thread(
            target=self._monitor_and_react,
            args=(schedule_file, combination_name,
                  prev_combo_name, same_models, prev_vscore, run_duration),
            name="restart_rollback_monitor",
            daemon=True,
        )
        monitor.start()

    # ------------------------------------------------------------------
    # Monitor thread
    # ------------------------------------------------------------------
    def _monitor_and_react(self, schedule_file, new_combo, prev_combo,
                           same_models, prev_vscore, run_duration):
        v = self.viewer
        print(f"[RestartRollbackDeployManager] Monitoring: waiting "
              f"{self.STABILISATION_SEC}s for stabilisation...")

        for _ in range(self.STABILISATION_SEC):
            if not getattr(v, "_run_active", False):
                print("[RestartRollbackDeployManager] Run stopped during stabilisation. "
                      "Aborting monitor.")
                return
            time.sleep(1.0)

        if not getattr(v, "_run_active", False):
            return

        new_vscore = _collect_vscore_mode4(v, window_T=self.WINDOW_T)
        print(f"[RestartRollbackDeployManager] Stable V(t) = {new_vscore:.4f} "
              f"(prev={prev_vscore}, same_models={same_models})")

        # Decision: same model set + V(t) jumped → rollback
        if same_models and prev_combo and prev_vscore is not None:
            try:
                delta = float(new_vscore) - float(prev_vscore)
            except (TypeError, ValueError):
                delta = 0.0
            if delta >= self.ROLLBACK_DELTA:
                print(f"[RestartRollbackDeployManager] ROLLBACK: V(t) rose by "
                      f"{delta:.2f} >= {self.ROLLBACK_DELTA}. Reverting to {prev_combo}")
                self._dispatch_rollback_restart(schedule_file, prev_combo, run_duration)
                return
            print(f"[RestartRollbackDeployManager] V(t) delta={delta:.2f} < "
                  f"{self.ROLLBACK_DELTA}. Keeping {new_combo}.")
            return

        # Different model set or no baseline available → no rollback
        if not same_models:
            print(f"[RestartRollbackDeployManager] Model set changed. No rollback "
                  f"(fallback heuristic is out of scope for mode 4).")
            return

        print(f"[RestartRollbackDeployManager] No prev_vscore baseline; skipping rollback decision.")

    # ------------------------------------------------------------------
    # Rollback dispatch (background thread → Qt main thread)
    # ------------------------------------------------------------------
    def _dispatch_rollback_restart(self, schedule_file, prev_combo, run_duration):
        try:
            self._dispatcher.rollback_signal.emit(
                str(schedule_file), str(prev_combo),
                int(run_duration) if run_duration else 30,
            )
        except Exception as e:
            print(f"[RestartRollbackDeployManager] Failed to dispatch rollback restart: {e}")

    def _do_rollback_restart(self, schedule_file, prev_combo, run_duration):
        """Run on the Qt main thread: stop workers, re-init, restart with prev_combo."""
        v = self.viewer
        if not getattr(v, "_run_active", False):
            print("[RestartRollbackDeployManager] Run no longer active at rollback time. Skipping.")
            return

        print(f"[RestartRollbackDeployManager] Rollback restart -> {prev_combo} "
              f"(schedule={os.path.basename(schedule_file)})")

        try:
            v.cancel_timed_shutdown()
        except Exception:
            pass

        try:
            v.stop_execution()
        except Exception as e:
            print(f"[RestartRollbackDeployManager] stop_execution error: {e}")

        try:
            v.schedule_file = schedule_file
            v.requested_combination = prev_combo
            v.initialize_model_settings(schedule_file, prev_combo)
            v.initialize_state_variables()
        except Exception as e:
            print(f"[RestartRollbackDeployManager] re-init error: {e}")
            return

        # Small gap so the worker shutdown can complete before we restart.
        # Already on the main thread, so QTimer.singleShot is safe here.
        try:
            QTimer.singleShot(
                self.RESTART_GAP_MS,
                lambda: self._restart_after_gap(run_duration),
            )
        except Exception as e:
            print(f"[RestartRollbackDeployManager] Failed to schedule restart: {e}")

    def _restart_after_gap(self, run_duration):
        v = self.viewer
        try:
            v.start_execution(int(run_duration) if run_duration else 30)
        except Exception as e:
            print(f"[RestartRollbackDeployManager] start_execution after rollback failed: {e}")
