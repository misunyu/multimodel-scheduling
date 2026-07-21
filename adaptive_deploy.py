"""
Adaptive deployment manager for hot-swapping model workers when devices change.

This module is ONLY invoked when the use_adaptive_deploy checkbox is checked.
It keeps models on the same device running and gracefully transitions models
whose device changes by starting the new worker in the background and
switching over once it is ready.
"""
import copy
import os
import queue
import yaml
from threading import Event, Thread
from queue import Queue

from model_processors import (
    run_yolo_cpu_process,
    run_resnet_cpu_process,
    run_yolo_gpu_process,
    run_resnet_gpu_process,
    worker_target,
    classify_view,
)


def _output_queue_attr(view_name):
    """Return the attribute name used on UnifiedViewer for the output queue of a view."""
    if view_name in ("view1", "view2"):
        return f"{view_name}_output_queue"
    return f"{view_name}_result_queue"


def _same_device(old_cfg, new_cfg):
    """Return True when model AND execution device are identical."""
    if not old_cfg or not new_cfg:
        return False
    return (old_cfg.get("model", "") == new_cfg.get("model", "")
            and old_cfg.get("execution", "cpu") == new_cfg.get("execution", "cpu"))


def _create_worker_thread(view_name, cfg, frame_queue, output_queue, shutdown_event, ready_event):
    """Create (but do not start) the appropriate worker thread for *cfg*.

    Registry-driven routing (replaces the old "yolov4" substring test); device from
    the schedule's execution token (cpu/gpu/npu). Returns None for a generative
    (llm/vlm — deferred) or unknown model, so the caller can skip the hot-swap.
    """
    model = cfg.get("model", "")
    execution = cfg.get("execution", "cpu")

    target, kind, device = worker_target(model, execution)
    if target is None:
        print(f"[AdaptiveDeploy] No vision worker for {view_name}: model={model} "
              f"execution={execution} (generative/deferred or unknown).")
        return None
    return Thread(
        target=target,
        args=(frame_queue, output_queue, shutdown_event),
        kwargs={"view_name": view_name, "model_name": model, "ready_event": ready_event},
        daemon=True,
    )


class AdaptiveDeployManager:
    """Manages adaptive (hot-swap) deployment transitions on a UnifiedViewer.

    Usage (from unified_viewer.py):
        mgr = AdaptiveDeployManager(viewer)
        mgr.execute(schedule_file, combination_name)
    """

    # Named views whose queues/processes live as attributes on the viewer
    NAMED_VIEWS = ("view1", "view2", "view3", "view4")

    def __init__(self, viewer):
        self.viewer = viewer

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_combination(schedule_file, combination_name):
        """Parse a schedule YAML and return model_settings dict for one combination.

        Returns (settings_dict, views_without_model_set, headless_ids_list).
        This does NOT touch the viewer — it is a pure read operation.
        """
        with open(schedule_file, "r") as f:
            config = yaml.safe_load(f) or {}

        combo_key = combination_name
        if combo_key not in config and config:
            combo_key = next(iter(config))

        settings = {}
        views_without = set()
        headless_ids = []

        if combo_key in config:
            for model_config_name, model_config in (config[combo_key] or {}).items():
                if not isinstance(model_config, dict) or "display" not in model_config:
                    continue
                view_name = model_config.get("display")
                vnorm = str(view_name).strip().lower() if view_name is not None else ""

                if (not vnorm) or (vnorm in {"none", "off", "hidden", "no", "false", "0"}):
                    safe_name = str(model_config_name).replace(" ", "_")
                    hid = f"headless_{safe_name}"
                    headless_ids.append(hid)
                    settings[hid] = {
                        "model": model_config.get("model", ""),
                        "execution": model_config.get("execution", "cpu"),
                        "infps": model_config.get("infps", None),
                    }
                    continue

                if vnorm in {"view1", "view2", "view3", "view4"}:
                    settings[vnorm] = {
                        "model": model_config.get("model", ""),
                        "execution": model_config.get("execution", "cpu"),
                        "infps": model_config.get("infps", None),
                    }

        for vn in ("view1", "view2", "view3", "view4"):
            if vn not in settings:
                views_without.add(vn)
                settings[vn] = {"model": "", "execution": "cpu"}

        return settings, views_without, headless_ids, combo_key

    def execute(self, schedule_file, combination_name):
        """Perform an adaptive combination switch.

        Key difference from the non-adaptive path: we do NOT call
        viewer.initialize_model_settings() which would replace model_settings
        and views_without_model atomically, causing a brief state where
        metrics reads 0. Instead we parse the YAML independently and
        update the viewer's state incrementally, keeping kept-view
        handler references and stats intact throughout.
        """
        v = self.viewer

        # --- 1. Snapshot old state -------------------------------------------
        old_settings = copy.deepcopy(v.model_settings)
        old_yolo_views = set(getattr(v, 'yolo_views', set()))
        old_resnet_views = set(getattr(v, 'resnet_views', set()))

        # --- 2. Parse new settings WITHOUT touching the viewer ---------------
        new_settings, new_views_without, new_headless_ids, combo_key = \
            self._parse_combination(schedule_file, combination_name)

        # --- 3. Per-view comparison & hot-swap -------------------------------
        kept_views = set()
        swapped_views = set()

        for vname in self.NAMED_VIEWS:
            old_cfg = old_settings.get(vname, {})
            new_cfg = new_settings.get(vname, {})

            # View unused in both old and new
            old_unused = not old_cfg.get("model", "")
            new_unused = not new_cfg.get("model", "")
            if old_unused and new_unused:
                kept_views.add(vname)
                continue

            # View newly unused -> stop old worker
            if new_unused and not old_unused:
                self._stop_view_worker(vname)
                swapped_views.add(vname)
                continue

            # View newly used -> start fresh worker
            if old_unused and not new_unused:
                self._start_fresh_view(vname, new_cfg)
                swapped_views.add(vname)
                continue

            if _same_device(old_cfg, new_cfg):
                print(f"[AdaptiveDeploy] {vname}: same device ({new_cfg.get('execution','cpu')}), keeping worker")
                kept_views.add(vname)
            else:
                print(f"[AdaptiveDeploy] {vname}: device changed "
                      f"({old_cfg.get('execution','cpu')} -> {new_cfg.get('execution','cpu')}), hot-swapping")
                self._hot_swap_view(vname, old_cfg, new_cfg)
                swapped_views.add(vname)

        # --- 4. Incrementally update viewer state (kept views stay intact) ---
        # Update model_settings in-place: overwrite entries for swapped views,
        # keep existing entries for kept views so handler references stay valid.
        # For kept views, propagate the new infps so live feeders pick up the
        # rate change (the placement is the same but the input rate may have
        # changed between the two combos).
        for vname in self.NAMED_VIEWS:
            if vname in swapped_views:
                v.model_settings[vname] = new_settings[vname]
            elif vname in kept_views:
                old_cfg = v.model_settings.get(vname, {}) or {}
                new_infps = (new_settings.get(vname, {}) or {}).get("infps", None)
                if new_infps is not None and old_cfg.get("infps") != new_infps:
                    old_cfg["infps"] = new_infps
                    v.model_settings[vname] = old_cfg
        # Update views_without_model
        v.views_without_model = new_views_without
        # Update combination name and schedule label
        v.current_combination = combo_key
        v.schedule_file = schedule_file
        v.requested_combination = combination_name
        try:
            v.info_window.update_schedule_name(f"Current Schedule: {combo_key}")
        except Exception:
            pass
        # Update headless ids for new combination
        v.headless_ids = new_headless_ids
        # Update model_settings for headless entries
        for hid in new_headless_ids:
            if hid in new_settings:
                v.model_settings[hid] = new_settings[hid]

        # --- 5. Handle headless views (stop all old, start new) -----
        self._restart_headless(old_settings, new_settings)

        # --- 6. Update feeder yolo/resnet view-sets --------------------------
        self._update_feeders(old_yolo_views, old_resnet_views, swapped_views, new_settings)

        # --- 7. Push updated input intervals to running feeders --------------
        # Even when every view kept its worker (same placement, only infps
        # changed) we must still re-arm the per-model send intervals so the
        # rate change actually takes effect. Without this, the feeders keep
        # using the previous combo's intervals indefinitely.
        try:
            if getattr(v, 'video_feeder', None):
                v.video_feeder.update_intervals(v.model_settings)
            if getattr(v, 'resnet_feeder', None):
                v.resnet_feeder.update_intervals(v.model_settings)
        except Exception as e:
            print(f"[AdaptiveDeploy] failed to push updated intervals: {e}")

        print(f"[AdaptiveDeploy] Transition complete. kept={kept_views}, swapped={swapped_views}")

    # ------------------------------------------------------------------
    # Internal: hot-swap a single named view
    # ------------------------------------------------------------------
    def _hot_swap_view(self, vname, old_cfg, new_cfg):
        """Start new worker in background; once ready, switch queues and stop old worker."""
        v = self.viewer

        # References to old resources
        old_shutdown = getattr(v, f"{vname}_shutdown_event", None)
        old_process = getattr(v, f"{vname}_process", None)
        old_frame_q = getattr(v, f"{vname}_frame_queue", None)
        old_output_q = getattr(v, _output_queue_attr(vname), None)

        # Create new resources
        new_frame_q = Queue(maxsize=2)
        new_output_q = Queue(maxsize=1)
        new_shutdown = Event()
        ready_event = Event()

        new_process = _create_worker_thread(vname, new_cfg, new_frame_q, new_output_q, new_shutdown, ready_event)
        if new_process is None:
            # Generative/deferred or unknown model: nothing to hot-swap to. Stop the
            # old worker so the view goes idle rather than serving a stale placement.
            self._stop_view_worker(vname)
            return
        new_process.start()

        # Watcher thread: waits for ready, then atomically swaps queues
        def _watcher():
            ready_event.wait()  # blocks until model loaded
            print(f"[AdaptiveDeploy] {vname}: new worker ready, switching queues")

            # 1. Swap feeder input queue (thread-safe via lock)
            self._swap_feeder_queue(vname, new_cfg, new_frame_q)

            # 2. Swap handler output queue (atomic under GIL)
            handler = getattr(v, f"{vname}_handler", None)
            if handler is not None:
                handler.result_queue = new_output_q

            # 3. Stop old worker
            if old_shutdown is not None:
                old_shutdown.set()
            if old_process is not None and old_process.is_alive():
                old_process.join(timeout=3.0)

            # 4. Drain old queues
            for q in (old_frame_q, old_output_q):
                self._drain(q)

            # 5. Install new resources on the viewer
            setattr(v, f"{vname}_frame_queue", new_frame_q)
            setattr(v, _output_queue_attr(vname), new_output_q)
            setattr(v, f"{vname}_shutdown_event", new_shutdown)
            setattr(v, f"{vname}_process", new_process)

            print(f"[AdaptiveDeploy] {vname}: hot-swap complete")

        watcher = Thread(target=_watcher, name=f"adaptive_watcher_{vname}", daemon=True)
        watcher.start()

    # ------------------------------------------------------------------
    # Internal: stop a view worker (view becomes unused)
    # ------------------------------------------------------------------
    def _stop_view_worker(self, vname):
        v = self.viewer
        ev = getattr(v, f"{vname}_shutdown_event", None)
        proc = getattr(v, f"{vname}_process", None)
        if ev is not None:
            ev.set()
        if proc is not None and proc.is_alive():
            proc.join(timeout=3.0)
        # Drain queues
        self._drain(getattr(v, f"{vname}_frame_queue", None))
        self._drain(getattr(v, _output_queue_attr(vname), None))
        print(f"[AdaptiveDeploy] {vname}: stopped (no longer used)")

    # ------------------------------------------------------------------
    # Internal: start a fresh view worker (view newly used)
    # ------------------------------------------------------------------
    def _start_fresh_view(self, vname, cfg):
        v = self.viewer
        frame_q = Queue(maxsize=2)
        output_q = Queue(maxsize=1)
        shutdown_ev = Event()

        setattr(v, f"{vname}_frame_queue", frame_q)
        setattr(v, _output_queue_attr(vname), output_q)
        setattr(v, f"{vname}_shutdown_event", shutdown_ev)

        proc = _create_worker_thread(vname, cfg, frame_q, output_q, shutdown_ev, ready_event=None)
        if proc is None:
            return
        proc.start()
        setattr(v, f"{vname}_process", proc)

        # Update yolo/resnet set
        model = cfg.get("model", "")
        if classify_view(model) == "yolo":
            v.yolo_views.add(vname)
            v.resnet_views.discard(vname)
        else:
            v.resnet_views.add(vname)
            v.yolo_views.discard(vname)

        print(f"[AdaptiveDeploy] {vname}: started fresh ({cfg.get('execution','cpu')})")

    # ------------------------------------------------------------------
    # Internal: restart all headless workers
    # ------------------------------------------------------------------
    def _restart_headless(self, old_settings, new_settings):
        v = self.viewer
        # Stop old headless
        for ev in (getattr(v, 'headless_shutdown_events', {}) or {}).values():
            try:
                ev.set()
            except Exception:
                pass
        for proc in (getattr(v, 'headless_processes', []) or []):
            if proc and proc.is_alive():
                proc.join(timeout=3.0)
        for q in (getattr(v, 'headless_frame_queues', {}) or {}).values():
            self._drain(q)
        for q in (getattr(v, 'headless_output_queues', {}) or {}).values():
            self._drain(q)

        # Reset headless containers
        v.headless_frame_queues = {}
        v.headless_output_queues = {}
        v.headless_shutdown_events = {}
        v.headless_processes = []

        # Start new headless workers (reuse viewer's existing logic for headless in initialize_processes)
        # We only call the headless portion; named views are already handled above.
        for hid in list(getattr(v, 'headless_ids', []) or []):
            if hid not in v.headless_frame_queues:
                v.headless_frame_queues[hid] = Queue(maxsize=2)
            if hid not in v.headless_output_queues:
                v.headless_output_queues[hid] = Queue(maxsize=1)
            if hid not in v.headless_shutdown_events:
                v.headless_shutdown_events[hid] = Event()

            cfg = v.model_settings.get(hid, {})
            proc = _create_worker_thread(
                hid, cfg,
                v.headless_frame_queues[hid],
                v.headless_output_queues[hid],
                v.headless_shutdown_events[hid],
                ready_event=None,
            )
            if proc is None:
                continue
            proc.start()
            v.headless_processes.append(proc)

            model = cfg.get("model", "")
            if classify_view(model) == "yolo":
                v.yolo_views.add(hid)
            else:
                v.resnet_views.add(hid)

    # ------------------------------------------------------------------
    # Internal: update feeders after swap
    # ------------------------------------------------------------------
    def _update_feeders(self, old_yolo, old_resnet, swapped_views, new_settings):
        """Synchronise feeder view-sets and queue maps for swapped views."""
        v = self.viewer

        for vname in swapped_views:
            cfg = new_settings.get(vname, {})
            model = cfg.get("model", "")

            # Update yolo/resnet membership
            if classify_view(model) == "yolo":
                v.yolo_views.add(vname)
                v.resnet_views.discard(vname)
            elif model:
                v.resnet_views.add(vname)
                v.yolo_views.discard(vname)
            else:
                v.yolo_views.discard(vname)
                v.resnet_views.discard(vname)

        # Push updated view_frame_queues into feeders
        if getattr(v, 'video_feeder', None):
            v.video_feeder.yolo_views = set(v.yolo_views)
            v.video_feeder.update_intervals(v.model_settings)
        if getattr(v, 'resnet_feeder', None):
            v.resnet_feeder.resnet_views = set(v.resnet_views)
            v.resnet_feeder.update_intervals(v.model_settings)

    # ------------------------------------------------------------------
    # Internal: swap feeder queue for a view
    # ------------------------------------------------------------------
    def _swap_feeder_queue(self, vname, cfg, new_frame_q):
        v = self.viewer
        model = cfg.get("model", "")
        if classify_view(model) == "yolo":
            feeder = getattr(v, 'video_feeder', None)
        else:
            feeder = getattr(v, 'resnet_feeder', None)
        if feeder is not None:
            feeder.swap_view_queue(vname, new_frame_q)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _drain(q):
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass
