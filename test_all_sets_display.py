#!/usr/bin/env python3
"""Strictly-sequential display test for all 15 trained working sets (CPU-NPU).

Drives the REAL main window (BestDeployFinderApp): it selects a working set, presses
Predict Best / Run Best, flips the "Show all models" checkbox, and stops runs through
the same code the buttons call. The display it launches reports what is on screen
through the DISPLAY_PROBE status lines, which is what every check below reads.

Three phases (PHASES env, default all):

  B2   15 sets x 2 starting modes (measure / exhibition) = 30 runs of HOLD_SECS each.
       Every run switches mode once, at SWITCH_AT seconds -- half the runs from the
       main window's checkbox, half from the display's own toggle. Each run ends by
       pressing the display's Stop All and then terminating the executor.
  BIS  Changing the working set while one is running (Run Best pressed mid-run):
       three changes in measure mode, three in exhibition mode.
  B4   The same set run to completion in each mode, with no switch, to compare what
       was loaded and how much inference completed.

Only ONE set runs at a time: a run is launched, held, stopped and verified gone
(exit code, no process of it left, NPU memory released) before the next starts. A
failing run is still cleaned up before moving on, so a crash cannot poison later runs.

The lead whitelist and the size ranking live in unified_viewer / model_registry; the
expected leads below are derived from unified_viewer.select_leads, never hardcoded.

Run with the runtime interpreter (PYTHON_BIN) on the real display (DISPLAY set), e.g.
  ./test_all_sets_display.sh
Results -> $DISPLAY_TEST_OUT/results.json ; window captures -> .../shots/
"""
import os, sys, json, time, subprocess, shutil, tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Never render the child windows off-screen: this test exists to see them.
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ["DISABLE_VISUALIZATION"] = "0"          # rendering ON
os.environ["DISPLAY_PROBE"] = "1"                  # the display reports its state
os.environ["PYTHONUNBUFFERED"] = "1"

OUT = Path(os.environ.get("DISPLAY_TEST_OUT",
                          os.path.join(tempfile.gettempdir(), "display_test")))
SHOTS = OUT / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

HOLD_SECS   = float(os.environ.get("HOLD_SECS", "30"))    # run length, from launch
SWITCH_AT   = float(os.environ.get("SWITCH_AT", "15"))    # mode switch, from launch
SHOT_BEFORE = float(os.environ.get("SHOT_BEFORE", "12"))
SHOT_AFTER  = float(os.environ.get("SHOT_AFTER", "19"))
DURATION    = int(os.environ.get("SET_DURATION", "40"))   # > HOLD: no cycle restart mid-run
WARMUP      = os.environ.get("WARMUP_SECONDS", "3")       # short warmup so figures show fast
STABILIZE   = float(os.environ.get("STABILIZE_SECS", "2"))
BIS_DWELL   = float(os.environ.get("BIS_DWELL", "15"))    # run time before changing set
B4_DURATION = int(os.environ.get("B4_DURATION", "60"))
B4_WARMUP   = os.environ.get("B4_WARMUP", "20")
PHASES      = [p for p in os.environ.get("PHASES", "B2,BIS,B4").split(",") if p]
ONLY_SETS   = [int(x) for x in os.environ.get("ONLY_SETS", "").split(",") if x]

LAUNCHER = ROOT / "schedule_executor_main.sh"

# Working sets by the names working_sets.yaml gives them (sorted model lists).
NAMED_SETS = {
    "base2": ["llama1b", "qwen2_vl", "resnet50", "yolo11s"],
    "S1":    ["resnet50", "yolo11s"],
    "S4":    ["qwen2_vl", "resnet50", "yolo11s"],
    "S8":    ["llama1b", "mobilenet_v2", "qwen2_vl", "resnet50",
              "yolo11l", "yolo11m", "yolo11s", "yolo11x"],
    "S10":   ["mobilenet_v2", "resnet50", "yolo11l", "yolo11m", "yolo11n", "yolo11s", "yolo11x"],
}


def now_ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def stray_executors():
    """Any schedule_executor / spawn worker anywhere (must be 0 between sets)."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "schedule_executor_main.py"], capture_output=True, text=True).stdout
        return [l for l in out.splitlines() if "pgrep" not in l]
    except Exception:
        return []


def screenshot(path):
    """Root-window screenshot (only meaningful when the session is not locked)."""
    for cmd in (["gnome-screenshot", "-f", str(path)],
                ["import", "-window", "root", str(path)],
                ["scrot", str(path)]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, timeout=15, capture_output=True)
                if path.exists() and path.stat().st_size > 0:
                    return True
            except Exception:
                pass
    return False


def num(text):
    """First number in a label's text ('−29.44' -> -29.44), or None."""
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", str(text).replace("−", "-"))
    return float(m.group(0)) if m else None


class Harness:
    """The main window plus everything it hears from the display it launched."""

    def __init__(self, app, finder):
        from PyQt5.QtCore import QEventLoop
        self._all_events = QEventLoop.AllEvents
        self.app, self.finder = app, finder
        self.statuses, self.events, self.logs, self.saved = [], [], [], []
        finder._status_hook = self.statuses.append
        finder._ctrl_hook = lambda e: self.events.append(dict(e, _t=time.time()))
        orig_log = finder.log

        def log(msg):
            self.logs.append((time.time(), str(msg)))
            # The executor has just written a results entry: read it before a later
            # cycle of the same combination can overwrite the file.
            if "Throughput data saved to" in str(msg):
                try:
                    path = str(msg).split("Throughput data saved to", 1)[1].strip()
                    blob = json.loads(Path(path).read_text())
                    self.saved.append((blob if isinstance(blob, list) else [blob])[-1])
                except Exception as e:
                    print(f"[Test] could not read saved results: {e}")
            orig_log(msg)
        finder.log = log
        self.rows = {tuple(sorted(finder.model_list.models_at(r))): r
                     for r in range(finder.model_list.rowCount())}

    def reset(self):
        del self.statuses[:], self.events[:], self.logs[:], self.saved[:]

    def pump(self, seconds=None, until=None, timeout=60.0):
        """Run the Qt event loop for `seconds`, or until `until()` is true."""
        end = time.time() + (seconds if seconds is not None else timeout)
        while time.time() < end:
            self.app.processEvents(self._all_events, 50)
            if until is not None and until():
                return True
            time.sleep(0.01)
        return bool(until()) if until is not None else True

    def select(self, models):
        row = self.rows[tuple(sorted(models))]
        self.finder.model_tree_view.setCurrentIndex(self.finder.model_list.index(row, 0))

    def last(self):
        return self.statuses[-1] if self.statuses else None

    def event(self, kind, after=0.0, **match):
        for e in self.events:
            if (e.get("evt") == kind and e["_t"] >= after
                    and all(e.get(k) == v for k, v in match.items())):
                return e
        return None

    def grab(self, name):
        """Have the display save what its window shows (works under a locked screen)."""
        path = SHOTS / name
        t = time.time()
        self.finder._send_to_executors({"cmd": "grab", "path": str(path)})
        self.pump(until=lambda: self.event("grabbed", after=t) is not None, timeout=5)
        return path.exists()

    def stop_and_confirm(self, timeout=90.0):
        """Terminate the running executor through the main window's own stop path and
        wait for its confirmation (exit code, no survivors, NPU memory)."""
        if not self.finder._running_executors():
            return None
        self.finder._pending_run = None
        self.finder._stop_executors_then_run()
        self.pump(until=lambda: "t_confirmed" in (self.finder.last_restart or {}), timeout=timeout)
        return dict(self.finder.last_restart or {})

    def log_lines(self, needle):
        return [(t, m) for t, m in self.logs if needle in m]


def expected_cast(models):
    """(lead models in slot order, supporting models) for a set -- from select_leads,
    applied to the same view order the schedule builder produces."""
    from unified_viewer import select_leads
    view_models = [(f"view{i + 1}", m) for i, m in enumerate(models)]
    whitelisted, fill = select_leads(view_models)
    by_view = dict(view_models)
    leads = [by_view[v] for v in whitelisted + fill]
    return leads, [m for v, m in view_models if v not in set(whitelisted + fill)]


def views_of(status):
    return {v["view"]: v for v in status["leads"] + status["support"]}


def analyse_run(h, row, models, start_exh, t_cmd, target, placement):
    """Turn a B2 run's status stream into the B.2 checks. Fills `row` in place."""
    S = [s for s in h.statuses if not s["stopped"]]
    if not S:
        row["error"] = row["error"] or "no status from the display"
        return
    painted = h.event("mode_painted", after=t_cmd, value=target)
    t_painted = painted["t_painted"] if painted else None
    pre = [s for s in S if s["t"] < t_cmd]
    post = [s for s in S if t_painted and s["t"] > t_painted]
    first, last = S[0], S[-1]
    n = len(models)
    exp_leads, exp_support = expected_cast(models)

    # --- 5a. leads: four cells filled (whitelist -> size -> X)
    row["leads"] = [l["model"] for l in last["leads"]]
    row["x_slots"] = last["x_slots"]
    row["leads_ok"] = (row["leads"] == exp_leads and last["x_slots"] == 4 - len(exp_leads))

    # --- 3/4. loaded, inferring
    final = views_of(last)
    row["counts"] = {v["model"]: v["count"] for v in final.values()}
    row["loaded"] = {v["model"]: bool(v["msgs"] > 0 and not v["error"]) for v in final.values()}
    row["load_ok"] = len(final) == n and all(row["loaded"].values())
    idle, allowed_idle = [], []
    for v in final.values():
        if v["count"] <= 0:
            (allowed_idle if (v["model"] == "llama1b" and placement.get("llama1b") == "cpu")
             else idle).append(v["model"])
    row["idle_models"], row["idle_allowed"] = idle, allowed_idle
    row["infer_ok"] = not idle

    # --- 5b/5c. lead strips carry live numbers; lead rendering keeps moving
    # A strip can only change when its view completes something, so how much change
    # to demand depends on how much completed: a vision view (hundreds of results)
    # must show at least two different readings; a generative view that finished only
    # a handful of generations in the run must have gone from the "--" placeholder to
    # a real number. A CPU llama1b that completed nothing legitimately stays at "--".
    strip_ok, render_ok, strip_states = {}, {}, {}
    for lead in last["leads"]:
        view = lead["view"]
        series = [views_of(s).get(view) for s in S]
        series = [x for x in series if x]
        strips = [x["strip"] for x in series]
        with_numbers = [t for t in strips if num(t.split("|")[0]) is not None]
        strip_states[lead["model"]] = {"readings": len(set(with_numbers)),
                                       "completions": lead["count"]}
        if lead["model"] in allowed_idle:
            strip_ok[lead["model"]] = True
        elif lead["count"] >= 5:
            strip_ok[lead["model"]] = len(set(with_numbers)) >= 2
        else:
            strip_ok[lead["model"]] = (bool(with_numbers) and len(set(strips)) >= 2
                                       and num(strips[-1].split("|")[0]) is not None)
        frames = [x["frames"] for x in series]
        render_ok[lead["model"]] = (frames[-1] > frames[0] and len(set(
            x["pixmap_key"] for x in series)) >= 3)
    row["strip_states"] = strip_states
    row["strip_ok"], row["render_ok"] = strip_ok, render_ok
    row["last_strips"] = {l["model"]: l["strip"] for l in last["leads"]}

    # --- 5d/5e. the right column, per mode section
    def section(samples, exh):
        return [s for s in samples if (s["mode"] == "exhibition") == exh]
    meas, exh = section(S, False), section(S, True)
    row["measure_panel_ok"] = bool(meas) and all(s["panel"]["page"] == 0 and not s["panel"]["badge"]
                                                 for s in meas)
    live_meas = [s for s in meas if s["measuring"] and s["panel"]["live"]]
    if live_meas:
        m_last = live_meas[-1]
        row["measure_score"] = m_last["panel"]["score"]
        row["measure_panel_ok"] = (row["measure_panel_ok"] and num(m_last["panel"]["score"]) is not None
                                   and m_last["panel"]["live"]["total_fps"] > 0)
    else:
        row["measure_panel_ok"] = False
    row["exh_panel_ok"] = bool(exh) and all(s["panel"]["page"] == 1 and s["panel"]["badge"] for s in exh)
    row["support_expected"] = len(exp_support)
    if exh:
        e_last = exh[-1]
        vis = [t for t in e_last["support"] if t["visible"]]
        row["support_visible"] = len(vis)
        moved = {}
        for t in e_last["support"]:
            fr = [views_of(s)[t["view"]]["frames"] for s in exh]
            moved[t["model"]] = fr[-1] > fr[0]
        row["support_render"] = moved
        row["compact"] = e_last["panel"]["compact"]
        row["exh_panel_ok"] = (row["exh_panel_ok"] and len(vis) == len(exp_support)
                               and [t["model"] for t in e_last["support"]] == exp_support
                               and all(moved.values()) and num(e_last["panel"]["compact"]) is not None)
    else:
        row["support_visible"] = None
        row["exh_panel_ok"] = False
    # supporting tiles stay hidden and undrawn in measure mode
    row["support_hidden_in_measure"] = all(not t["visible"] and not t["render_enabled"]
                                           for s in meas for t in s["support"])

    # --- 6. the switch
    row["switch_painted"] = bool(painted)
    if painted:
        row["switch_latency_ms"] = round((t_painted - t_cmd) * 1000.0, 1)
        row["switch_internal_ms"] = round((t_painted - painted["t_recv"]) * 1000.0, 1)
    sup = h.event("support_rendering", after=t_cmd)
    if target and exp_support:
        row["support_all_rendering_ms"] = round((sup["t_all"] - t_cmd) * 1000.0, 1) if sup else None
    if pre and post:
        a, b = pre[-1], post[0]
        va, vb, vz = views_of(a), views_of(b), views_of(last)
        # inference never paused: no counter went backwards across the switch, and the
        # total kept climbing in the samples on either side of it
        seq = [s for s in S if s["measuring"]]
        totals = [sum(v["count"] for v in views_of(s).values()) for s in seq]
        backwards = any(y < x for x, y in zip(totals, totals[1:]))
        near = [sum(v["count"] for v in views_of(s).values())
                for s in seq if abs(s["t"] - t_cmd) <= 3.0]
        row["count_before_switch"] = sum(v["count"] for v in va.values())
        row["count_after_switch"] = sum(v["count"] for v in vz.values())
        row["counts_continuous"] = (not backwards and len(near) >= 4 and near[-1] > near[0]
                                    and row["count_after_switch"] > row["count_before_switch"])
        # leads: same widget, same handler, same thread, same worker; frames never reset
        same = all(va[l["view"]][k] == vb[l["view"]][k]
                   for l in a["leads"] for k in ("handler_id", "thread", "pid"))
        same = same and [l["widget_id"] for l in a["leads"]] == [l["widget_id"] for l in b["leads"]]
        same = same and all(l["thread_alive"] and l["alive"] for l in b["leads"])
        frames_ok = True
        for lead in a["leads"]:
            fr = [views_of(s)[lead["view"]]["frames"] for s in S]
            frames_ok = frames_ok and all(y >= x for x, y in zip(fr, fr[1:]))
            around = [views_of(s)[lead["view"]]["frames"] for s in S if abs(s["t"] - t_cmd) <= 3.0]
            frames_ok = frames_ok and len(around) >= 4 and around[-1] > around[0]
        row["leads_untouched"] = bool(same)
        row["leads_continuous"] = bool(frames_ok)
        # only the supporting views' RENDERING changed
        sa, sb = {t["view"]: t for t in a["support"]}, {t["view"]: t for t in b["support"]}
        row["only_support_toggled"] = (
            all(sa[v]["render_enabled"] != target and sb[v]["render_enabled"] == target for v in sa)
            and all(sa[v]["tile_id"] == sb[v]["tile_id"] and sa[v]["handler_id"] == sb[v]["handler_id"]
                    and sa[v]["pid"] == sb[v]["pid"] for v in sa))
        row["left_geom_same"] = (a["left"] == b["left"] and
                                 [l["geom"] for l in a["leads"]] == [l["geom"] for l in b["leads"]])
        row["models_same"] = ({(k, v["pid"]) for k, v in va.items()} == {(k, v["pid"]) for k, v in vb.items()}
                              and all(v["alive"] for v in vb.values()))
        row["display_toggle_after"] = b["panel"]["toggle"]
    row["main_checkbox_after"] = h.finder.show_all_checkbox.isChecked()
    row["sync_ok"] = (row["main_checkbox_after"] == target
                      and row.get("display_toggle_after") == target and last["mode"] ==
                      ("exhibition" if target else "measure"))

    # --- 7. geometry (compared across runs in main)
    row["window"] = last["window"]
    row["left"] = last["left"]
    row["right_w"] = last["right_w"]
    row["lead_geoms"] = [l["geom"] for l in last["leads"]]
    row["lead_sizes"] = sorted({tuple(l["geom"][2:]) for s in S for l in s["leads"]})


def run_b2(h, idx, models, start_exh, toggle_from):
    """One 30 s run: predict, Run Best, switch mode at 15 s, Stop All, terminate."""
    finder = h.finder
    mode = "exhibition" if start_exh else "measure"
    tag = f"set{idx:02d}_{mode}"
    row = {"phase": "B2", "set": idx, "models": models, "n_models": len(models),
           "start_mode": mode, "toggle_from": toggle_from, "error": "",
           "predict_ok": False, "start_ts": None, "end_ts": None}
    h.reset()
    try:
        finder.generated_schedule_path = str(OUT / f"set{idx:02d}.yaml")
        h.select(models)
        finder.show_all_checkbox.setChecked(start_exh)      # nothing running: start mode
        finder.duration_input.setText(str(DURATION))

        # 1) Predict Best (production path; must be in-distribution)
        ok = finder.on_predict_best_clicked()
        row["ood_warnings"] = [m for _, m in h.log_lines("[BUG]")]
        best = (finder._read_predictions() or [{}])[0].get("combination")
        row["best"] = best
        row["predict_ok"] = bool(ok and best and not row["ood_warnings"])
        placement = finder._combo_placement(best) if best else {}
        row["placement"] = placement

        # 2) Run Best
        row["start_ts"] = now_ts()
        t_launch = time.time()
        finder.on_load_execute_best_clicked()
        if not h.pump(until=lambda: h.last() is not None, timeout=40):
            raise RuntimeError("the display never reported (window did not come up)")
        row["window_up_s"] = round(h.last()["t"] - t_launch, 2)
        row["start_mode_ok"] = h.last()["mode"] == mode

        h.pump(until=lambda: time.time() - t_launch >= SHOT_BEFORE, timeout=SHOT_BEFORE + 5)
        row["shot_before"] = h.grab(f"{tag}_1before.png")

        # 3) the switch, at SWITCH_AT
        h.pump(until=lambda: time.time() - t_launch >= SWITCH_AT, timeout=SWITCH_AT + 5)
        target = not start_exh
        t_cmd = time.time()
        if toggle_from == "main":
            finder.show_all_checkbox.setChecked(target)
        else:
            finder._send_to_executors({"cmd": "click_show_all_toggle"})
        h.pump(until=lambda: h.event("mode_painted", after=t_cmd, value=target) is not None, timeout=5)

        h.pump(until=lambda: time.time() - t_launch >= SHOT_AFTER, timeout=SHOT_AFTER + 5)
        row["shot_after"] = h.grab(f"{tag}_2after.png")
        h.pump(until=lambda: time.time() - t_launch >= HOLD_SECS, timeout=HOLD_SECS + 5)
        row["executor_alive_at_end"] = bool(finder._running_executors())

        analyse_run(h, row, models, start_exh, t_cmd, target, placement)

        # 8) Stop All (the display's button), then terminate the executor
        t_stop = time.time()
        finder._send_to_executors({"cmd": "click_stop_all"})
        stopped = h.pump(until=lambda: h.event("stopped", after=t_stop) is not None, timeout=60)
        ev = h.event("stopped", after=t_stop)
        row["stop_all_s"] = round(ev["_t"] - t_stop, 1) if ev else None
        row["stop_all_workers_alive"] = ev.get("workers_alive") if ev else None
        row["stop_all_ok"] = bool(stopped and ev and ev.get("workers_alive") == 0)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        # cleanup even on failure: make sure NOTHING from this run survives
        trace = h.stop_and_confirm() or {}
        row["end_ts"] = now_ts()
        row["exit_codes"] = trace.get("exit_codes")
        row["forced_kill"] = trace.get("killed")
        row["survivors"] = len(trace.get("survivors") or [])
        row["npu_used_mb_after"] = trace.get("npu_used_mb")
        row["exit_ok"] = (trace.get("exit_codes") == [0] and not trace.get("killed")
                          and not trace.get("survivors"))
        err = [m for _, m in h.logs if "[Exec][stderr]" in m]
        row["stderr_tracebacks"] = sum(1 for m in err if "Traceback" in m)
        row["fatal_lines"] = [m[:160] for _, m in h.logs if "FATAL" in m or "view error" in m][:5]
    checks = ["predict_ok", "start_mode_ok", "load_ok", "infer_ok", "leads_ok", "measure_panel_ok",
              "exh_panel_ok", "support_hidden_in_measure", "switch_painted", "counts_continuous",
              "leads_untouched", "leads_continuous", "only_support_toggled", "left_geom_same",
              "models_same", "sync_ok", "stop_all_ok", "exit_ok"]
    failed = [c for c in checks if not row.get(c)]
    failed += [f"strip:{m}" for m, ok in (row.get("strip_ok") or {"?": False}).items() if not ok]
    failed += [f"render:{m}" for m, ok in (row.get("render_ok") or {"?": False}).items() if not ok]
    if not (row.get("switch_latency_ms") is not None and row["switch_latency_ms"] < 1000):
        failed.append("switch_latency")
    if row["stderr_tracebacks"]:
        failed.append("stderr_traceback")
    row["failed_checks"] = failed
    row["ok"] = not failed and not row["error"]
    return row


def run_bis(h, exhibition, names):
    """Change the working set mid-run: names[0] is started, then each later set is
    selected and Run Best pressed while the previous one is still running."""
    import psutil
    finder = h.finder
    mode = "exhibition" if exhibition else "measure"
    rows = []
    h.reset()
    finder.generated_schedule_path = str(OUT / f"bis_{mode}.yaml")
    finder.show_all_checkbox.setChecked(exhibition)
    finder.duration_input.setText(str(DURATION))
    h.select(NAMED_SETS[names[0]])
    finder.on_load_execute_best_clicked()           # predicts for the set, then launches
    if not h.pump(until=lambda: h.last() is not None, timeout=60):
        return [{"phase": "BIS", "mode": mode, "error": "initial set never came up", "ok": False}]

    for prev_name, name in zip(names, names[1:]):
        models = NAMED_SETS[name]
        row = {"phase": "BIS", "mode": mode, "from": prev_name, "to": name,
               "n_models": len(models), "error": ""}
        try:
            t_up = h.last()["t"]
            h.pump(until=lambda: time.time() - t_up >= BIS_DWELL, timeout=BIS_DWELL + 5)
            old = h.last()
            old_pid = old["pid"]
            old_worker_pids = [v["pid"] for v in views_of(old).values()]
            row["old_total_count"] = sum(v["count"] for v in views_of(old).values())
            row["old_score"] = old["panel"]["score"]
            row["old_window"], row["old_left"] = old["window"], old["left"]
            row["old_lead_geoms"] = [l["geom"] for l in old["leads"]]

            # the trigger: pick another set, press Run Best while the old one runs
            n_status = len(h.statuses)
            t_press = time.time()
            h.select(models)
            finder.on_load_execute_best_clicked()
            h.pump(until=lambda: (finder.last_restart or {}).get("t_request", 0) >= t_press
                   and ("t_launch" in finder.last_restart or finder.last_restart.get("ok") is False),
                   timeout=120)
            tr = dict(finder.last_restart or {})
            row["stop_s"] = round(tr.get("t_confirmed", 0) - tr.get("t_request", 0), 1)
            row["exit_codes"], row["forced_kill"] = tr.get("exit_codes"), tr.get("killed")
            row["survivors"] = len(tr.get("survivors") or [])
            row["npu_used_mb"] = tr.get("npu_used_mb")
            row["dram_before_stop_mb"] = round(tr.get("dram_used_before_stop_mb", 0))
            row["dram_after_stop_mb"] = round(tr.get("dram_used_after_stop_mb", 0))
            # 1) every process of the old set is gone
            row["old_pids_alive_at_launch"] = [p for p in [old_pid] + old_worker_pids
                                               if p and psutil.pid_exists(p)
                                               and psutil.Process(p).status() != psutil.STATUS_ZOMBIE]
            row["all_stopped"] = (tr.get("ok") is True and not row["survivors"]
                                  and not row["old_pids_alive_at_launch"] and row["npu_used_mb"] in (0, None))
            # 2) confirmed BEFORE the new set was launched / its first worker started
            row["t_confirmed"], row["t_launch"] = tr.get("t_confirmed"), tr.get("t_launch")

            if not h.pump(until=lambda: any(s["pid"] != old_pid for s in h.statuses[n_status:]),
                          timeout=60):
                raise RuntimeError("the new set's display never reported")
            new0 = next(s for s in h.statuses[n_status:] if s["pid"] != old_pid)
            # (the new set's workers are started before its display first reports)
            starts = [t for t, m in h.log_lines("[UnifiedViewer] Starting view") if t >= t_press]
            row["t_first_worker_start"] = min(starts) if starts else None
            row["no_overlap"] = bool(tr.get("t_confirmed") and tr.get("t_launch") and starts
                                     and tr["t_confirmed"] <= tr["t_launch"]
                                     and min(starts) > tr["t_confirmed"])
            row["gap_s"] = (round(min(starts) - tr["t_confirmed"], 2) if starts and tr.get("t_confirmed")
                            else None)
            exp_leads, exp_support = expected_cast(models)
            # 3) started in the checkbox's mode
            row["mode_at_launch"] = "exhibition" if tr.get("mode_at_launch") else "measure"
            row["mode_ok"] = (new0["mode"] == mode and row["mode_at_launch"] == mode)
            # 5) nothing carried over from the old set
            fresh = views_of(new0)
            row["new_first_counts"] = sum(v["count"] for v in fresh.values())
            row["new_first_score"] = new0["panel"]["score"]
            row["new_first_strips"] = [l["strip"] for l in new0["leads"]]
            row["reset_ok"] = (row["new_first_counts"] == 0
                               and all(num(s.split("|")[0]) is None for s in row["new_first_strips"])
                               and (not new0["panel"]["live"] or new0["panel"]["live"]["total_fps"] == 0)
                               and {v["pid"] for v in fresh.values()}.isdisjoint(old_worker_pids))

            t_new = new0["t"]
            h.pump(until=lambda: time.time() - t_new >= BIS_DWELL - 2, timeout=BIS_DWELL + 5)
            row["shot"] = h.grab(f"bis_{mode}_{prev_name}_to_{name}.png")
            cur = h.last()
            # 4) leads recast for the new set
            row["leads"] = [l["model"] for l in cur["leads"]]
            row["x_slots"] = cur["x_slots"]
            row["leads_ok"] = row["leads"] == exp_leads and cur["x_slots"] == 4 - len(exp_leads)
            vis = [t for t in cur["support"] if t["visible"]]
            row["support_visible"], row["support_expected"] = len(vis), len(exp_support)
            if exhibition:
                frames0 = {t["view"]: t["frames"] for t in new0["support"]}
                row["mode_ok"] = (row["mode_ok"] and len(vis) == len(exp_support) and
                                  all(t["frames"] > frames0.get(t["view"], 0) for t in cur["support"]))
            else:
                row["mode_ok"] = row["mode_ok"] and not vis
            # 6) window and the 2x2 where they were
            row["window"], row["left"] = cur["window"], cur["left"]
            row["geom_ok"] = (cur["window"] == row["old_window"] and cur["left"] == row["old_left"]
                              and sorted({tuple(l["geom"][2:]) for l in cur["leads"]})
                              == sorted({tuple(g[2:]) for g in row["old_lead_geoms"]}))
            # 7) the new set is really inferring
            placement = finder._combo_placement((finder._read_predictions() or [{}])[0].get("combination"))
            counts = {v["model"]: v["count"] for v in views_of(cur).values()}
            row["counts"] = counts
            row["infer_ok"] = all(c > 0 or (m == "llama1b" and placement.get(m) == "cpu")
                                  for m, c in counts.items())
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        checks = ["all_stopped", "no_overlap", "mode_ok", "leads_ok", "reset_ok", "geom_ok", "infer_ok"]
        row["failed_checks"] = [c for c in checks if not row.get(c)]
        row["ok"] = not row["failed_checks"] and not row["error"]
        rows.append(row)
        print(f"[BIS {mode}] {prev_name} -> {name}: ok={row['ok']} failed={row['failed_checks']} "
              f"stop={row.get('stop_s')}s gap={row.get('gap_s')}s err='{row['error']}'", flush=True)
    h.stop_and_confirm()
    return rows


def run_b4(h, name, exhibition):
    """One full, unswitched run in one mode; returns what the executor saved."""
    finder = h.finder
    mode = "exhibition" if exhibition else "measure"
    models = NAMED_SETS[name]
    row = {"phase": "B4", "set": name, "mode": mode, "error": ""}
    h.reset()
    os.environ["WARMUP_SECONDS"] = B4_WARMUP
    try:
        finder.generated_schedule_path = str(OUT / f"b4_{name}.yaml")
        finder.show_all_checkbox.setChecked(exhibition)
        finder.duration_input.setText(str(B4_DURATION))
        h.select(models)
        finder.on_load_execute_best_clicked()
        total = B4_DURATION + float(B4_WARMUP)
        if not h.pump(until=lambda: bool(h.saved), timeout=total + 60):
            raise RuntimeError("the run never saved its results")
        entry = h.saved[0]
        row["combination"] = entry.get("combination")
        row["window_sec"] = entry.get("window_sec")
        row["display_mode"] = entry.get("display_mode")
        row["total"] = entry.get("total")
        row["models"] = {v["model"]: {k: v.get(k) for k in (
            "execution", "inference_count", "throughput_fps", "tokens_per_s",
            "avg_inference_time_ms", "deadline_miss_rate")} for v in entry.get("models", {}).values()}
        row["loaded_models"] = sorted(row["models"])
        row["warmup_line"] = next((m for _, m in h.log_lines("Starting execution with duration")), "")
        row["ok"] = True
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["ok"] = False
    finally:
        h.stop_and_confirm()
        os.environ["WARMUP_SECONDS"] = str(WARMUP)
    return row


def main():
    from PyQt5.QtWidgets import QApplication
    from best_deploy_finder_executor import BestDeployFinderApp as BestDeployFinder
    from unified_viewer import LEAD_WHITELIST

    if not LAUNCHER.exists():
        print(f"[FATAL] launcher not found: {LAUNCHER}"); sys.exit(2)
    os.environ["WARMUP_SECONDS"] = str(WARMUP)

    # The GUI rewrites these two as it works; put the repo's copies back afterwards.
    keep = {p: (ROOT / p).read_bytes() for p in ("predictions.csv",) if (ROOT / p).exists()}

    app = QApplication(sys.argv)
    finder = BestDeployFinder()
    if hasattr(finder, "device_config_combo"):
        finder.device_config_combo.setCurrentText("CPU-NPU")  # -> deploy_cpu_npu, [cpu,npu]
    finder.show()
    pred_prefix = finder.prediction_model_input.text()
    h = Harness(app, finder)
    print(f"[Test] predictor prefix = {pred_prefix}")
    print(f"[Test] lead whitelist = {list(LEAD_WHITELIST)}")
    h.pump(0.5)
    finder.grab().save(str(SHOTS / "main_window.png"))

    sets = finder.trained_working_sets(pred_prefix)   # 15 lists, coverage order
    print(f"[Test] {len(sets)} trained working sets (CPU-NPU). HOLD={HOLD_SECS}s "
          f"SWITCH_AT={SWITCH_AT}s WARMUP={WARMUP}s phases={PHASES}")

    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text()) if (ONLY_SETS or len(PHASES) < 3) and results_path.exists() else {}
    results.setdefault("B2", []); results.setdefault("BIS", []); results.setdefault("B4", [])

    def save():
        results_path.write_text(json.dumps(results, indent=1))

    try:
        if "B2" in PHASES:
            for idx, models in enumerate(sets, 1):
                if ONLY_SETS and idx not in ONLY_SETS:
                    continue
                for k, start_exh in enumerate((False, True)):
                    # alternate who asks for the switch, so both directions are driven
                    # from both sides across the 30 runs
                    toggle_from = "main" if (idx + k) % 2 == 0 else "display"
                    print(f"\n===== SET {idx}/{len(sets)} [{'exhibition' if start_exh else 'measure'} "
                          f"start, switch from {toggle_from}]: {','.join(models)} =====", flush=True)
                    stray_before = stray_executors()
                    if stray_before:
                        print(f"[Test][WARN] stray executor(s) BEFORE set {idx}: {stray_before}")
                    row = run_b2(h, idx, models, start_exh, toggle_from)
                    row["stray_before"] = len(stray_before)
                    h.pump(1.0)
                    row["stray_after"] = len(stray_executors())
                    results["B2"] = [r for r in results["B2"]
                                     if not (r["set"] == idx and r["start_mode"] == row["start_mode"])]
                    results["B2"].append(row)
                    print(f"[Set {idx} {row['start_mode']}] ok={row['ok']} failed={row['failed_checks']} "
                          f"leads={row.get('leads')}+{row.get('x_slots')}X support={row.get('support_visible')}"
                          f"/{row.get('support_expected')} switch={row.get('switch_latency_ms')}ms "
                          f"exit={row.get('exit_codes')} survivors={row.get('survivors')} "
                          f"err='{row['error']}'", flush=True)
                    save()
                    h.pump(STABILIZE)       # let NPU/OS resources settle

        if "BIS" in PHASES:
            results["BIS"] = []
            for exhibition, names in ((False, ["S4", "base2", "S8", "S1"]),
                                      (True, ["S4", "S8", "base2", "S10"])):
                print(f"\n===== BIS [{'exhibition' if exhibition else 'measure'}]: "
                      f"{' -> '.join(names)} =====", flush=True)
                results["BIS"] += run_bis(h, exhibition, names)
                save()
                h.pump(STABILIZE)

        if "B4" in PHASES:
            results["B4"] = []
            for name in ("base2", "S8"):
                for exhibition in (False, True):
                    print(f"\n===== B4 {name} [{'exhibition' if exhibition else 'measure'}], "
                          f"{B4_DURATION}s + {B4_WARMUP}s warmup, no switch =====", flush=True)
                    row = run_b4(h, name, exhibition)
                    results["B4"].append(row)
                    print(f"[B4 {name} {row['mode']}] ok={row['ok']} total={row.get('total')} "
                          f"display_mode={row.get('display_mode')} err='{row['error']}'", flush=True)
                    save()
                    h.pump(STABILIZE)
    finally:
        h.stop_and_confirm()
        for p, data in keep.items():
            (ROOT / p).write_bytes(data)

    b2 = sorted(results["B2"], key=lambda r: (r["set"], r["start_mode"] != "measure"))
    results["B2"] = b2
    ok = sum(1 for r in b2 if r["ok"])
    print(f"\n[Test] B2: {ok}/{len(b2)} runs fully OK.")
    for r in b2:
        if not r["ok"]:
            print(f"   FAIL set {r['set']} {r['start_mode']}: {r['failed_checks']} {r['error']}")
    # 7) the lead tiles are the same size in every run, at every sample
    sizes = sorted({tuple(s) for r in b2 for s in r.get("lead_sizes", [])})
    lefts = sorted({tuple(r["left"]) for r in b2 if r.get("left")})
    results["summary"] = {"b2_ok": ok, "b2_runs": len(b2), "lead_tile_sizes": sizes, "left_geoms": lefts,
                          "bis_ok": sum(1 for r in results["BIS"] if r.get("ok")),
                          "bis_runs": len(results["BIS"]),
                          "stray_between_runs": sum(r.get("stray_after", 0) for r in b2)}
    print(f"[Test] lead tile sizes seen across all runs: {sizes} (must be exactly one)")
    print(f"[Test] BIS: {results['summary']['bis_ok']}/{len(results['BIS'])} set changes OK.")
    # overlap check: end_ts[i] < start_ts[i+1]
    overlaps = sum(1 for a, b in zip(b2, b2[1:])
                   if a.get("end_ts") and b.get("start_ts") and a["end_ts"] >= b["start_ts"]
                   and (a["set"], a["start_mode"]) < (b["set"], b["start_mode"]) and not ONLY_SETS)
    results["summary"]["overlaps"] = overlaps
    print(f"[Test] sequential overlap violations: {overlaps} (must be 0)")
    save()
    print(f"[Test] results -> {results_path}")
    print("TEST_ALL_SETS_DONE")
    os._exit(0)


if __name__ == "__main__":
    main()
