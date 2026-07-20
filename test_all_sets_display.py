#!/usr/bin/env python3
"""Strictly-sequential display smoke test for all 15 trained working sets (CPU-NPU).

For each working set the predictor was trained on, this:
  1. builds the placement schedule (production path: BestDeployFinder methods),
  2. predicts the best combination with deploy_cpu_npu (must pass with no OOD warning),
  3. launches the real execution window (schedule_executor_main.sh) for that combo,
  4. holds it ~HOLD_SECS so the fixed 2x2 grid (M model tiles + (4-M) X) is on screen,
  5. terminates it CLEANLY (SIGTERM to the executor -> shutdown_all disposes the NPU
     models and reaps the view workers), waits for full exit, and
  6. verifies nothing is left behind (exit code, no lingering group/executor process)
     before a short stabilization pause and the NEXT set.

Only ONE set runs at a time: a set is fully launched, held, terminated and verified
gone before the next is started. A failing set is still cleaned up before moving on,
so a crash cannot leak a zombie that poisons later sets.

The visualization whitelist lives in ONE place (unified_viewer.VISUALIZABLE_MODELS);
the expected tile/X counts below are derived from it, never hardcoded per set.

Run with the runtime interpreter (PYTHON_BIN) on the real display (DISPLAY set), e.g.
  ./test_all_sets_display.sh
Results -> <SCRATCH>/display_test/results.json ; screenshots -> .../shot_setNN.png
"""
import os, sys, json, time, signal, subprocess, shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Never render the child windows off-screen: this test exists to see them.
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ["DISABLE_VISUALIZATION"] = "0"          # rendering ON

OUT = Path(os.environ.get("DISPLAY_TEST_OUT",
      "/tmp/claude-1001/-home-msyu-PycharmProjects-multimodel-scheduling-mobilint/"
      "6f0af394-5a71-4f54-bead-adacfbd8c6b9/scratchpad/display_test"))
OUT.mkdir(parents=True, exist_ok=True)

HOLD_SECS   = int(os.environ.get("HOLD_SECS", "20"))    # window on screen per set
DURATION    = int(os.environ.get("SET_DURATION", "20"))  # --duration handed to executor
WARMUP      = os.environ.get("WARMUP_SECONDS", "3")      # short warmup so frames show fast
STABILIZE   = float(os.environ.get("STABILIZE_SECS", "3"))
TERM_WAIT   = float(os.environ.get("TERM_WAIT", "20"))   # grace before SIGKILL backstop
SHOT_AT     = max(2, HOLD_SECS - 4)                       # screenshot near end of hold

LAUNCHER = ROOT / "schedule_executor_main.sh"


def now_ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def group_alive(pgid):
    """PIDs still alive in the executor's process group (0 => nothing left)."""
    try:
        out = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def stray_executors():
    """Any schedule_executor / spawn worker anywhere (must be 0 between sets)."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "schedule_executor_main.py"], capture_output=True, text=True).stdout
        return [l for l in out.splitlines() if "pgrep" not in l]
    except Exception:
        return []


def screenshot(path):
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


def parse_grid_line(log_path):
    """Read the child's '[UnifiedViewer] Display grid fixed 2x2: M model tile(s) + K X
    slot(s).' line -> (filled, x). None if never printed (window never laid out)."""
    try:
        txt = Path(log_path).read_text(errors="replace")
    except Exception:
        return None
    marker = "Display grid fixed 2x2:"
    for line in reversed(txt.splitlines()):
        if marker in line:
            try:
                seg = line.split(marker, 1)[1]
                filled = int(seg.split("model tile", 1)[0].strip())
                xs = int(seg.split("+", 1)[1].split("X slot", 1)[0].strip())
                return filled, xs
            except Exception:
                return None
    return None


def run_one_set(finder, idx, models, pred_prefix, wl, do_shot):
    from unified_viewer import VISUALIZABLE_MODELS  # noqa
    inter = sorted(set(models) & wl)
    exp_fill = min(4, len(inter)); exp_x = 4 - exp_fill
    row = {"set": idx, "models": models, "n_models": len(models),
           "expected_gaze": exp_fill, "expected_x": exp_x,
           "predict_ok": False, "exec_ok": False, "completed_hold": False,
           "actual_fill": None, "actual_x": None, "exit_code": None,
           "orphans": None, "ood_warnings": [], "best": None,
           "start_ts": None, "end_ts": None, "error": ""}
    sched = OUT / f"set{idx:02d}.yaml"
    log_path = OUT / f"set{idx:02d}.log"
    proc = None
    try:
        # 1) build schedule + 2) predict best (production path, no OOD warning expected)
        finder.build_schedule_from_selection(models, str(sched))
        row["ood_warnings"] = finder.check_selection_coverage(models, pred_prefix)
        best, df = finder.predict_best_combination(str(sched), pred_prefix)
        row["best"] = best
        row["predict_ok"] = bool(best) and not row["ood_warnings"]

        # 3) launch the real execution window for this combo
        env = dict(os.environ)
        env["WARMUP_SECONDS"] = str(WARMUP)
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("QT_QPA_PLATFORM", None)
        args = [str(LAUNCHER), "--schedule", str(sched),
                "--schedule_name", best or "combination_1", "--duration", str(DURATION)]
        row["start_ts"] = now_ts()
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(args, cwd=str(ROOT), env=env,
                                    stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        pgid = proc.pid  # start_new_session -> pid is the new group/session leader

        # 4) hold ~HOLD_SECS; screenshot near the end for the designated set
        t0 = time.time(); shot_done = False
        while time.time() - t0 < HOLD_SECS:
            if proc.poll() is not None:
                break  # child died on its own before the hold elapsed
            if do_shot and not shot_done and (time.time() - t0) >= SHOT_AT:
                screenshot(OUT / f"shot_set{idx:02d}.png")
                shot_done = True
            time.sleep(0.5)
        row["completed_hold"] = proc.poll() is None  # still alive at end of hold

        # 5) terminate CLEANLY: SIGTERM the executor -> shutdown_all disposes NPU models
        if proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=TERM_WAIT)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)  # backstop if graceful exit hangs
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
        row["exit_code"] = proc.returncode
        row["end_ts"] = now_ts()

        # read what the window actually laid out
        gp = parse_grid_line(log_path)
        if gp is not None:
            row["actual_fill"], row["actual_x"] = gp
            row["exec_ok"] = True
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["end_ts"] = now_ts()
    finally:
        # 6) cleanup even on failure: make sure NOTHING from this set survives
        if proc is not None and proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGTERM); time.sleep(2)
            except Exception:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                proc.wait(timeout=8)
            except Exception:
                pass
            if row["exit_code"] is None:
                row["exit_code"] = proc.returncode
        # confirm the group is gone
        leftover = group_alive(proc.pid) if proc is not None else []
        # exclude our own pid just in case
        leftover = [p for p in leftover if p != os.getpid()]
        row["orphans"] = len(leftover)
        row["orphan_pids"] = leftover
    return row


def main():
    from PyQt5.QtWidgets import QApplication
    from best_deploy_finder_executor import BestDeployFinderApp as BestDeployFinder
    from unified_viewer import VISUALIZABLE_MODELS

    if not LAUNCHER.exists():
        print(f"[FATAL] launcher not found: {LAUNCHER}"); sys.exit(2)

    app = QApplication(sys.argv)          # widgets need an app; the finder is never shown
    finder = BestDeployFinder()
    if hasattr(finder, "device_config_combo"):
        finder.device_config_combo.setCurrentText("CPU-NPU")  # -> deploy_cpu_npu, [cpu,npu]
    pred_prefix = finder.prediction_model_input.text()
    print(f"[Test] predictor prefix = {pred_prefix}")
    print(f"[Test] whitelist = {sorted(VISUALIZABLE_MODELS)}")

    sets = finder.trained_working_sets(pred_prefix)   # 15 lists, coverage order
    print(f"[Test] {len(sets)} trained working sets (CPU-NPU). "
          f"HOLD={HOLD_SECS}s DURATION={DURATION}s STABILIZE={STABILIZE}s")

    # screenshot the first set whose whitelist intersection is exactly 2 (the 2-tile+2-X
    # case that this change is about).
    shot_idx = next((i for i, m in enumerate(sets, 1)
                     if len(set(m) & VISUALIZABLE_MODELS) == 2), None)

    results = []
    for idx, models in enumerate(sets, 1):
        print(f"\n===== SET {idx}/{len(sets)}: {','.join(models)} =====", flush=True)
        stray_before = stray_executors()
        if stray_before:
            print(f"[Test][WARN] stray executor(s) BEFORE set {idx}: {stray_before}")
        row = run_one_set(finder, idx, models, pred_prefix,
                          VISUALIZABLE_MODELS, do_shot=(idx == shot_idx))
        row["stray_before"] = len(stray_before)
        # global confirmation: no executor anywhere before we continue
        time.sleep(1.0)
        row["stray_after"] = len(stray_executors())
        results.append(row)
        print(f"[Set {idx}] predict_ok={row['predict_ok']} exec_ok={row['exec_ok']} "
              f"grid={row['actual_fill']}+{row['actual_x']}X "
              f"(exp {row['expected_gaze']}+{row['expected_x']}X) "
              f"exit={row['exit_code']} orphans={row['orphans']} "
              f"completed_hold={row['completed_hold']} err='{row['error']}'", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
        # 2.2.2 short stabilization so NPU/OS resources are fully returned
        time.sleep(STABILIZE)

    # summary
    ok = sum(1 for r in results if r["predict_ok"] and r["exec_ok"]
             and r["actual_fill"] == r["expected_gaze"] and r["orphans"] == 0)
    print(f"\n[Test] DONE. {ok}/{len(results)} sets fully OK "
          f"(predict+exec+grid match+no orphan).")
    # overlap check: end_ts[i] < start_ts[i+1]
    overlaps = 0
    for a, b in zip(results, results[1:]):
        if a["end_ts"] and b["start_ts"] and a["end_ts"] >= b["start_ts"]:
            overlaps += 1
    print(f"[Test] sequential overlap violations: {overlaps} (must be 0)")
    print(f"[Test] results -> {OUT/'results.json'}")
    print("TEST_ALL_SETS_DONE")
    os._exit(0)


if __name__ == "__main__":
    main()
