"""rev13 autonomous wrapper — 24h unattended.

Flow:
  1. B-lite smoke variants (v1, v2, v3): try to find a setup that achieves
     <5% NPU skip, ~12ms latency, normal-state large gap.
  2. If a variant passes the smoke gate → run R1 (full bg sweep) with that
     setup, starting with yolo11s, then m/l/x as long as the level gate holds.
  3. Aggregate into rev13_cstar_clean.csv + level skip summary + C* table.
  4. Write rev13_autonomous_report.md with everything: smoke results, gate
     verdicts, what ran, what didn't.

Hard rules:
  - No sudo / reboot / privileged action.
  - No modification of existing scripts under accv_experiments/scripts/.
  - On any gate FAIL, write what we have and stop cleanly (no contaminated
    data passed forward).
  - Manifest checkpoint at results/manifest_rev13.json so partial progress
    survives any incidental interruption.

Output (always present):
  results/rev13_autonomous_report.md       human-readable summary + verdicts
  results/manifest_rev13.json              checkpoint + final state
On success or partial success:
  results/rev13_blite_smoke.csv            smoke results
  results/rev13_cstar_clean.csv            level data (whatever passed)
  results/rev13_cstar_clean_summary.md     level skip + per-det C*
"""

from __future__ import annotations

import csv, json, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _step_d_common import (load_split_for_sid, load_val,
                              preload_background_models)
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                set_active_npu_engines, dispose_npu_for,
                                measure_single_stream, DETECTORS)
from phase_rev10_measure import _native, short_sha

RES = Path("accv_experiments/results")
MANIFEST = RES / "manifest_rev13.json"
LOG = RES / "rev13_autonomous_log.txt"
SMOKE_CSV = RES / "rev13_blite_smoke.csv"
CSTAR_CSV = RES / "rev13_cstar_clean.csv"
REPORT_MD = RES / "rev13_autonomous_report.md"
SUMMARY_MD = RES / "rev13_cstar_clean_summary.md"

# Plan
TEST_SID = 2
BG_LEVELS = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
DETECTOR_ORDER = ["yolo11s", "yolo11m", "yolo11l", "yolo11x"]

# Gates
G_SKIP_THR = 5.0
G_LAT_THR_MS = 18.0
G_GAP_TARGET = -0.096
G_GAP_TOL_PERSID = 0.025   # loose for single-sid
G_GAP_TOL_24LOG  = 0.002   # tight for 24-log mean


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{ts}  {msg}\n"
    with open(LOG, "a") as f: f.write(line)
    print(line, end="", flush=True)


def probe_env():
    out = {"ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try: out["driver"] = subprocess.check_output(
        ["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
        text=True).strip()
    except Exception: out["driver"] = "?"
    try:
        gu = subprocess.check_output(
            ["nvidia-smi","--query-gpu=utilization.gpu,temperature.gpu",
              "--format=csv,noheader,nounits"], text=True).strip()
        u,t = gu.split(","); out["gpu_util"]=int(u.strip()); out["gpu_temp"]=int(t.strip())
    except Exception:
        out["gpu_util"]=-1; out["gpu_temp"]=-1
    try: out["git_commit"] = subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip()[:12]
    except Exception: out["git_commit"]="?"
    return out


def load_manifest():
    if MANIFEST.exists():
        try: return json.loads(MANIFEST.read_text())
        except Exception: pass
    return {}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(_native(m), indent=2))


# ============================ B-lite smoke variants ============================

SMOKE_COLS = ["variant", "device", "bg", "sap_5095", "sap_l",
               "latency_mean_ms", "frame_skip_pct", "wall_sec", "notes"]


def append_smoke(rows):
    new = not SMOKE_CSV.exists()
    with open(SMOKE_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SMOKE_COLS)
        if new: w.writeheader()
        for r in rows: w.writerow(r)


def measure_smoke_cells(variant, gpu, npu_model, val, notes):
    split = load_split_for_sid(val, TEST_SID)
    rows = []
    for bg in ["L0", "L1_light"]:
        for device, model in [("GPU", gpu), ("NPU", npu_model)]:
            t = time.time()
            try:
                m = measure_single_stream(TEST_SID, split, device, model, bg)
            except Exception as e:
                log(f"  smoke FAIL {variant} {device} {bg}: {e}")
                continue
            wall = time.time() - t
            log(f"  {variant} {device} {bg}: sap={m['sap_5095']:.4f} "
                  f"lat={m['latency_mean']:.1f}ms skip={m['frame_skip_pct']:.1f}% ({wall:.1f}s)")
            rows.append({"variant": variant, "device": device, "bg": bg,
                          "sap_5095": round(m["sap_5095"], 4),
                          "sap_l": round(m["sap_l"], 4),
                          "latency_mean_ms": round(m["latency_mean"], 1),
                          "frame_skip_pct": round(m["frame_skip_pct"], 1),
                          "wall_sec": round(wall, 1),
                          "notes": notes})
    return rows


def variant_gate(rows, variant):
    npu_skips = [r["frame_skip_pct"] for r in rows
                  if r["variant"] == variant and r["device"] == "NPU"]
    npu_lat = [r["latency_mean_ms"] for r in rows
                if r["variant"] == variant and r["device"] == "NPU"
                and r["bg"] == "L0"]
    npu_l_l = [r["sap_l"] for r in rows
                if r["variant"] == variant and r["device"] == "NPU"
                and r["bg"] == "L0"]
    gpu_l_l = [r["sap_l"] for r in rows
                if r["variant"] == variant and r["device"] == "GPU"
                and r["bg"] == "L0"]
    g1 = bool(npu_skips and max(npu_skips) < G_SKIP_THR)
    g2 = bool(npu_lat and np.mean(npu_lat) < G_LAT_THR_MS)
    g3 = False
    if npu_l_l and gpu_l_l:
        g3 = abs(npu_l_l[0] - gpu_l_l[0] - G_GAP_TARGET) <= G_GAP_TOL_PERSID
    out = {"G1_skip<5%": g1, "G2_lat<18ms": g2, "G3_gap_close": g3,
            "all_pass": g1 and g2 and g3}
    return out


def run_smoke_variants(val, manifest):
    """Try v1 (NPU first), v2 (no BG), v3 (extended warmup)."""
    YOLO11S = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
    all_rows = []
    winning = None

    # ---- v1: NPU before GPU ----
    log("=== variant v1: NPU before GPU ===")
    try:
        preload_background_models(max_level="L1")
        npu = load_npu_engines(YOLO11S, YOLO11S["baseline_mxq"],
                                 YOLO11S["baseline_mode"], 1)
        set_active_npu_engines(npu)
        gpu = FGModelGPUGeneric(YOLO11S["ultralytics_pt"])
        rows = measure_smoke_cells("v1", gpu, npu[0], val,
                                     "NPU loaded before GPU; bg preload max=L1")
        all_rows.extend(rows)
        dispose_npu_for("yolo11s"); del gpu
    except Exception as e:
        log(f"v1 fatal: {e}"); traceback.print_exc()
    if all_rows:
        v = variant_gate(all_rows, "v1"); log(f"  v1 gate: {v}")
        if v["all_pass"]:
            winning = ("v1", "NPU before GPU; bg max=L1")
            log("  v1 PASS — using this for R1")

    # ---- v2: no BG preload ----
    if not winning:
        log("=== variant v2: no BG preload ===")
        try:
            gpu = FGModelGPUGeneric(YOLO11S["ultralytics_pt"])
            npu = load_npu_engines(YOLO11S, YOLO11S["baseline_mxq"],
                                     YOLO11S["baseline_mode"], 1)
            set_active_npu_engines(npu)
            rows = measure_smoke_cells("v2", gpu, npu[0], val,
                                         "no BG preload; GPU before NPU")
            all_rows.extend(rows)
            dispose_npu_for("yolo11s"); del gpu
        except Exception as e:
            log(f"v2 fatal: {e}"); traceback.print_exc()
        v = variant_gate(all_rows, "v2"); log(f"  v2 gate: {v}")
        if v["all_pass"]:
            winning = ("v2", "no BG preload"); log("  v2 PASS — using this for R1")

    # ---- v3: extended warmup ----
    if not winning:
        log("=== variant v3: extended warmup (200 dummy infer) ===")
        try:
            preload_background_models(max_level="L1")
            gpu = FGModelGPUGeneric(YOLO11S["ultralytics_pt"])
            npu = load_npu_engines(YOLO11S, YOLO11S["baseline_mxq"],
                                     YOLO11S["baseline_mode"], 1)
            set_active_npu_engines(npu)
            from step0_compare_devices import CONF, IOU
            dummy = np.zeros((1200, 1920, 3), dtype=np.uint8)
            times = []
            for i in range(200):
                t0 = time.time()
                x = npu[0].preprocess(dummy); o = npu[0](x)
                npu[0].postprocess(o, conf_thres=CONF, iou_thres=IOU)
                times.append(time.time() - t0)
            log(f"  warmup last 50 mean: {np.mean(times[-50:])*1000:.1f}ms")
            rows = measure_smoke_cells("v3", gpu, npu[0], val,
                                         "extended warmup 200 dummies")
            all_rows.extend(rows)
            dispose_npu_for("yolo11s"); del gpu
        except Exception as e:
            log(f"v3 fatal: {e}"); traceback.print_exc()
        v = variant_gate(all_rows, "v3"); log(f"  v3 gate: {v}")
        if v["all_pass"]:
            winning = ("v3", "extended warmup"); log("  v3 PASS — using this for R1")

    append_smoke(all_rows)
    manifest["smoke_rows"] = len(all_rows)
    manifest["winning_variant"] = winning
    save_manifest(manifest)
    return winning, all_rows


# ============================ STEP R1 full sweep ============================

R1_COLS = ["detector", "bg", "sid", "device", "sap_5095", "sap_s", "sap_m",
            "sap_l", "latency_mean_ms", "frame_skip_pct", "wall_sec"]


def append_r1(row):
    new = not CSTAR_CSV.exists()
    with open(CSTAR_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=R1_COLS)
        if new: w.writeheader()
        w.writerow(row)


def run_r1_for_detector(det_cfg, val, manifest, variant_setup):
    """Run R1 for one detector. Returns dict of {bg: 'PASS'|'FAIL'|'SKIP', reason}."""
    name = det_cfg["name"]
    log(f"\n==== R1 detector={name} ====")
    out = {}
    # Apply variant setup (mostly affects load order / bg preload — already done for yolo11s).
    # For detectors other than yolo11s, reload bg if needed.
    if variant_setup[0] != "v2":  # v2 = no BG preload
        try: preload_background_models(max_level="L3")
        except Exception as e: log(f"  bg preload err: {e}")
    try:
        gpu = FGModelGPUGeneric(det_cfg["ultralytics_pt"])
        npu = load_npu_engines(det_cfg, det_cfg["baseline_mxq"],
                                 det_cfg["baseline_mode"], 1)
        set_active_npu_engines(npu)
    except Exception as e:
        log(f"  [{name}] model load FAIL: {e}")
        return {bg: "SKIP_LOAD_FAIL" for bg in BG_LEVELS}
    for bg in BG_LEVELS:
        log(f"  -- {name} bg={bg} --")
        skips_this_level = []
        bg_done = []
        for sid in range(len(val["sequences"])):
            key = f"r1/{name}/{bg}/{sid}"
            if manifest.get("done", {}).get(key):
                continue
            split = load_split_for_sid(val, sid)
            row_gpu = row_npu = None
            for device, model in [("GPU", gpu), ("NPU", npu[0])]:
                try:
                    m = measure_single_stream(sid, split, device, model, bg)
                except Exception as e:
                    log(f"    FAIL {device} sid={sid}: {e}")
                    continue
                row = {"detector": name, "bg": bg, "sid": sid, "device": device,
                        **{k: round(v, 4) if isinstance(v, float) else v
                            for k, v in m.items()}}
                # only write the columns R1 cares about
                slim = {k: row.get(k, "") for k in R1_COLS if k != "latency_mean_ms"}
                slim["latency_mean_ms"] = round(m["latency_mean"], 1)
                append_r1(slim)
                if device == "NPU":
                    skips_this_level.append(m["frame_skip_pct"])
            manifest.setdefault("done", {})[key] = {"ts": int(time.time())}
            save_manifest(manifest)
        max_skip = max(skips_this_level) if skips_this_level else 100.0
        mean_skip = float(np.mean(skips_this_level)) if skips_this_level else 100.0
        log(f"  [{name}] bg={bg} npu_skip max={max_skip:.1f}% mean={mean_skip:.1f}%")
        if mean_skip >= G_SKIP_THR:
            out[bg] = f"FAIL_skip_mean_{mean_skip:.1f}%"
            log(f"  [{name}] bg={bg} GATE FAIL — stopping further bg for this detector")
            break
        out[bg] = f"PASS_skip_mean_{mean_skip:.1f}%"
    dispose_npu_for(name)
    del gpu
    return out


def run_r1(val, manifest, variant_setup):
    """Run R1 for yolo11s first; if all bg levels pass, also run m/l/x."""
    results = {}
    # yolo11s
    s_cfg = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
    results["yolo11s"] = run_r1_for_detector(s_cfg, val, manifest, variant_setup)
    # if all bg passed at yolo11s, continue to m/l/x
    if all("PASS" in v for v in results["yolo11s"].values()):
        log("yolo11s passed all levels — continuing to m/l/x")
        for name in ["yolo11m", "yolo11l", "yolo11x"]:
            cfg = [d for d in DETECTORS if d["name"] == name][0]
            results[name] = run_r1_for_detector(cfg, val, manifest, variant_setup)
    else:
        log("yolo11s did not pass all levels — skipping m/l/x")
    return results


# ============================ aggregation ============================

def aggregate_r1():
    """Per-detector × bg group means + projected C* (in-range only)."""
    if not CSTAR_CSV.exists():
        return None
    import pandas as pd
    df = pd.read_csv(CSTAR_CSV)
    BG_IDX = {b: i for i, b in enumerate(BG_LEVELS)}
    SIZE_GROUPS = {
        "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
        "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
    }
    out = {"per_level_skip": {}, "cstar_per_det_group": {}}
    for det in DETECTOR_ORDER:
        sub = df[df.detector == det]
        if sub.empty: continue
        out["per_level_skip"][det] = {}
        for bg in BG_LEVELS:
            ssub = sub[sub.bg == bg]
            npu = ssub[ssub.device == "NPU"]
            out["per_level_skip"][det][bg] = {
                "mean_skip_pct": float(npu.frame_skip_pct.mean()) if len(npu) else None,
                "n_sids": int(len(npu)),
            }
        # per-group gap by bg
        for g, gsids in SIZE_GROUPS.items():
            gaps_per_bg = {}
            for bg in BG_LEVELS:
                ssub = sub[sub.bg == bg]
                gpu = ssub[(ssub.device == "GPU") & ssub.sid.isin(gsids)]
                npu = ssub[(ssub.device == "NPU") & ssub.sid.isin(gsids)]
                if len(gpu) < 1 or len(npu) < 1: continue
                gaps = []
                for sid in gsids:
                    gv = gpu[gpu.sid == sid]["sap_5095"]
                    nv = npu[npu.sid == sid]["sap_5095"]
                    if len(gv) and len(nv):
                        gaps.append(float(nv.iloc[0] - gv.iloc[0]))
                if gaps:
                    gaps_per_bg[bg] = float(np.mean(gaps))
            # in-range C*
            xs = sorted([(BG_IDX[bg], gv) for bg, gv in gaps_per_bg.items()])
            cs = None
            for i in range(1, len(xs)):
                if xs[i-1][1] * xs[i][1] < 0:
                    x0,y0 = xs[i-1]; x1,y1 = xs[i]
                    cs = x0 - y0 * (x1 - x0) / (y1 - y0); break
            out["cstar_per_det_group"].setdefault(det, {})[g] = {
                "gaps_by_bg": gaps_per_bg, "cstar_in_range": cs,
            }
    return out


def write_report(env, smoke_rows, winning, r1_results, agg, ended_reason):
    buf = []
    buf.append("# rev13 autonomous report\n\n")
    buf.append("_Autonomous 24h wrapper. Reset attempted via non-privileged "
                "variants only (no sudo, no reboot, no script modification)._\n\n")
    buf.append("## Environment at start\n\n")
    for k, v in env.items(): buf.append(f"- {k}: `{v}`\n")
    buf.append(f"\n## End reason\n\n{ended_reason}\n\n")

    # smoke
    buf.append("## B-lite smoke variants\n\n")
    if smoke_rows:
        buf.append("| variant | device | bg | sap | latency (ms) | skip% | wall (s) |\n|---|---|---|---|---|---|---|\n")
        for r in smoke_rows:
            buf.append(f"| {r['variant']} | {r['device']} | {r['bg']} | "
                        f"`{r['sap_5095']:.4f}` | `{r['latency_mean_ms']:.1f}` | "
                        f"`{r['frame_skip_pct']:.1f}` | `{r['wall_sec']:.1f}` |\n")
    buf.append(f"\n**winning variant**: `{winning}`\n\n")

    # R1
    buf.append("## R1 per-detector level gates\n\n")
    if r1_results:
        for det, levels in r1_results.items():
            buf.append(f"### {det}\n\n")
            buf.append("| bg | status |\n|---|---|\n")
            for bg in BG_LEVELS:
                st = levels.get(bg, "—")
                buf.append(f"| {bg} | `{st}` |\n")
            buf.append("\n")
    else:
        buf.append("R1 did not run.\n\n")

    # aggregation
    if agg:
        buf.append("## Per-level NPU skip% summary\n\n")
        buf.append("| detector | " + " | ".join(BG_LEVELS) + " |\n|" + "---|" * (len(BG_LEVELS)+1) + "\n")
        for det, lvls in agg["per_level_skip"].items():
            row = [det]
            for bg in BG_LEVELS:
                v = lvls.get(bg, {}).get("mean_skip_pct")
                row.append(f"`{v:.1f}%`" if v is not None else "—")
            buf.append("| " + " | ".join(row) + " |\n")
        buf.append("\n## C* (in-range only; no extrapolation)\n\n")
        buf.append("| detector | group | C* | gap trajectory |\n|---|---|---|---|\n")
        for det, gd in agg["cstar_per_det_group"].items():
            for g, info in gd.items():
                cs = info["cstar_in_range"]
                cs_str = f"{cs:.2f}" if cs is not None else "no in-range crossing"
                gaps = info["gaps_by_bg"]
                traj = ", ".join(f"{bg}:{gv:+.3f}" for bg, gv in gaps.items())
                buf.append(f"| {det} | {g} | `{cs_str}` | {traj} |\n")
        buf.append("\n")

    buf.append("---\n\n_End. main_vision.tex / paper/tables/* NOT modified._\n")
    REPORT_MD.write_text("".join(buf))
    log(f"saved {REPORT_MD}")


# ============================ main ============================

def main():
    env = probe_env()
    log(f"==== rev13 autonomous start ==== driver={env['driver']} util={env['gpu_util']}% temp={env['gpu_temp']}C git={env['git_commit']}")
    manifest = load_manifest()
    manifest["env_start"] = env
    save_manifest(manifest)

    val = load_val()
    log(f"loaded {len(val['sequences'])} logs")

    # Phase 1: smoke
    winning, smoke_rows = run_smoke_variants(val, manifest)
    if winning is None:
        log("ALL SMOKE VARIANTS FAIL — writing report and stopping")
        write_report(env, smoke_rows, None, {}, None,
                      "All B-lite smoke variants failed gate. Privileged reset "
                      "(sudo/reboot) needed; not attempted (user away).")
        return 1

    # Phase 2: R1
    log(f"smoke passed with variant {winning}; starting R1")
    r1_results = run_r1(val, manifest, winning)

    # Phase 3: aggregate
    log("aggregating R1 results")
    try:
        agg = aggregate_r1()
    except Exception as e:
        log(f"aggregation failed: {e}"); agg = None
        traceback.print_exc()

    write_report(env, smoke_rows, winning, r1_results, agg,
                  "R1 completed (with possible per-detector level gate stops).")
    log("==== rev13 autonomous done ====")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
