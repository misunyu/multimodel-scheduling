"""rev12 aggregation — T1 + T2 + T3 from pinned-state data.

Inputs (all in same pinned state: mxq b2441f9d, driver 580.159.03):
  results/rev10_gen_single.csv    L0 single-stream for s/m/l/x × {GPU,NPU} × 24 sids
  results/rev10_gen_gain.csv      N=4 multi-stream for s/m/l/x × 4 strat × 2 bg × 3 reps
  results/rev12_l2lm_single.csv   L2_LM single-stream for s/m/l/x × {GPU,NPU} × 24 sids  (this rev12)
  results/repro_driverstate.csv   T-D0 8-run anchor (yolo11s L0 large gap)

Outputs (T1/T2/T3 per claude_code_experiments.md spec):
  results/rev12_single_stream.csv     T1
  results/rev12_gen_decomp.csv        T2
  results/rev12_gen_gain.csv          T3
  results/rev12_state_meta.json       state pin (mxq sha, driver, git, ts, sources)
  results/rev12_summary.md            human-readable summary + gate verdicts

No prose / table edits to paper/.
"""

from __future__ import annotations

import csv, json, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path("accv_experiments/results")
DETECTORS = [("yolo11s", 9.4), ("yolo11m", 20.1),
              ("yolo11l", 25.3), ("yolo11x", 56.9)]
INT8_MXQ = {
    "yolo11s": ("models/mobilint_backup/yolo11s.mxq", "b2441f9d"),
    "yolo11m": ("/home/msyu/.mblt_model_zoo/vision/aries/single/yolo11m.mxq", "0c95402b"),
    "yolo11l": ("/home/msyu/.mblt_model_zoo/vision/aries/single/yolo11l.mxq", "471971b3"),
    "yolo11x": ("/home/msyu/.mblt_model_zoo/vision/aries/single/yolo11x.mxq", "11ccf20f"),
}

L0   = RES / "rev10_gen_single.csv"
L2   = RES / "rev12_l2lm_single.csv"
GAIN = RES / "rev10_gen_gain.csv"
TD0  = RES / "repro_driverstate.csv"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def state_meta():
    return {
        "mxq_sha8_anchor_yolo11s": "b2441f9d",
        "driver": "580.159.03",
        "sdk_pin": "models/mblt_model_zoo  (T-D0 verified state)",
        "git_commit": git_commit(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {
            "L0_single_stream":  str(L0)   + " (rev10 A-1, GATE-Q PASS rel=-0.201)",
            "L2_LM_single_stream": str(L2) + " (rev12, fresh)",
            "N4_multi_stream":   str(GAIN) + " (rev10 A-2)",
            "TD0_anchor":        str(TD0)  + " (8-run cold/warm/loaded)",
        },
    }


# ============================ T1 ============================

def task_T1():
    """yolo11s 24-log single-stream L0 per-size: GPU/NPU mean±std + Wilcoxon p."""
    df = pd.read_csv(L0)
    s = df[df.detector == "yolo11s"]
    g = s[s.device == "GPU"].sort_values("sid").reset_index(drop=True)
    n = s[s.device == "NPU"].sort_values("sid").reset_index(drop=True)
    assert len(g) == len(n) == 24, f"yolo11s L0 24-log expected, got gpu={len(g)} npu={len(n)}"
    out = []
    rows_per_size = []
    for col_g, col_n, size in [("sap_s","sap_s","small"),
                                  ("sap_m","sap_m","medium"),
                                  ("sap_l","sap_l","large")]:
        gv = g[col_g].to_numpy(); nv = n[col_n].to_numpy()
        gap = nv - gv
        try:
            stat, p = wilcoxon(gv, nv, alternative="two-sided", zero_method="wilcox")
            p = float(p)
        except Exception as e:
            stat, p = float("nan"), float("nan")
        rel_gap = float(np.mean(gap) / np.mean(gv)) if np.mean(gv) else float("nan")
        out.append({
            "size": size,
            "n_sids": 24,
            "gpu_mean": float(np.mean(gv)), "gpu_std": float(np.std(gv, ddof=0)),
            "npu_mean": float(np.mean(nv)), "npu_std": float(np.std(nv, ddof=0)),
            "gap_mean": float(np.mean(gap)), "gap_std": float(np.std(gap, ddof=0)),
            "rel_gap_mean": rel_gap,
            "wilcoxon_p_2sided": p,
        })
    # state meta
    pd.DataFrame(out).round(4).to_csv(RES / "rev12_single_stream.csv", index=False)
    # acceptance gate
    large = next(r for r in out if r["size"] == "large")
    target = -0.096; tol = 0.002
    gap = large["gap_mean"]
    gate = {
        "pass": bool(abs(gap - target) <= tol),
        "measured_large_gap": gap,
        "target": target,
        "tol": tol,
        "rel_large": large["rel_gap_mean"],
    }
    return out, gate


# ============================ T2 ============================

def int8_export_status():
    """For each detector, report whether a usable INT8 mxq exists in pinned state."""
    out = {}
    for det, _pm in DETECTORS:
        path, sha = INT8_MXQ[det]
        exists = Path(path).exists()
        out[det] = {"int8_export_available": exists, "path": path, "expected_sha8": sha}
    return out


def task_T2():
    """For each detector: per-size quantization at L0 + single-stream staleness at L2_LM.

    Quantization at L0 = NPU L0 mean - GPU L0 mean (negative = NPU loses).
    Staleness at L2_LM = X L0 mean - X L2_LM mean (positive = X degrades under bg).
    """
    l0 = pd.read_csv(L0)
    l2 = pd.read_csv(L2)
    int8 = int8_export_status()
    out = []
    for det, pm in DETECTORS:
        sub0 = l0[l0.detector == det]
        sub2 = l2[l2.detector == det]
        row = {"detector": det, "params_M": pm,
                "int8_export_available": int8[det]["int8_export_available"],
                "mxq_path": int8[det]["path"],
                "mxq_sha8": int8[det]["expected_sha8"]}
        for dev, prefix in [("GPU", "gpu"), ("NPU", "npu")]:
            g0 = sub0[sub0.device == dev]
            g2 = sub2[sub2.device == dev]
            row[f"{prefix}_l0_n_sids"] = int(len(g0))
            row[f"{prefix}_l2lm_n_sids"] = int(len(g2))
            for col, sfx in [("sap_s","s"), ("sap_m","m"), ("sap_l","l")]:
                row[f"{prefix}_l0_{sfx}"]   = float(g0[col].mean()) if len(g0) else float("nan")
                row[f"{prefix}_l2lm_{sfx}"] = float(g2[col].mean()) if len(g2) else float("nan")
                row[f"{prefix}_stale_{sfx}"] = row[f"{prefix}_l0_{sfx}"] - row[f"{prefix}_l2lm_{sfx}"]
        # quantization per size = NPU L0 - GPU L0
        for sfx in ["s", "m", "l"]:
            row[f"Q_{sfx}"] = row[f"npu_l0_{sfx}"] - row[f"gpu_l0_{sfx}"]
            if row[f"gpu_l0_{sfx}"]:
                row[f"Q_{sfx}_rel"] = row[f"Q_{sfx}"] / row[f"gpu_l0_{sfx}"]
            else:
                row[f"Q_{sfx}_rel"] = float("nan")
        out.append(row)
    pd.DataFrame(out).round(4).to_csv(RES / "rev12_gen_decomp.csv", index=False)
    # T2 acceptance gate: yolo11s large = -0.096 ± 0.002 matches T1
    s_row = next(r for r in out if r["detector"] == "yolo11s")
    gate = {
        "pass": bool(abs(s_row["Q_l"] - (-0.096)) <= 0.002),
        "yolo11s_Q_large": s_row["Q_l"],
        "yolo11s_Q_large_rel": s_row["Q_l_rel"],
        "target": -0.096,
        "tol": 0.002,
    }
    return out, gate


# ============================ T3 ============================

def task_T3():
    """For each detector: N=4 / L1_light / Comp.A — Contention-aware (SizeAware) vs
    Isolated (SizeBlindRev) worst & mean gain across 3 reps."""
    df = pd.read_csv(GAIN)
    df = df[df.bg_level == "L1_light"]
    out = []
    for det, pm in DETECTORS:
        sub = df[df.detector == det]
        sa  = sub[sub.placement_name == "SizeAware"].sort_values("rep_idx")
        sbr = sub[sub.placement_name == "SizeBlindRev"].sort_values("rep_idx")
        if not (len(sa) == 3 and len(sbr) == 3):
            continue
        worst_gains = (sa["worst_sap"].to_numpy() - sbr["worst_sap"].to_numpy()).tolist()
        mean_gains  = (sa["mean_sap"].to_numpy()  - sbr["mean_sap"].to_numpy()).tolist()
        inverts_worst = all(g > 0 for g in worst_gains)
        inverts_mean  = all(g > 0 for g in mean_gains)
        out.append({
            "detector": det, "params_M": pm,
            "n_reps": 3,
            "worst_gain_rep0": worst_gains[0],
            "worst_gain_rep1": worst_gains[1],
            "worst_gain_rep2": worst_gains[2],
            "worst_gain_mean": float(np.mean(worst_gains)),
            "worst_gain_std":  float(np.std(worst_gains, ddof=0)),
            "mean_gain_rep0":  mean_gains[0],
            "mean_gain_rep1":  mean_gains[1],
            "mean_gain_rep2":  mean_gains[2],
            "mean_gain_mean":  float(np.mean(mean_gains)),
            "mean_gain_std":   float(np.std(mean_gains, ddof=0)),
            "inverts_worst_all_reps": inverts_worst,
            "inverts_mean_all_reps":  inverts_mean,
        })
    pd.DataFrame(out).round(4).to_csv(RES / "rev12_gen_gain.csv", index=False)
    # T3 acceptance: all 4 detectors have positive worst_gain_mean
    gate = {
        "pass": bool(all(r["worst_gain_mean"] > 0 for r in out)),
        "n_detectors": len(out),
        "positive_worst_gain": [r["detector"] for r in out if r["worst_gain_mean"] > 0],
    }
    return out, gate


# ============================ summary ============================

def write_summary(t1, gq1, t2, gq2, t3, gq3, meta):
    buf = []
    buf.append("# rev12 summary — Tables 1·5·6 in single pinned state\n\n")
    buf.append("_Aggregation only. Re-measurement of L2_LM 24-log added for staleness; "
                "L0 and N=4 reused from rev10 (same pinned state per T-D0 + GATE-Q PASS). "
                "`paper/main_vision.tex` and `paper/tables/*.tex` NOT modified._\n\n")

    # State meta
    buf.append("## State pin\n\n")
    buf.append(f"- mxq sha8 (yolo11s anchor): `{meta['mxq_sha8_anchor_yolo11s']}`\n")
    buf.append(f"- driver: `{meta['driver']}`\n")
    buf.append(f"- SDK: `{meta['sdk_pin']}`\n")
    buf.append(f"- git: `{meta['git_commit'][:12]}`\n")
    buf.append(f"- ts: `{meta['ts_iso']}`\n\n")
    buf.append("Source CSVs:\n")
    for k, v in meta["sources"].items():
        buf.append(f"  - {k}: `{v}`\n")
    buf.append("\n")

    # T1
    buf.append("## T1 — single-stream yolo11s L0 (24 logs, per size)\n\n")
    ok = "**PASS**" if gq1["pass"] else "**FAIL**"
    buf.append(f"### Acceptance gate (large gap = -0.096 ± 0.002): {ok}\n\n")
    buf.append(f"- measured: `{gq1['measured_large_gap']:+.4f}`, rel = `{gq1['rel_large']:+.3f}`\n\n")
    buf.append("| size | GPU mean±std | NPU mean±std | gap (mean ± std) | rel gap | Wilcoxon p |\n|---|---|---|---|---|---|\n")
    for r in t1:
        buf.append(f"| {r['size']} | `{r['gpu_mean']:.4f}±{r['gpu_std']:.4f}` | "
                    f"`{r['npu_mean']:.4f}±{r['npu_std']:.4f}` | "
                    f"`{r['gap_mean']:+.4f} ± {r['gap_std']:.4f}` | "
                    f"`{r['rel_gap_mean']:+.3f}` | `{r['wilcoxon_p_2sided']:.2e}` |\n")
    buf.append("\n")

    # T2
    buf.append("## T2 — gen-decomp (4 detectors × {L0, L2_LM})\n\n")
    ok = "**PASS**" if gq2["pass"] else "**FAIL**"
    buf.append(f"### Acceptance gate (yolo11s Q_large = -0.096 ± 0.002 ≈ T1): {ok}\n\n")
    buf.append(f"- yolo11s Q_large = `{gq2['yolo11s_Q_large']:+.4f}`, rel = `{gq2['yolo11s_Q_large_rel']:+.3f}`\n\n")
    buf.append("| detector | params (M) | INT8 export? | Q_S | Q_M | Q_L (rel) | GPU stale_L (L0→L2_LM) | NPU stale_L |\n|---|---|---|---|---|---|---|---|\n")
    for r in t2:
        ex = "✓" if r["int8_export_available"] else "✗"
        buf.append(f"| {r['detector']} | {r['params_M']:.1f} | {ex} | "
                    f"`{r['Q_s']:+.3f}` | `{r['Q_m']:+.3f}` | "
                    f"`{r['Q_l']:+.3f} ({r['Q_l_rel']:+.2f})` | "
                    f"`{r['gpu_stale_l']:+.3f}` | `{r['npu_stale_l']:+.3f}` |\n")
    buf.append("\n_Q_X = NPU L0 sAP_X - GPU L0 sAP_X (quantization loss per size). "
                "stale_L = device L0 sAP_L - device L2_LM sAP_L (positive = degrades under bg)._\n\n")

    # T3
    buf.append("## T3 — gen-gain (4 detectors × N=4 / L1_light / Comp.A, 3 reps)\n\n")
    ok = "**PASS**" if gq3["pass"] else "**FAIL**"
    buf.append(f"### Acceptance gate (all 4 worst_gain > 0): {ok}\n\n")
    buf.append("| detector | worst_gain (mean ± std) | mean_gain (mean ± std) | inverts worst (3/3)? |\n|---|---|---|---|\n")
    for r in t3:
        inv = "✓" if r["inverts_worst_all_reps"] else "✗"
        buf.append(f"| {r['detector']} | `{r['worst_gain_mean']:+.4f} ± {r['worst_gain_std']:.4f}` | "
                    f"`{r['mean_gain_mean']:+.4f} ± {r['mean_gain_std']:.4f}` | {inv} |\n")
    buf.append("\n_Gain = Contention-aware (SizeAware) − Isolated (SizeBlindRev)._\n\n")

    # T4 status
    buf.append("## T4 — N=8 8-run reproducibility\n\n")
    buf.append("**SKIPPED per spec recommendation.** Use rev11 3-run wording: "
                "\"across three runs, per-strategy std ≤ 0.0021\" (rev11_n8.md verified).\n\n")

    buf.append("---\n\n_End. main_vision.tex and paper/tables/* NOT modified._\n")
    (RES / "rev12_summary.md").write_text("".join(buf))


# ============================ main ============================

def main():
    meta = state_meta()
    (RES / "rev12_state_meta.json").write_text(json.dumps(meta, indent=2))

    t1, gq1 = task_T1()
    t2, gq2 = task_T2()
    t3, gq3 = task_T3()
    write_summary(t1, gq1, t2, gq2, t3, gq3, meta)

    print("== rev12 aggregation done ==")
    print(f"T1 gate: {gq1}")
    print(f"T2 gate: {gq2}")
    print(f"T3 gate: {gq3}")


if __name__ == "__main__":
    main()
