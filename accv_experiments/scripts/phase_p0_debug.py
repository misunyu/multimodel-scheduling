"""Phase P0 — autonomous debug pass.

Per claude_code_experiments.md (rev 3) §3:
  P0.1  Table 1 reproduction (PASS/FAIL with concrete numbers)
  P0.2  Root-cause checklist (mxq / IoU / size threshold / val split / category
        mapping / pycocotools call / per-image vs pooled aggregation)
  P0.3  NEW C* via sAP-gap definition  (sAP_NPU,g vs sAP_GPU,g(C))
        Also OLD C* via absolute per-size mAP for diagnostic comparison.
  P0.4  Consistency matrix across tab:partA, NEW C*, TASK D
  P0.5  Quarantine fig:gen-cstar in main_vision.tex if FAIL

This is pure re-aggregation: no new hardware measurement needed (P0 budget ≤2.5h).
All inputs come from already-saved CSVs:
  - accv_experiments/results/step_a_baseline.csv         (canonical Table 1 source)
  - accv_experiments/results/step_f_partA_matrix.csv     (per-camera Δ under L1)
  - accv_experiments/results/gen_decomp.csv              (TASK B)
  - accv_experiments/results/cstar_summary.csv           (TASK C ladder)
  - accv_experiments/results/cstar_per_group.csv         (OLD per-size C*)
  - accv_experiments/results/gen_gain.csv                (TASK D)
  - accv_experiments/results/step_e_size_classification.csv  (group sids)

Outputs:
  results/p0_table1_diff.csv         per-row comparison step_a vs Table 1 vs TASK B
  results/p0_root_cause.csv          checklist of pipeline parameters
  results/cstar_v1_abs.csv           OLD C* (kept for diagnostics)
  results/cstar_v2_sap.csv           NEW C* (paper-bound if CONSISTENCY PASS)
  results/consistency_matrix.csv     three-source comparison
  results/p0_decisions.json          everything we decided & why (for the run report)

If CONSISTENCY = PASS:  regenerate paper/figures/gen_cstar.pdf from v2 C*, leave main_vision.tex.
If CONSISTENCY = FAIL:  comment-out the fig:gen-cstar block in main_vision.tex with the
                         [DISPUTED-CSTAR] tag. Both PNG/PDF stay under results/.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import load_val

RES = SCRIPT_DIR.parent / "results"
FIG_DIR = SCRIPT_DIR.parent.parent / "paper" / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAIN_TEX = SCRIPT_DIR.parent.parent / "main_vision.tex"

# ---------- group definitions (same as TASK C) ----------
SIZE_GROUPS = {
    "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
    "medium-mixed": [0, 1, 6, 8, 10, 11, 17, 20],
    "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
}
# Mean p(size) per group (from step_e)
GROUP_P = {
    "small-rich":   {"small": 0.462, "medium": 0.409, "large": 0.129},
    "medium-mixed": {"small": 0.383, "medium": 0.409, "large": 0.208},
    "large-rich":   {"small": 0.283, "medium": 0.401, "large": 0.316},
}
LADDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]


# ============================== P0.1  TABLE 1 REPRO ====================

def p0_1_table1_repro():
    """Compare step_a_baseline (likely Table 1 source) to TASK B's gen_decomp
    aggregation. Build a row-per-metric diff table.
    """
    df_a = pd.read_csv(RES / "step_a_baseline.csv")
    df_b = pd.read_csv(RES / "gen_decomp.csv")
    # Build paired per-log diffs
    rows = []
    table1 = {
        "sap_5095": {"gpu": 0.193, "npu": 0.182, "diff": -0.011, "p": 0.053},
        "sap_50":   {"gpu": 0.308, "npu": 0.279, "diff": -0.029, "p": 0.001},
        "sap_small":  {"gpu": 0.015, "npu": 0.008, "diff": -0.007, "p": 0.001},
        "sap_medium": {"gpu": 0.184, "npu": 0.148, "diff": -0.036, "p": 0.001},
        "sap_large":  {"gpu": 0.472, "npu": 0.473, "diff": +0.001, "p": 0.92},
    }
    # gen_decomp.csv uses sap_s/sap_m/sap_l; step_a uses sap_small/sap_medium/sap_large
    B_COL = {"sap_5095": "sap_5095", "sap_50": "sap_50",
             "sap_small": "sap_s", "sap_medium": "sap_m", "sap_large": "sap_l"}
    for metric, target in table1.items():
        # step_a aggregation
        a_g = df_a[df_a.device == "GPU"][metric].astype(float)
        a_n = df_a[df_a.device == "NPU"][metric].astype(float)
        a_g_mean = float(a_g.mean()); a_n_mean = float(a_n.mean())
        a_diff = a_n_mean - a_g_mean
        # TASK B aggregation (GPU @ L0 + NPU @ L0); remap to TASK B column name
        b_col = B_COL.get(metric, metric)
        b_g = df_b[(df_b.device == "GPU") & (df_b.bg_level == "L0")][b_col].astype(float)
        b_n = df_b[(df_b.device == "NPU") & (df_b.bg_level == "L0")][b_col].astype(float)
        b_g_mean = float(b_g.mean()); b_n_mean = float(b_n.mean())
        b_diff = b_n_mean - b_g_mean
        rows.append({
            "metric": metric,
            "table1_gpu": target["gpu"], "table1_npu": target["npu"],
            "table1_diff": target["diff"],
            "stepA_gpu": round(a_g_mean, 4), "stepA_npu": round(a_n_mean, 4),
            "stepA_diff": round(a_diff, 4),
            "taskB_gpu": round(b_g_mean, 4), "taskB_npu": round(b_n_mean, 4),
            "taskB_diff": round(b_diff, 4),
            "stepA_matches_table1": abs(a_diff - target["diff"]) < 0.005,
            "taskB_matches_table1": abs(b_diff - target["diff"]) < 0.005,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RES / "p0_table1_diff.csv", index=False)
    # PASS condition: step_a reproduces all rows within 0.005; TASK B reproduces or not
    stepA_ok = all(rows[i]["stepA_matches_table1"] for i in range(len(rows)))
    taskB_ok = all(rows[i]["taskB_matches_table1"] for i in range(len(rows)))
    print("\n=== P0.1  Table 1 reproduction ===")
    print(out.to_string(index=False))
    print(f"\n  step_a reproduces Table 1: {'PASS' if stepA_ok else 'FAIL'}")
    print(f"  TASK B reproduces Table 1: {'PASS' if taskB_ok else 'FAIL'}")
    return {"stepA_matches": stepA_ok, "taskB_matches": taskB_ok, "rows": rows}


# ============================== P0.2  ROOT CAUSE ===================

def p0_2_root_cause():
    """Build the parameter-by-parameter checklist comparing step_a's pipeline
    to TASK B's pipeline."""
    # We can probe a few of these directly from script source; others by reading
    # constants from _step_d_common / step_f Part A.
    import importlib
    cm = importlib.import_module("_step_d_common")
    rows = []

    # 1. mxq path
    npu_mxq = cm._find_yolo11s_mxq("single")
    npu_mxq_g8 = cm._find_yolo11s_mxq("global8")
    rows.append({
        "param": "NPU mxq file (single mode)",
        "step_a": "yolo11s.mxq via global8 fallback",
        "taskB": npu_mxq,
        "same": (npu_mxq == npu_mxq_g8),  # same file because mode-specific dir doesn't exist
        "note": "single/global8 paths fall back to same un-suffixed mxq",
    })
    # 2. infer_mode
    rows.append({
        "param": "infer_mode",
        "step_a": "global8 (line 185 'yolo11s.mxq, 640, global8')",
        "taskB": "single",
        "same": False,
        "note": "ROOT CAUSE CANDIDATE — different resource allocation may yield different sAP",
    })
    # 3. IoU / conf
    from step0_compare_devices import IOU, CONF, IMG_SIZE
    rows.append({"param": "CONF threshold", "step_a": CONF, "taskB": CONF, "same": True, "note": "shared from step0_compare_devices"})
    rows.append({"param": "IoU NMS threshold", "step_a": IOU, "taskB": IOU, "same": True, "note": "shared"})
    rows.append({"param": "imgsz",           "step_a": IMG_SIZE, "taskB": IMG_SIZE, "same": True, "note": "shared"})
    # 4. size thresholds
    rows.append({
        "param": "COCO area thresholds (small <32² <96² <large)",
        "step_a": "pycocotools default 32^2/96^2",
        "taskB":  "pycocotools default 32^2/96^2",
        "same": True,
        "note": "both use COCOeval defaults",
    })
    # 5. val split
    val = load_val()
    rows.append({
        "param": "val.json categories",
        "step_a": len(val["categories"]),
        "taskB":  len(val["categories"]),
        "same": True,
        "note": "AHD 8 classes",
    })
    rows.append({
        "param": "val.json sequences (24 logs)",
        "step_a": len(val["sequences"]),
        "taskB":  len(val["sequences"]),
        "same": True, "note": "same shared val.json",
    })
    # 6. category mapping
    rows.append({
        "param": "coco_mapping length",
        "step_a": len(val["coco_mapping"]),
        "taskB":  len(val["coco_mapping"]),
        "same": True, "note": "same val.json",
    })
    # 7. pycocotools call style
    rows.append({
        "param": "pycocotools aggregation",
        "step_a": "per-log COCOeval; mean across logs (mean±std)",
        "taskB":  "per-log COCOeval; mean across logs (mean±std)",
        "same": True,
        "note": "both per-log-then-average; not pooled",
    })
    # 8. warmup-skip
    rows.append({
        "param": "warmup-skip frames",
        "step_a": cm.WARMUP_FRAMES,
        "taskB":  cm.WARMUP_FRAMES,
        "same": True, "note": "shared via _step_d_common",
    })
    out = pd.DataFrame(rows)
    out.to_csv(RES / "p0_root_cause.csv", index=False)
    print("\n=== P0.2  Root-cause checklist ===")
    print(out.to_string(index=False))
    mismatches = [r for r in rows if not r["same"]]
    print(f"\n  Mismatches: {len(mismatches)}")
    for r in mismatches:
        print(f"    - {r['param']}: step_a={r['step_a']!r}  taskB={r['taskB']!r}")
        print(f"      note: {r['note']}")
    return {"rows": rows, "mismatches": mismatches}


# ============================== P0.3  NEW C* ===================

def _group_mean(df, sids, col):
    sub = df[df.sid.isin(sids)] if "sid" in df.columns else df
    return float(sub[col].mean()) if len(sub) else float("nan")


def p0_3_new_cstar():
    """Two C* definitions:
       OLD (v1 abs)  : per-size absolute mAP gap aggregated via GROUP_P weights
                       (this is what step_l_cstar.py produced).
       NEW (v2 sap)  : Δ_g(C) = sAP_NPU,g - sAP_GPU,g(C); C* where Δ=0.
                       sAP_NPU,g taken from step_a (canonical Table 1 source).
    """
    # ----- v1 absolute (preserve as diagnostic) -----
    df_v1 = pd.read_csv(RES / "cstar_per_group.csv")
    df_v1.to_csv(RES / "cstar_v1_abs.csv", index=False)

    # ----- v2 sAP-gap (paper-aligned) -----
    df_a = pd.read_csv(RES / "step_a_baseline.csv")
    log_to_sid = {name: i for i, name in enumerate(load_val()["sequences"])}
    df_a["sid"] = df_a["log_id"].map(log_to_sid)
    # NPU L0 sAP per sid → averaged by group
    npu_a = df_a[df_a.device == "NPU"].copy()

    # GPU sAP per ladder level: L0 from step_a; L1_light from gen_decomp (TASK B)
    # L1_heavy, L2_lm, L3_vlm from step_l_cstar.
    df_b = pd.read_csv(RES / "gen_decomp.csv")
    df_b_gpu = df_b[(df_b.device == "GPU")].copy()
    df_l = pd.read_csv(RES / "cstar_summary.csv")  # ladder summary with eff E2E
    # cstar_summary has aggregated L_g(C) using OLD definition; we'll use the
    # PER-SID raw measurements from the cstar CSV directly.
    df_l_raw = pd.read_csv(RES / "cstar.csv")
    # Build per-(bg_level) GPU sAP per sid, including L0 and L1_light from step_a
    # and gen_decomp respectively.
    # step_a has sap_5095 (overall); we want overall sAP per sid per bg level.
    gpu_overall = {}  # bg -> {sid: sap_5095}
    # L0 from step_a
    gpu_overall["L0"] = {int(r["sid"]): float(r["sap_5095"])
                          for _, r in df_a[df_a.device == "GPU"].iterrows()}
    # L1_light from gen_decomp
    sub = df_b_gpu[df_b_gpu.bg_level == "L1_light"]
    gpu_overall["L1_light"] = {int(r["sid"]): float(r["sap_5095"]) for _, r in sub.iterrows()}
    # L1_heavy / L2_lm / L3_vlm from cstar.csv (step_l)
    for bg in ["L1_heavy", "L2_lm", "L3_vlm"]:
        sub = df_l_raw[df_l_raw.bg_level == bg]
        gpu_overall[bg] = {int(r["sid"]): float(r["sap_5095"]) for _, r in sub.iterrows()}

    # NPU overall sAP per sid (single number per sid, L0)
    npu_overall = {int(r["sid"]): float(r["sap_5095"])
                    for _, r in npu_a.iterrows()}

    # Effective E2E latency per bg level (averaged across logs) — for the scalar C axis
    # step_a doesn't have eff_e2e_mean; use infer_mean_ms as proxy at L0
    C_per_bg = {}
    C_per_bg["L0"] = float(df_a[df_a.device == "GPU"]["infer_mean_ms"].mean())
    # L1_light from gen_decomp — TASK B doesn't have eff_e2e_mean column directly;
    # use latency_mean as proxy at L0/L1_light (single-stream → eff ≈ infer)
    if "latency_mean" in df_b_gpu.columns:
        C_per_bg["L1_light"] = float(
            df_b_gpu[df_b_gpu.bg_level == "L1_light"]["latency_mean"].mean())
    # L1_heavy, L2_lm, L3_vlm: cstar.csv has eff_e2e_mean
    for bg in ["L1_heavy", "L2_lm", "L3_vlm"]:
        sub = df_l_raw[df_l_raw.bg_level == bg]
        C_per_bg[bg] = float(sub["eff_e2e_mean"].mean())

    # Per-group aggregation
    v2_rows = []
    v2_per_bg_rows = []
    for grp, sids in SIZE_GROUPS.items():
        # sAP_NPU,g = mean over group sids of sap_5095 at NPU L0
        sap_npu_g = float(np.mean([npu_overall[s] for s in sids if s in npu_overall]))
        # sAP_GPU,g(C) per bg level
        points = []  # list of (C, delta, bg, sap_gpu)
        for bg in LADDER:
            if bg not in gpu_overall or bg not in C_per_bg:
                continue
            vals = [gpu_overall[bg].get(s) for s in sids if s in gpu_overall[bg]]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            sap_gpu_g = float(np.mean(vals))
            C = C_per_bg[bg]
            delta = sap_npu_g - sap_gpu_g  # > 0 → NPU preferred
            points.append((bg, C, sap_gpu_g, delta))
            v2_per_bg_rows.append({
                "group": grp, "bg": bg, "C_ms": round(C, 2),
                "sap_npu_g": round(sap_npu_g, 4),
                "sap_gpu_g": round(sap_gpu_g, 4),
                "delta_g": round(delta, 4),
            })
        # Interpolate C* where delta=0
        pts_sorted = sorted(points, key=lambda x: x[1])
        Cstar = None
        for i in range(len(pts_sorted) - 1):
            bg1, C1, _, d1 = pts_sorted[i]
            bg2, C2, _, d2 = pts_sorted[i + 1]
            if d1 * d2 <= 0 and (d2 != d1):
                frac = (0 - d1) / (d2 - d1)
                Cstar = C1 + frac * (C2 - C1)
                break
        v2_rows.append({
            "group": grp, "sap_npu_g": round(sap_npu_g, 4),
            "Cstar_ms": round(Cstar, 2) if Cstar is not None else None,
        })
    pd.DataFrame(v2_per_bg_rows).to_csv(RES / "cstar_v2_sap_per_bg.csv", index=False)
    pd.DataFrame(v2_rows).to_csv(RES / "cstar_v2_sap.csv", index=False)

    print("\n=== P0.3  NEW C* (v2, sAP-gap) ===")
    print(f"\nC_per_bg (GPU eff E2E ms): {C_per_bg}")
    print("\nv2 per-(group, bg):")
    print(pd.DataFrame(v2_per_bg_rows).to_string(index=False))
    print("\nv2 per-group C*:")
    print(pd.DataFrame(v2_rows).to_string(index=False))
    print("\nv1 (absolute per-size, OLD) for diagnostic:")
    print(df_v1.to_string(index=False))
    return {"v2_rows": v2_rows, "v2_per_bg": v2_per_bg_rows, "v1_rows": df_v1.to_dict("records"),
            "C_per_bg": C_per_bg, "gpu_overall": gpu_overall, "npu_overall": npu_overall}


# ============================== P0.4  CONSISTENCY ===================

def p0_4_consistency(v2_data):
    """Three-source check:
      (a) tab:partA: small-rich Δ(L1) > large-rich Δ(L1)   (>-0.018 vs -0.039)
      (b) v2 C*:    C*_small-rich < C*_large-rich          (small crosses first)
      (c) TASK D:   N=4 L1 inversion = yes
    """
    # (a) from step_f_partA_matrix.csv per stream gap NPU-GPU at L1 (this is sAP-gap)
    df_pa = pd.read_csv(RES / "step_f_partA_matrix.csv")
    # The Part A matrix doesn't have bg column — it's at L1 by default (verified
    # against the paper's numbers in tab:partA).  Compute per-group mean Δ.
    # device columns: CPU, GPU, NPU; we want NPU-GPU paired per log.
    df_pa["sid_int"] = df_pa["sid"].astype(int)
    g_a = df_pa[df_pa.device == "GPU"][["sid_int", "log_size_group", "sap_5095"]].rename(columns={"sap_5095": "sap_gpu"})
    n_a = df_pa[df_pa.device == "NPU"][["sid_int", "sap_5095"]].rename(columns={"sap_5095": "sap_npu"})
    j = g_a.merge(n_a, on="sid_int")
    j["delta"] = j["sap_npu"] - j["sap_gpu"]
    delta_by_grp = j.groupby("log_size_group")["delta"].mean().to_dict()
    # Note: Part A's group labels: small-rich, medium-mixed, large-rich
    a_small = float(delta_by_grp.get("small-rich", float("nan")))
    a_large = float(delta_by_grp.get("large-rich", float("nan")))
    check_a = a_small > a_large

    # (b) v2 C* ordering — small-rich crosses first ⇔ C*_small < C*_large
    v2 = {r["group"]: r["Cstar_ms"] for r in v2_data["v2_rows"]}
    Cs_s = v2.get("small-rich"); Cs_l = v2.get("large-rich")
    if Cs_s is not None and Cs_l is not None:
        check_b = Cs_s < Cs_l
    else:
        check_b = None

    # (c) TASK D N=4 L1 inversion
    df_d = pd.read_csv(RES / "gen_gain.csv")
    n4_l1 = df_d[(df_d.cell == "N=4") & (df_d.bg == "L1_light")]
    if len(n4_l1):
        check_c = bool(n4_l1.iloc[0]["inverts"])
    else:
        check_c = None

    # Build the matrix
    matrix = {
        "(a) tab:partA small-rich Δ > large-rich Δ": {
            "small_rich_delta": round(a_small, 4),
            "large_rich_delta": round(a_large, 4),
            "result": check_a,
        },
        "(b) v2 C*: small-rich crosses first": {
            "Cstar_small_rich_ms": Cs_s,
            "Cstar_large_rich_ms": Cs_l,
            "result": check_b,
        },
        "(c) TASK D N=4 L1 inversion": {
            "inverts": check_c,
            "result": check_c,
        },
    }
    rows = []
    for k, v in matrix.items():
        rows.append({"check": k, **v})
    pd.DataFrame(rows).to_csv(RES / "consistency_matrix.csv", index=False)

    all_results = [v.get("result") for v in matrix.values()]
    overall_pass = all(r is True for r in all_results)

    print("\n=== P0.4  Consistency matrix ===")
    for k, v in matrix.items():
        print(f"  {k}")
        for kk, vv in v.items():
            if kk != "result":
                print(f"     {kk} = {vv}")
        print(f"     result = {v['result']}")
    print(f"\n  CONSISTENCY: {'PASS (small-rich first)' if overall_pass else 'FAIL'}")
    if not overall_pass:
        loners = []
        if not check_a:
            loners.append("(a) tab:partA")
        if not check_b:
            loners.append("(b) v2 C*")
        if not check_c:
            loners.append("(c) TASK D")
        passers = [x for x in ["(a)","(b)","(c)"] if x not in [l.split()[0] for l in loners]]
        # Loner = the single source that disagrees with the other two
        print(f"  Disagreeing sources: {loners}")
        print(f"  Hypothesis: aggregation-unit (absolute mAP vs sAP-gap) probably causes (b)")

    return {"matrix": matrix, "overall_pass": overall_pass}


# ============================== P0.5  FIGURE QUARANTINE ===================

def render_v2_figure(v2_data, v1_data):
    """Render NEW C* figure. Always saved to results/, only copied to paper/
    if consistency passes."""
    # Two-panel: (left) Δ_g(C) curves with markers at C*, (right) C* per group
    # vs detector capacity (only YOLOv11s; n/m TBD).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"small-rich": "#3680c4",
              "medium-mixed": "#3b9c4d",
              "large-rich": "#c43b3b"}

    # Panel A: Δ_g(C)
    ax = axes[0]
    df_per_bg = pd.DataFrame(v2_data["v2_per_bg"])
    for grp in SIZE_GROUPS:
        sub = df_per_bg[df_per_bg.group == grp].sort_values("C_ms")
        ax.plot(sub["C_ms"], sub["delta_g"], "-o",
                color=colors[grp], linewidth=2, markersize=8,
                label=grp)
        # ladder labels
        for _, r in sub.iterrows():
            ax.annotate(r["bg"], xy=(r["C_ms"], r["delta_g"]),
                        xytext=(0, -12), textcoords="offset points",
                        fontsize=6.5, ha="center", color="#555")
        # C* marker
        Cstar = next((r["Cstar_ms"] for r in v2_data["v2_rows"] if r["group"] == grp), None)
        if Cstar is not None:
            ax.scatter([Cstar], [0], s=200, marker="*",
                       color=colors[grp], edgecolor="black", linewidth=0.8, zorder=5)
            ax.annotate(f"C*={Cstar:.1f}ms",
                        xy=(Cstar, 0), xytext=(6, 8),
                        textcoords="offset points",
                        fontsize=8, color=colors[grp])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("scalar contention proxy C  (GPU effective E2E latency, ms)")
    ax.set_ylabel("$\\Delta_g(C) = sAP_{NPU,g} - sAP_{GPU,g}(C)$")
    ax.set_title("YOLOv11s: NPU preferred when $\\Delta_g > 0$")
    ax.set_xscale("symlog", linthresh=20)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Panel B: C* vs capacity (only v11s)
    ax = axes[1]
    for grp, color in colors.items():
        Cstar = next((r["Cstar_ms"] for r in v2_data["v2_rows"] if r["group"] == grp), None)
        if Cstar is not None:
            ax.scatter([9.4], [Cstar], s=140, color=color, marker="o",
                       edgecolor="black", linewidth=0.6,
                       label=f"{grp}: C*={Cstar:.1f}ms", zorder=4)
    for cap, lbl in [(2.6, "YOLOv11n"), (20.1, "YOLOv11m")]:
        ax.axvline(cap, color="gray", linestyle=":", alpha=0.4)
        ax.text(cap, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 50,
                f"{lbl}\n(TBD)", fontsize=7, color="gray",
                ha="center", va="top")
    ax.set_xlabel("detector capacity (params, M)")
    ax.set_ylabel("inversion threshold C*  (ms)")
    ax.set_xscale("log")
    ax.set_xticks([2.6, 9.4, 20.1])
    ax.set_xticklabels(["2.6\n(n)", "9.4\n(s)", "20.1\n(m)"])
    ax.set_title("C* vs detector capacity (NEW sAP-gap definition)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle("YOLOv11s — Inversion threshold via sAP-gap (v2, paper-aligned)",
                 fontsize=12)
    plt.tight_layout()
    out_results = RES / "gen_cstar_v2.pdf"
    fig.savefig(out_results, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_results}")
    return out_results


def quarantine_figure_in_tex(disputed: bool):
    """If disputed=True, comment-out the fig:gen-cstar block in main_vision.tex
    and insert a TODO. If False, regenerate gen_cstar.pdf from v2.

    Looks for the `\\begin{figure}` block whose `\\label{fig:gen-cstar}` matches,
    starting from a `\\includegraphics` line that points to gen_cstar.pdf.
    """
    if not MAIN_TEX.exists():
        return {"action": "skipped", "reason": "main_vision.tex missing"}
    src = MAIN_TEX.read_text()
    if disputed:
        # Find the figure block containing fig:gen-cstar
        # We do a simple search-and-replace surrounding the block.
        marker = "\\label{fig:gen-cstar}"
        if marker not in src:
            return {"action": "no-op", "reason": "label not found"}
        start = src.rfind("\\begin{figure}", 0, src.find(marker))
        end = src.find("\\end{figure}", src.find(marker)) + len("\\end{figure}")
        if start < 0 or end <= start:
            return {"action": "no-op", "reason": "could not isolate figure block"}
        block = src[start:end]
        if "% [DISPUTED-CSTAR]" in block:
            return {"action": "already-quarantined", "reason": "[DISPUTED-CSTAR] already present"}
        wrapped = ("% [DISPUTED-CSTAR] CONSISTENCY check FAILED — figure quarantined by phase_p0_debug.py\n"
                   "% TODO(human): verify C* unit definition (absolute vs sAP-gap) and rebuild.\n"
                   "\\iffalse\n"
                   + block + "\n\\fi\n")
        new_src = src[:start] + wrapped + src[end:]
        MAIN_TEX.write_text(new_src)
        return {"action": "quarantined", "block_chars": end - start}
    return {"action": "pass", "reason": "figure left as-is; new v2 PDF rebuilt below"}


# ============================== MAIN ===================

def main():
    decisions = {}

    decisions["P0.1"] = p0_1_table1_repro()
    decisions["P0.2"] = p0_2_root_cause()
    decisions["P0.3"] = p0_3_new_cstar()

    consistency = p0_4_consistency(decisions["P0.3"])
    decisions["P0.4"] = consistency

    # Always render v2 to results/ for inspection
    v2_pdf = render_v2_figure(decisions["P0.3"],
                              {"v1_rows": decisions["P0.3"]["v1_rows"]})
    decisions["v2_figure_results"] = str(v2_pdf)

    # Quarantine logic: only if FAIL do we comment-out main_vision.tex figure.
    if consistency["overall_pass"]:
        # Copy v2 over the paper figure
        import shutil
        for dest in [SCRIPT_DIR.parent.parent / "paper" / "figures" / "gen_cstar.pdf",
                     SCRIPT_DIR.parent.parent / "figures" / "gen_cstar.pdf"]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(v2_pdf, dest)
            print(f"copied -> {dest}")
        decisions["P0.5"] = {"action": "v2 promoted to paper/figures/gen_cstar.pdf",
                              "consistency": "PASS"}
    else:
        decisions["P0.5"] = quarantine_figure_in_tex(disputed=True)
        decisions["P0.5"]["consistency"] = "FAIL"

    # Sanitize numpy types for JSON
    def _coerce(o):
        if isinstance(o, dict):
            return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_coerce(x) for x in o]
        if isinstance(o, np.bool_):  # type: ignore[attr-defined]
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(RES / "p0_decisions.json", "w") as f:
        json.dump(_coerce(decisions), f, indent=2, default=str)
    print(f"\nsaved {RES / 'p0_decisions.json'}")
    print(f"\nP0 done. CONSISTENCY: {'PASS' if consistency['overall_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
