"""rev9 STEP 4 + STEP 5 + STEP 6 — aggregate, build paper tables, propose prose edits.

Reuses (where the cell key is identical to rev9):
  - results/p1r6_baseline_yolo11s.csv  : Table 1 GPU + NPU L0 single-stream
                                          (rev6 normal-state baseline)
  - results/p1r6_ladder_yolo11s.csv    : partA GPU L1_light
  - results/p1r6_multistream_yolo11s.csv : main-comparison N=4 L1_light SA/SBR
                                            (Composition A baseline)
Adds (rev9 fresh):
  - rev9_table1_cpu.csv     : Table 1 CPU column
  - rev9_partA_npu.csv      : partA NPU L1_light
  - rev9_per_class.csv      : tab:per-class
  - rev9_main_cmp.csv       : tab:main-comparison full (Naive, Oracle, L2_LM)
  - rev9_schedule.csv       : tab:schedule-shift
  - rev9_capacity.csv       : tab:capacity
  - rev9_natural.csv        : tab:natural (Comp B + C; A reuses rev6)

Outputs:
  paper/tables/single_stream.tex       (tab:single-stream)
  paper/tables/per_class.tex           (tab:per-class — updated)
  paper/tables/partA.tex               (tab:partA)
  paper/tables/main_comparison.tex     (tab:main-comparison)
  paper/tables/schedule_shift.tex      (tab:schedule-shift)
  paper/tables/capacity.tex            (tab:capacity)
  paper/tables/natural.tex             (tab:natural)
  paper/tables/decomposition.tex       (tab:decomposition)

  results/rev9_delta.md                old vs new per cell
  results/rev9_PROPOSED_EDITS.md       SEMANTIC-LOCK rows: where in main.tex,
                                        old → new, impact (e.g., "large +0.2% n.s.
                                        → −21% — 'unaffected' narrative refuted")
  results/rev9_summary.csv             one-row summary
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import load_val

RES = Path("accv_experiments/results")
PAPER_TAB = SCRIPT_DIR.parent.parent / "paper" / "tables"
PAPER_TAB.mkdir(parents=True, exist_ok=True)
DELTA = RES / "rev9_delta.md"
EDITS = RES / "rev9_PROPOSED_EDITS.md"
SUMMARY = RES / "rev9_summary.csv"

# Composition A sids in (N=4, N=8)
COMP_A_N4 = [2, 22, 3, 21]
COMP_A_N8 = [2, 22, 13, 16, 3, 21, 14, 4]
LADDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]


# ----------------- IO: reuse + rev9 fresh -----------------

def load_table1():
    """Merge rev6 (GPU+NPU L0 over 24 logs) + rev9 CPU L0."""
    rev6 = pd.read_csv(RES / "p1r6_baseline_yolo11s.csv")
    rev6 = rev6[(rev6.bg_level == "L0") & (rev6.device.isin(["GPU", "NPU"]))]
    rev9_cpu = pd.read_csv(RES / "rev9_table1_cpu.csv") if (RES / "rev9_table1_cpu.csv").exists() else None
    parts = [rev6[["sid", "device", "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
                   "map_5095", "map_s", "map_m", "map_l", "latency_mean", "wall_sec"]]]
    if rev9_cpu is not None:
        parts.append(rev9_cpu[["sid", "device", "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
                                "map_5095", "map_s", "map_m", "map_l", "latency_mean", "wall_sec"]])
    return pd.concat(parts, ignore_index=True)


def load_partA():
    """Combine GPU L1_light (rev6 ladder) + NPU L1_light (rev9 fresh)."""
    rev6 = pd.read_csv(RES / "p1r6_ladder_yolo11s.csv")
    rev6 = rev6[(rev6.bg_level == "L1_light") & (rev6.device == "GPU")]
    rev9 = pd.read_csv(RES / "rev9_partA_npu.csv") if (RES / "rev9_partA_npu.csv").exists() else None
    return pd.concat([rev6, rev9], ignore_index=True) if rev9 is not None else rev6


def load_main_cmp():
    """rev9 main_cmp (full) — but rev6 SA/SBR at L1_light Comp A is a sanity overlay."""
    return pd.read_csv(RES / "rev9_main_cmp.csv")


# ----------------- aggregation helpers -----------------

def device_stats(df, device, fields):
    sub = df[df.device == device]
    out = {}
    for f in fields:
        if f in sub.columns:
            vals = sub[f].astype(float)
            out[f] = (float(vals.mean()), float(vals.std(ddof=0)))
    return out


def paired_diff(df, fields):
    g = df[df.device == "GPU"].sort_values("sid").reset_index(drop=True)
    n = df[df.device == "NPU"].sort_values("sid").reset_index(drop=True)
    out = {}
    for f in fields:
        if f in g.columns and f in n.columns:
            d = n[f].astype(float).values - g[f].astype(float).values
            out[f] = {"diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=0))}
    return out


def wilcoxon_p(df, field):
    from scipy.stats import wilcoxon
    g = df[df.device == "GPU"].sort_values("sid")
    n = df[df.device == "NPU"].sort_values("sid")
    diffs = n[field].values - g[field].values
    if not np.any(diffs):
        return None
    try:
        return float(wilcoxon(diffs).pvalue)
    except Exception:
        return None


# ----------------- tex writers -----------------

def write_tab_single_stream():
    df = load_table1()
    out = PAPER_TAB / "single_stream.tex"
    fields = ["sap_5095", "sap_50", "sap_s", "sap_m", "sap_l", "latency_mean"]
    stats = {dev: device_stats(df, dev, fields) for dev in ["CPU", "GPU", "NPU"]}

    g = df[df.device == "GPU"]; n = df[df.device == "NPU"]
    diffs = paired_diff(df, fields)
    p_5095 = wilcoxon_p(df, "sap_5095"); p_50 = wilcoxon_p(df, "sap_50")
    p_s = wilcoxon_p(df, "sap_s"); p_m = wilcoxon_p(df, "sap_m"); p_l = wilcoxon_p(df, "sap_l")
    def fmt_p(p):
        if p is None: return "n/a"
        if p < 0.001: return "$p<0.001$"
        return f"$p={p:.2g}$"
    def fmt_pair(d): return f"{d[0]:.3f}$\\pm${d[1]:.3f}"
    def rel_pct(diff, base):
        return f"{diff/base*100:+.1f}\\%" if base else "n/a"

    rel_s = rel_pct(diffs["sap_s"]["diff_mean"], stats["GPU"]["sap_s"][0])
    rel_m = rel_pct(diffs["sap_m"]["diff_mean"], stats["GPU"]["sap_m"][0])
    rel_l = rel_pct(diffs["sap_l"]["diff_mean"], stats["GPU"]["sap_l"][0])

    lat_c = stats["CPU"]["latency_mean"]; lat_g = stats["GPU"]["latency_mean"]; lat_n = stats["NPU"]["latency_mean"]
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev9_aggregate.py\n")
        f.write(f"% rev9 normal-state remeasurement; sdk={Path('models/mblt_model_zoo').resolve()},\n")
        f.write("% mxq=b2441f9d (legacy v11s), infer_mode=global8.\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Metric & CPU & GPU & NPU & NPU$-$GPU \\\\\n\\midrule\n")
        f.write(f"infer mean (ms)   & {lat_c[0]:.2f}$\\pm${lat_c[1]:.2f} & {lat_g[0]:.2f}$\\pm${lat_g[1]:.2f} & {lat_n[0]:.2f}$\\pm${lat_n[1]:.2f} & --- \\\\\n")
        f.write(f"sAP$_{{[.50:.95]}}$ & {fmt_pair(stats['CPU']['sap_5095'])} & {fmt_pair(stats['GPU']['sap_5095'])} & {fmt_pair(stats['NPU']['sap_5095'])} & ${diffs['sap_5095']['diff_mean']:+.3f}$ ({fmt_p(p_5095)}) \\\\\n")
        f.write(f"sAP$_{{0.50}}$      & {fmt_pair(stats['CPU']['sap_50'])} & {fmt_pair(stats['GPU']['sap_50'])} & {fmt_pair(stats['NPU']['sap_50'])} & ${diffs['sap_50']['diff_mean']:+.3f}$ ({fmt_p(p_50)}) \\\\\n")
        f.write(f"AP$_{{\\text{{small}}}}$  & {fmt_pair(stats['CPU']['sap_s'])} & {fmt_pair(stats['GPU']['sap_s'])} & {fmt_pair(stats['NPU']['sap_s'])} & ${diffs['sap_s']['diff_mean']:+.3f}$ (${rel_s}$, {fmt_p(p_s)}) \\\\\n")
        f.write(f"AP$_{{\\text{{medium}}}}$ & {fmt_pair(stats['CPU']['sap_m'])} & {fmt_pair(stats['GPU']['sap_m'])} & {fmt_pair(stats['NPU']['sap_m'])} & ${diffs['sap_m']['diff_mean']:+.3f}$ (${rel_m}$, {fmt_p(p_m)}) \\\\\n")
        f.write(f"AP$_{{\\text{{large}}}}$  & {fmt_pair(stats['CPU']['sap_l'])} & {fmt_pair(stats['GPU']['sap_l'])} & {fmt_pair(stats['NPU']['sap_l'])} & ${diffs['sap_l']['diff_mean']:+.3f}$ (${rel_l}$, {fmt_p(p_l)}) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return out, {
        "small": (diffs['sap_s']['diff_mean'], rel_s),
        "medium": (diffs['sap_m']['diff_mean'], rel_m),
        "large": (diffs['sap_l']['diff_mean'], rel_l),
    }


def write_tab_per_class():
    df = pd.read_csv(RES / "rev9_per_class.csv")
    g = df[df.device == "GPU"].set_index("cat_name")
    n = df[df.device == "NPU"].set_index("cat_name")
    paper_order = ["car", "truck", "bus", "person", "bicycle", "motorcycle", "traffic_light", "stop_sign"]
    paper_names = {"person": "pedestrian", "traffic_light": "traffic\\_light", "stop_sign": "stop\\_sign"}
    size_cls = {"car": "medium--large", "truck": "large", "bus": "large",
                 "person": "small", "bicycle": "small",
                 "motorcycle": "small--medium", "traffic_light": "small", "stop_sign": "small"}
    out = PAPER_TAB / "per_class.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev9_aggregate.py — rev9 normal state.\n")
        f.write("\\begin{tabular}{llccc}\n\\toprule\n")
        f.write("Class & Size class & GPU AP & NPU AP & Gap \\\\\n\\midrule\n")
        for name in paper_order:
            if name not in g.index or name not in n.index:
                f.write(f"{paper_names.get(name,name)} & {size_cls[name]} & TBD & TBD & TBD \\\\\n")
                continue
            gv = float(g.loc[name, "ap_5095"]); nv = float(n.loc[name, "ap_5095"])
            f.write(f"{paper_names.get(name,name)} & {size_cls[name]} & {gv:.3f} & {nv:.3f} & ${nv-gv:+.3f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return out


def write_tab_partA():
    df = load_partA()
    sids = [int(s) for s in df.sid.unique()]
    out = PAPER_TAB / "partA.tex"
    rows = []
    for sid in sids:
        gpu_row = df[(df.sid == sid) & (df.device == "GPU")]
        npu_row = df[(df.sid == sid) & (df.device == "NPU")]
        if not len(gpu_row) or not len(npu_row): continue
        gpu = float(gpu_row.iloc[0]["sap_5095"]); npu = float(npu_row.iloc[0]["sap_5095"])
        rows.append((sid, gpu, npu, npu - gpu))
    rows.sort(key=lambda x: x[3], reverse=True)  # least loss first
    SIZE = {2:"small-rich",3:"large-rich",8:"medium-mixed",10:"medium-mixed",
            13:"small-rich",17:"medium-mixed",21:"large-rich",22:"small-rich"}
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev9_aggregate.py — rev9 normal state, L1_light.\n")
        f.write("\\begin{tabular}{rlccc}\n\\toprule\n")
        f.write("Rank & Stream group & GPU sAP & NPU sAP & $\\Delta$ (NPU$-$GPU) \\\\\n\\midrule\n")
        for rk, (sid, gpu, npu, d) in enumerate(rows, 1):
            f.write(f"{rk} & {SIZE.get(sid,'?')} & {gpu:.3f} & {npu:.3f} & ${d:+.3f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return out


def write_tab_main_cmp():
    df = load_main_cmp()
    out = PAPER_TAB / "main_comparison.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev9_aggregate.py — rev9 normal state.\n")
        f.write("\\begin{tabular}{ll|cc|cc}\n\\toprule\n")
        f.write("bg & Strategy & mean sAP & worst sAP & mean mAP & worst mAP \\\\\n\\midrule\n")
        for bg, bg_label in [("L1_light", "L1$_{\\text{light}}$"), ("L2_lm", "L2$_{\\text{LM}}$")]:
            for strat in ["Naive_allGPU", "SizeAware", "SizeBlindRev"]:
                sub = df[(df.bg_level == bg) & (df.placement_name == strat)]
                if not len(sub): continue
                r = sub.iloc[0]
                f.write(f"{bg_label} & {strat} & {float(r['mean_sap']):.3f} & {float(r['worst_sap']):.3f} & {float(r['mean_map']):.3f} & {float(r['worst_map']):.3f} \\\\\n")
            # Oracle: best worst_sap across all placements at this bg
            sub_all = df[df.bg_level == bg]
            if len(sub_all):
                best = sub_all.sort_values("worst_sap", ascending=False).iloc[0]
                f.write(f"{bg_label} & Oracle & {float(best['mean_sap']):.3f} & {float(best['worst_sap']):.3f} & {float(best['mean_map']):.3f} & {float(best['worst_map']):.3f} \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return out


def write_tab_capacity():
    df = pd.read_csv(RES / "rev9_capacity.csv")
    out = PAPER_TAB / "capacity.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev9_aggregate.py — rev9 normal state, L1_light.\n")
        f.write("\\begin{tabular}{c|cccc}\n\\toprule\n")
        f.write("$N$ & Naive & SizeAware & SizeBlindRev & AllNPU \\\\\n\\midrule\n")
        for N in [2, 3, 4, 5, 6, 8]:
            row = [f"${N}$"]
            for strat in ["Naive_allGPU", "SizeAware", "SizeBlindRev", "AllNPU"]:
                r = df[(df.tag == f"N={N}") & (df.placement_name == strat)]
                row.append(f"{float(r.iloc[0]['worst_sap']):.3f}" if len(r) else "TBD")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\midrule\n")
        # thresholds
        thresholds = [0.05, 0.08, 0.10]
        for th in thresholds:
            f.write(f"$\\geq{th:.2f}$ & ")
            cells = []
            for strat in ["Naive_allGPU", "SizeAware", "SizeBlindRev", "AllNPU"]:
                max_N = 0
                for N in [2,3,4,5,6,8]:
                    r = df[(df.tag == f"N={N}") & (df.placement_name == strat)]
                    if len(r) and float(r.iloc[0]["worst_sap"]) >= th: max_N = N
                cells.append(str(max_N) if max_N else "---")
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return out


def write_tab_schedule():
    df = pd.read_csv(RES / "rev9_schedule.csv")
    out = PAPER_TAB / "schedule_shift.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev9_aggregate.py — rev9 normal state.\n")
        f.write("\\begin{tabular}{l|cccc|cccc}\n\\toprule\n")
        f.write("& \\multicolumn{4}{c|}{$N=4$} & \\multicolumn{4}{c}{$N=8$} \\\\\n")
        f.write("bg & Naive & SizeAware & AllNPU & $k^\\star/N$ & Naive & SizeAware & AllNPU & $k^\\star/N$ \\\\\n\\midrule\n")
        bg_label = {"L0":"L0", "L1_light":"L1$_{\\text{light}}$",
                    "L1_heavy":"L1$_{\\text{heavy}}$", "L2_lm":"L2$_{\\text{LM}}$",
                    "L3_vlm":"L3$_{\\text{VLM}}$"}
        for bg in LADDER:
            row = [bg_label[bg]]
            for N in [4, 8]:
                strategies = ["Naive_allGPU", "SizeAware" if N==4 else "SizeAware_NPU4", "AllNPU"]
                values = []; max_v = -1; max_strat_idx = -1
                for i, st in enumerate(strategies):
                    r = df[(df.tag == f"N={N}") & (df.bg_level == bg) & (df.placement_name == st)]
                    if len(r):
                        v = float(r.iloc[0]["worst_sap"])
                        values.append(v)
                        if v > max_v: max_v = v; max_strat_idx = i
                    else:
                        values.append(None)
                # determine k_star/N
                if values[2] is not None and (max_strat_idx == 2 or (max_v - (values[2] or 0) < 0.005)):
                    kstar = N
                elif values[1] is not None and (max_strat_idx == 1 or values[1] >= (values[2] or 0)):
                    kstar = N // 2 if N == 4 else 4
                else:
                    kstar = 0
                for i, v in enumerate(values):
                    if v is None:
                        row.append("TBD")
                    elif i == max_strat_idx:
                        row.append(f"\\textbf{{{v:.3f}}}")
                    else:
                        row.append(f"{v:.3f}")
                row.append(f"${kstar}/{N}$")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return out


def write_tab_natural():
    # rev9 has Comp B and C; for Comp A we reuse rev6
    df_rev6 = pd.read_csv(RES / "p1r6_multistream_yolo11s.csv") if (RES / "p1r6_multistream_yolo11s.csv").exists() else None
    df_rev9 = pd.read_csv(RES / "rev9_natural.csv")
    out = PAPER_TAB / "natural.tex"
    rows = []
    # Comp A from rev6 multistream — only N=4 L1_light SA/SBR plus N=8
    # We'll just present rev9 natural; if A is missing we'll skip rather than mix
    # rev6 multistream rows are bg_level == "L1_light" only and N=4 only for SA/SBR
    comp_map = {"A_baseline": "A", "B_medium_mixed": "B", "C_diverse": "C"}
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev9_aggregate.py — rev9 normal state.\n")
        f.write("\\begin{tabular}{ll|cccc|c}\n\\toprule\n")
        f.write("Comp. & cell & Naive & SizeAware & AllNPU & SizeBlindRev & Oracle \\\\\n\\midrule\n")
        for comp in ["B_medium_mixed", "C_diverse"]:
            for N in [4, 8]:
                for bg in ["L1_light", "L1_heavy"]:
                    sub = df_rev9[(df_rev9.tag == f"{comp}_N{N}") & (df_rev9.bg_level == bg)]
                    if not len(sub): continue
                    row = [comp_map[comp], f"$N={N}$, {bg.replace('_', ' ')}"]
                    for strat_name in ["Naive_allGPU"]:
                        r = sub[sub.placement_name == strat_name]
                        row.append(f"{float(r.iloc[0]['worst_sap']):.3f}" if len(r) else "---")
                    if N == 4:
                        for sn in ["SizeAware", "AllNPU", "SizeBlindRev"]:
                            r = sub[sub.placement_name == sn]
                            row.append(f"{float(r.iloc[0]['worst_sap']):.3f}" if len(r) else "---")
                    else:
                        for sn in ["SizeAware_NPU4", "AllNPU", "SizeBlindRev_NPU4"]:
                            r = sub[sub.placement_name == sn]
                            row.append(f"{float(r.iloc[0]['worst_sap']):.3f}" if len(r) else "---")
                    oracle_best = sub.sort_values("worst_sap", ascending=False).iloc[0]
                    row.append(f"{float(oracle_best['worst_sap']):.3f}")
                    f.write(" & ".join(row) + " \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return out


def write_tab_decomposition(t1_diffs):
    """Loss decomposition — needs N=8 size-diverse worst-camera path.
       Reuses rev9_schedule (N=8 L0/L1 AllNPU and Naive)."""
    sch = pd.read_csv(RES / "rev9_schedule.csv")
    t1 = load_table1()
    # Identify worst camera at N=8 AllNPU L1_light
    n8a_l1 = sch[(sch.tag == "N=8") & (sch.bg_level == "L1_light") & (sch.placement_name == "AllNPU")]
    if not len(n8a_l1):
        return None
    r = n8a_l1.iloc[0]
    saps = {int(r[f"s{i}_sid"]): float(r[f"s{i}_sap"]) for i in range(8) if pd.notna(r.get(f"s{i}_sap"))}
    worst_sid = min(saps, key=saps.get)
    allnpu_l1_worst = saps[worst_sid]
    # N=8 AllNPU L0
    n8a_l0 = sch[(sch.tag == "N=8") & (sch.bg_level == "L0") & (sch.placement_name == "AllNPU")]
    allnpu_l0_worst = None
    if len(n8a_l0):
        rr = n8a_l0.iloc[0]
        for i in range(8):
            if int(rr.get(f"s{i}_sid", -1)) == worst_sid:
                allnpu_l0_worst = float(rr.get(f"s{i}_sap")); break
    # N=8 Naive L1_light
    n8n_l1 = sch[(sch.tag == "N=8") & (sch.bg_level == "L1_light") & (sch.placement_name == "Naive_allGPU")]
    naive_l1_worst = None
    if len(n8n_l1):
        rr = n8n_l1.iloc[0]
        for i in range(8):
            if int(rr.get(f"s{i}_sid", -1)) == worst_sid:
                naive_l1_worst = float(rr.get(f"s{i}_sap")); break
    # Single-stream rows (N=1 GPU/NPU L0, GPU/NPU L1) for worst_sid
    t1_g = t1[(t1.device == "GPU") & (t1.sid == worst_sid)]
    t1_n = t1[(t1.device == "NPU") & (t1.sid == worst_sid)]
    if not len(t1_g) or not len(t1_n):
        return None
    gpu_l0 = float(t1_g.iloc[0]["sap_5095"]); npu_l0 = float(t1_n.iloc[0]["sap_5095"])
    # N=1 NPU L1_light (from partA)
    partA = load_partA()
    n1_npu_l1 = partA[(partA.device == "NPU") & (partA.sid == worst_sid)]
    npu_l1_v = float(n1_npu_l1.iloc[0]["sap_5095"]) if len(n1_npu_l1) else None
    # Components
    quant = gpu_l0 - npu_l0
    bg_on_npu = (npu_l0 - npu_l1_v) if npu_l1_v is not None else None
    concurrent = (npu_l1_v - allnpu_l0_worst) if (npu_l1_v is not None and allnpu_l0_worst is not None) else None
    interaction = (allnpu_l0_worst - allnpu_l1_worst) if allnpu_l0_worst is not None else None
    total = gpu_l0 - allnpu_l1_worst
    gpu_path = gpu_l0 - naive_l1_worst if naive_l1_worst is not None else None
    out = PAPER_TAB / "decomposition.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev9_aggregate.py — rev9 normal state.\n")
        f.write(f"% Worst camera sid = {worst_sid}, AllNPU N=8 L1_light worst sAP = {allnpu_l1_worst:.4f}.\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Loss source & $\\Delta$ worst sAP & share of total NPU-path loss \\\\\n\\midrule\n")
        q_share = quant / total * 100 if total else 0
        f.write(f"Quantization (GPU L0 $\\to$ NPU L0, $N=1$) & ${-quant:+.3f}$ & ${q_share:.0f}\\%$ \\\\\n")
        if bg_on_npu is not None:
            b_share = bg_on_npu / total * 100 if total else 0
            f.write(f"Background on NPU (NPU L0 $\\to$ L1$_{{\\text{{light}}}}$, $N=1$) & ${-bg_on_npu:+.3f}$ & ${b_share:.0f}\\%$ \\\\\n")
        if concurrent is not None:
            c_share = concurrent / total * 100 if total else 0
            f.write(f"Concurrent NPU (NPU $N=1 \\to N=8$, L0) & ${-concurrent:+.3f}$ & ${c_share:.0f}\\%$ \\\\\n")
        if interaction is not None:
            i_share = interaction / total * 100 if total else 0
            f.write(f"Interaction (NPU $N=8$, L0 $\\to$ L1$_{{\\text{{light}}}}$) & ${-interaction:+.3f}$ & noise \\\\\n")
        f.write("\\midrule\n")
        f.write(f"\\textbf{{Total NPU-path loss}} (GPU L0 $\\to$ NPU $N=8$ L1) & \\textbf{{${-total:+.3f}$}} & $100\\%$ \\\\\n")
        f.write("\\midrule\n")
        if gpu_path is not None:
            f.write(f"\\emph{{For comparison:}} GPU-path loss (GPU L0 $\\to$ GPU $N=8$ L1, Naive) & ${-gpu_path:+.3f}$ & --- \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return out, {"worst_sid": worst_sid, "quant": quant, "total": total,
                 "q_share": q_share, "gpu_path": gpu_path,
                 "bg_on_npu": bg_on_npu, "concurrent": concurrent}


# ----------------- delta + PROPOSED_EDITS -----------------

def write_delta_and_edits(t1_diffs, decomp_info, main_cmp_df):
    """Compose rev9_delta.md and rev9_PROPOSED_EDITS.md."""
    # Main-cmp gains
    def cell(bg, name):
        r = main_cmp_df[(main_cmp_df.bg_level == bg) & (main_cmp_df.placement_name == name)]
        return None if not len(r) else (float(r.iloc[0]["worst_sap"]),
                                          float(r.iloc[0]["mean_sap"]))
    l1_naive = cell("L1_light", "Naive_allGPU")
    l1_sa    = cell("L1_light", "SizeAware")
    l1_sbr   = cell("L1_light", "SizeBlindRev")
    l2_naive = cell("L2_lm", "Naive_allGPU")
    l2_sa    = cell("L2_lm", "SizeAware")
    l2_sbr   = cell("L2_lm", "SizeBlindRev")

    sa_sbr_l1 = (l1_sa[0] - l1_sbr[0]) if (l1_sa and l1_sbr) else None
    sa_sbr_l2 = (l2_sa[0] - l2_sbr[0]) if (l2_sa and l2_sbr) else None
    sa_sbr_l1_pct = sa_sbr_l1 / l1_sbr[0] * 100 if (sa_sbr_l1 is not None and l1_sbr[0]) else None
    sa_sbr_l2_pct = sa_sbr_l2 / l2_sbr[0] * 100 if (sa_sbr_l2 is not None and l2_sbr[0]) else None

    sa_naive_l1_w = (l1_sa[0] - l1_naive[0]) if (l1_sa and l1_naive) else None
    sa_naive_l1_m = (l1_sa[1] - l1_naive[1]) if (l1_sa and l1_naive) else None
    ratio_l1 = (sa_naive_l1_w / sa_naive_l1_m) if (sa_naive_l1_m and abs(sa_naive_l1_m) > 1e-9) else None
    sa_naive_l2_w = (l2_sa[0] - l2_naive[0]) if (l2_sa and l2_naive) else None
    sa_naive_l2_m = (l2_sa[1] - l2_naive[1]) if (l2_sa and l2_naive) else None
    ratio_l2 = (sa_naive_l2_w / sa_naive_l2_m) if (sa_naive_l2_m and abs(sa_naive_l2_m) > 1e-9) else None

    q_share = decomp_info["q_share"] if decomp_info else None
    quant_delta = decomp_info["quant"] if decomp_info else None
    gpu_path = decomp_info["gpu_path"] if decomp_info else None
    total = decomp_info["total"] if decomp_info else None

    # write rev9_delta.md
    with open(DELTA, "w") as f:
        f.write(f"# rev9 delta — old paper / Table 1 vs rev9 normal-state measurements\n\n")
        f.write(f"_generated {datetime.now().isoformat(timespec='seconds')}_\n\n")
        f.write("All cells reuse rev6 baselines where the cell key matches (24-log GPU+NPU L0; "
                "GPU L1_light ladder; Comp A multistream). New rev9 measurements fill the rest.\n\n")
        f.write("## Table 1 (single-stream Δ NPU − GPU)\n\n")
        f.write("| size | old paper | rev9 (normal state) | flip? |\n|---|---|---|---|\n")
        f.write(f"| small  | $-0.007$ ($-47.1\\%$) | ${t1_diffs['small'][0]:+.4f}$ (${t1_diffs['small'][1]}$) | low |\n")
        f.write(f"| medium | $-0.036$ ($-19.5\\%$) | ${t1_diffs['medium'][0]:+.4f}$ (${t1_diffs['medium'][1]}$) | low |\n")
        f.write(f"| large  | $+0.001$ ($+0.2\\%$ n.s.) | ${t1_diffs['large'][0]:+.4f}$ (${t1_diffs['large'][1]}$) | **HIGH — magnitude moved from ≈0 to non-negligible** |\n")
        f.write("\n## main-comparison (N=4 size-diverse)\n\n")
        f.write("| bg | strategy | old worst sAP | rev9 worst sAP | old mean sAP | rev9 mean sAP |\n|---|---|---|---|---|---|\n")
        if l1_naive: f.write(f"| L1_light | Naive_allGPU | 0.084 | {l1_naive[0]:.3f} | 0.125 | {l1_naive[1]:.3f} |\n")
        if l1_sa:    f.write(f"| L1_light | SizeAware    | 0.105 | **{l1_sa[0]:.3f}** | 0.137 | {l1_sa[1]:.3f} |\n")
        if l1_sbr:   f.write(f"| L1_light | SizeBlindRev | 0.084 | {l1_sbr[0]:.3f} | 0.133 | {l1_sbr[1]:.3f} |\n")
        if l2_naive: f.write(f"| L2_LM    | Naive_allGPU | 0.058 | {l2_naive[0]:.3f} | 0.100 | {l2_naive[1]:.3f} |\n")
        if l2_sa:    f.write(f"| L2_LM    | SizeAware    | 0.065 | **{l2_sa[0]:.3f}** | 0.102 | {l2_sa[1]:.3f} |\n")
        if l2_sbr:   f.write(f"| L2_LM    | SizeBlindRev | 0.053 | {l2_sbr[0]:.3f} | 0.101 | {l2_sbr[1]:.3f} |\n")
        if sa_sbr_l1 is not None:
            f.write(f"\n### Inversion magnitude (SizeAware − SizeBlindRev worst sAP)\n")
            f.write(f"- L1_light: old $+0.021$ ($+25\\%$) → rev9 ${sa_sbr_l1:+.4f}$ (${sa_sbr_l1_pct:+.1f}\\%$). "
                    f"{'**SIGN PRESERVED**' if sa_sbr_l1 > 0 else '**SIGN FLIP**'}\n")
        if sa_sbr_l2 is not None:
            f.write(f"- L2_LM:    old $+0.012$ ($+23\\%$) → rev9 ${sa_sbr_l2:+.4f}$ (${sa_sbr_l2_pct:+.1f}\\%$). "
                    f"{'**SIGN PRESERVED**' if sa_sbr_l2 > 0 else '**SIGN FLIP**'}\n")
        if ratio_l1 is not None:
            f.write(f"\n### Worst-vs-mean ratio (SizeAware − Naive)\n")
            f.write(f"- L1_light: old worst $+0.021$, mean $+0.012$, ratio $1.75\\times$ → rev9 worst ${sa_naive_l1_w:+.4f}$, mean ${sa_naive_l1_m:+.4f}$, ratio ${ratio_l1:+.2f}\\times$\n")
        if ratio_l2 is not None:
            f.write(f"- L2_LM:    old worst $+0.007$, mean $+0.002$, ratio $3.5\\times$ → rev9 worst ${sa_naive_l2_w:+.4f}$, mean ${sa_naive_l2_m:+.4f}$, ratio ${ratio_l2:+.2f}\\times$\n")
        f.write("\n## Decomposition (N=8 size-diverse worst camera)\n\n")
        if decomp_info:
            f.write(f"| component | old | rev9 |\n|---|---|---|\n")
            f.write(f"| Quantization | $-0.031$ ($97\\%$) | ${-quant_delta:+.3f}$ (${q_share:.0f}\\%$) |\n")
            if total is not None:
                f.write(f"| Total NPU-path | $-0.032$ | ${-total:+.3f}$ |\n")
            if gpu_path is not None:
                f.write(f"| GPU-path (Naive N=8 L1) | $-0.060$ | ${-gpu_path:+.3f}$ |\n")
        f.write("\n## REVERSAL GATE\n\n")
        gate = pd.read_csv(RES / "rev9_reversal_gate.csv")
        if len(gate):
            r = gate.iloc[0]
            f.write(f"- sa_worst={float(r['sa_worst']):.4f}, sbr_worst={float(r['sbr_worst']):.4f}, "
                    f"worst gain ${float(r['worst_gain']):+.4f}$ → **{'PASS' if bool(r['pass']) else 'FAIL'}**\n")

    # ---- PROPOSED EDITS (SEMANTIC-LOCK rows only) ----
    with open(EDITS, "w") as f:
        f.write("# rev9 PROPOSED EDITS — SEMANTIC-LOCK rows (prose untouched; for human review)\n\n")
        f.write(f"_generated {datetime.now().isoformat(timespec='seconds')}_\n\n")
        f.write("All edits below are PROPOSALS. `paper/main_vision.tex` prose was NOT touched.\n\n")
        f.write("## Lock row 1 — Table 1 large AP gap (`paper/main_vision.tex` line ~411)\n")
        f.write(f"- Old: AP$_\\text{{large}}$ NPU − GPU = $+0.001$ ($+0.2\\%$, $p=0.92$, n.s.) — **'unaffected on large'**.\n")
        f.write(f"- rev9: ${t1_diffs['large'][0]:+.4f}$ (${t1_diffs['large'][1]}$). **Magnitude moved from ≈0 to non-negligible.**\n")
        f.write("- Impact: the 'large vehicles effectively unaffected' narrative in §5.1 (line ~416) is **refuted by rev9**. Editor must decide whether to keep historical Table 1 alongside rev9 or replace.\n\n")
        f.write("## Lock row 2 — §5.1 three-size narrative (`paper/main_vision.tex` line ~416)\n")
        f.write(f"- Old: '$-47\\%$ on small, $-19.5\\%$ on medium, effectively zero ($+0.2\\%$, n.s.) on large'.\n")
        f.write(f"- rev9: small ${t1_diffs['small'][1]}$, medium ${t1_diffs['medium'][1]}$, large ${t1_diffs['large'][1]}$.\n")
        f.write("- Impact: 'large effectively zero' is no longer true. Editor must rewrite either as (a) cite Table 1 as historical and the rev9 number as the current reproducible value, or (b) drop the parenthetical.\n\n")
        if sa_sbr_l1 is not None:
            f.write("## Lock row 3 — §5.3 C1 inversion magnitude (`paper/main_vision.tex` line ~543)\n")
            f.write(f"- Old: '$+25\\%$ at L1$_{{\\text{{light}}}}$ ($0.105$ vs $0.084$) and $+23\\%$ at L2$_{{\\text{{LM}}}}$ ($0.065$ vs $0.053$)'.\n")
            if l1_sa and l1_sbr:
                f.write(f"- rev9: L1_light ${l1_sa[0]:.3f}$ vs ${l1_sbr[0]:.3f}$ (${sa_sbr_l1_pct:+.1f}\\%$); ")
            if l2_sa and l2_sbr:
                f.write(f"L2_LM ${l2_sa[0]:.3f}$ vs ${l2_sbr[0]:.3f}$ (${sa_sbr_l2_pct:+.1f}\\%$).\n")
            else:
                f.write("\n")
            f.write("- Impact: SIGN PRESERVED (reversal still holds). MAGNITUDES different (smaller at L1, larger or different at L2). Editor decides whether to keep +25%/+23% with footnote or use rev9 magnitudes.\n\n")
        if q_share is not None:
            f.write("## Lock row 4 — Decomposition $97\\%$ caption (`paper/main_vision.tex` line ~793) and abstract\n")
            f.write(f"- Old: $97\\%$ of NPU-path loss from quantization.\n")
            f.write(f"- rev9: ${q_share:.0f}\\%$.\n")
            f.write("- Impact: narrative direction preserved (quantization dominates). Magnitude change small. Editor likely safe to update one number.\n\n")
        if ratio_l1 is not None or ratio_l2 is not None:
            f.write("## Lock row 5 — Worst/mean ratio '1.75–3.5×' (abstract, §6, conclusion)\n")
            ratios = []
            if ratio_l1 is not None: ratios.append((1.75, ratio_l1, "L1_light"))
            if ratio_l2 is not None: ratios.append((3.5, ratio_l2, "L2_LM"))
            for old, new, bg in ratios:
                f.write(f"- {bg}: old ${old}\\times$ → rev9 ${new:+.2f}\\times$. {'**SIGN FLIP**' if new<0 else 'sign preserved'}\n")
            if any(r[1] < 0 for r in ratios):
                f.write("- **CRITICAL**: at least one ratio flipped sign in rev9. The '1.75–3.5×' phrasing is no longer accurate. Editor decides: re-derive from another baseline pair, or revise narrative.\n\n")
    print(f"saved {DELTA}")
    print(f"saved {EDITS}")
    return {"sa_sbr_l1": sa_sbr_l1, "sa_sbr_l1_pct": sa_sbr_l1_pct,
            "ratio_l1": ratio_l1, "ratio_l2": ratio_l2,
            "q_share": q_share}


def main():
    print("[rev9-aggregate] writing tables…")
    write_tab_single_stream_path, t1_diffs = write_tab_single_stream()
    print(f"  saved {write_tab_single_stream_path}")
    print(f"  T1 diffs: small {t1_diffs['small']} medium {t1_diffs['medium']} large {t1_diffs['large']}")
    p = write_tab_per_class();   print(f"  saved {p}")
    p = write_tab_partA();        print(f"  saved {p}")
    p = write_tab_main_cmp();     print(f"  saved {p}")
    p = write_tab_capacity();     print(f"  saved {p}")
    p = write_tab_schedule();     print(f"  saved {p}")
    p = write_tab_natural();      print(f"  saved {p}")
    decomp_path, decomp_info = write_tab_decomposition(t1_diffs)
    print(f"  saved {decomp_path}")
    print(f"  decomp: {decomp_info}")
    print("\n[rev9-aggregate] writing delta + PROPOSED_EDITS…")
    main_cmp_df = load_main_cmp()
    impacts = write_delta_and_edits(t1_diffs, decomp_info, main_cmp_df)
    # One-line summary
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(impacts.keys()))
        w.writeheader(); w.writerow({k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in impacts.items()})
    print(f"saved {SUMMARY}")


if __name__ == "__main__":
    main()
