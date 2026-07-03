"""STEP 4 + STEP 5 of B3 (rev 4) — turn v11 family sweep CSVs into tables/figure.

Outputs:
  results/gen_decomp_v4.csv         per-detector Q(s) and L(s, L2_lm)
  results/cstar_v4_family.csv       per-(detector, group) sap_npu_g + Cstar_ms
  results/gen_gain_v4.csv           N=4 SizeAware vs SizeBlindRev gain per detector
  paper/tables/gen_decomp.tex       v11 family (main) + v11n TBD; v8 moved to appendix
  paper/tables/gen_gain.tex         v11 family main; v8 appendix
  paper/tables/gen_decomp_appendix_v8.tex   v8 family cross-family supplement
  paper/tables/gen_gain_appendix_v8.tex     v8 family cross-family supplement
  paper/figures/gen_cstar.pdf       v11 family multi-capacity C* (sAP-gap def)

v11 family uses NPU global8 mode for the single-stream phases (B'+C') and NPU
single mode for the multi-stream phase (D'). Both are documented in the table
captions; the multistream-mode caveat applies equally to v8.
"""

from __future__ import annotations

import json
import sys
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
PAPER_TAB = SCRIPT_DIR.parent.parent / "paper" / "tables"
PAPER_FIG = SCRIPT_DIR.parent.parent / "paper" / "figures"
FIG_PROJ = SCRIPT_DIR.parent.parent / "figures"
PAPER_TAB.mkdir(parents=True, exist_ok=True)
PAPER_FIG.mkdir(parents=True, exist_ok=True)
FIG_PROJ.mkdir(parents=True, exist_ok=True)

# Main: v11 family (global8). v11n TBD.
V11_FAMILY = [
    {"name": "yolo11n", "params_M": 2.6,  "display": "YOLO11n", "tbd": True},
    {"name": "yolo11s", "params_M": 9.4,  "display": "YOLO11s", "tbd": False},
    {"name": "yolo11m", "params_M": 20.1, "display": "YOLO11m", "tbd": False},
    {"name": "yolo11l", "params_M": 25.3, "display": "YOLO11l", "tbd": False},
    {"name": "yolo11x", "params_M": 56.9, "display": "YOLO11x", "tbd": False},
]
# Appendix: v8 family (single mode).
V8_FAMILY = [
    {"name": "yolov8n", "params_M": 3.2,  "display": "YOLOv8n"},
    {"name": "yolov8s", "params_M": 11.2, "display": "YOLOv8s"},
    {"name": "yolov8m", "params_M": 25.9, "display": "YOLOv8m"},
    {"name": "yolov8l", "params_M": 43.7, "display": "YOLOv8l"},
    {"name": "yolov8x", "params_M": 68.2, "display": "YOLOv8x"},
]
LADDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
SIZE_GROUPS = {
    "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
    "medium-mixed": [0, 1, 6, 8, 10, 11, 17, 20],
    "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
}


# ---------------------------------- v11 family loaders ----------------------

def load_v11_baseline(det):
    fp = RES / f"p1g8_baseline_{det}.csv"
    return pd.read_csv(fp) if fp.exists() else None


def load_v11_ladder(det):
    """Combine baseline (GPU L0) with ladder GPU rows."""
    base = load_v11_baseline(det)
    ldr_fp = RES / f"p1g8_ladder_{det}.csv"
    if base is None: return None
    g_l0 = base[(base.device == "GPU") & (base.bg_level == "L0")].copy()
    parts = [g_l0]
    if ldr_fp.exists():
        parts.append(pd.read_csv(ldr_fp))
    df = pd.concat(parts, ignore_index=True)
    return df[df.device == "GPU"]


def load_v11_multistream(det):
    fp = RES / f"p1g8_multistream_{det}.csv"
    return pd.read_csv(fp) if fp.exists() else None


def load_v8_baseline(det):
    fp = RES / f"p1_baseline_{det}.csv"
    return pd.read_csv(fp) if fp.exists() else None


def load_v8_ladder(det):
    base = load_v8_baseline(det); ldr_fp = RES / f"p1_ladder_{det}.csv"
    if base is None: return None
    g_l0 = base[(base.device == "GPU") & (base.bg_level == "L0")].copy()
    parts = [g_l0]
    if ldr_fp.exists():
        parts.append(pd.read_csv(ldr_fp))
    return pd.concat(parts, ignore_index=True)


def load_v8_multistream(det):
    fp = RES / f"p1_multistream_{det}.csv"
    return pd.read_csv(fp) if fp.exists() else None


# ---------------------------------- common compute --------------------------

def compute_QL(baseline_df, ladder_df):
    """Q(s) = single-stream L0 size-stratified sAP gap (GPU - NPU paired per sid).
       L(s, L2_lm) = mean over groups of (L0_GPU sAP - L2_lm_GPU sAP)."""
    if baseline_df is None or ladder_df is None:
        return None
    g_l0 = baseline_df[(baseline_df.device == "GPU") & (baseline_df.bg_level == "L0")].sort_values("sid").reset_index(drop=True)
    n_l0 = baseline_df[(baseline_df.device == "NPU") & (baseline_df.bg_level == "L0")].sort_values("sid").reset_index(drop=True)
    if not len(g_l0) or not len(n_l0):
        return None
    Q = {}
    for size in ["sap_s", "sap_m", "sap_l"]:
        Q[size.replace("sap_", "")] = float((n_l0[size] - g_l0[size]).mean())
    # L at L2_lm
    g_l2 = ladder_df[ladder_df.bg_level == "L2_lm"].sort_values("sid").reset_index(drop=True)
    L = {}
    for size in ["sap_s", "sap_m", "sap_l"]:
        if not len(g_l2):
            L[size.replace("sap_", "")] = None
            continue
        n = min(len(g_l0), len(g_l2))
        L[size.replace("sap_", "")] = float((g_l0[size].iloc[:n] - g_l2[size].iloc[:n]).mean())
    return {"Q": Q, "L": L}


def compute_cstar(baseline_df, ladder_df):
    """sAP-gap C*: per group, sap_npu_g - sap_gpu_g(C) crosses zero."""
    if baseline_df is None or ladder_df is None:
        return None
    n_l0 = baseline_df[(baseline_df.device == "NPU") & (baseline_df.bg_level == "L0")]
    npu_l0 = {int(r["sid"]): float(r["sap_5095"]) for _, r in n_l0.iterrows()}
    gpu_by_bg = {}
    C_per_bg = {}
    for bg in LADDER:
        sub = ladder_df[ladder_df.bg_level == bg]
        if not len(sub): continue
        gpu_by_bg[bg] = {int(r["sid"]): float(r["sap_5095"]) for _, r in sub.iterrows()}
        if "eff_e2e_mean" in sub.columns and sub["eff_e2e_mean"].notna().any():
            C_per_bg[bg] = float(sub["eff_e2e_mean"].mean())
        elif "latency_mean" in sub.columns:
            C_per_bg[bg] = float(sub["latency_mean"].mean())
        else:
            C_per_bg[bg] = float("nan")
    result = {}
    for grp, sids in SIZE_GROUPS.items():
        sap_npu_g = float(np.mean([npu_l0.get(s) for s in sids if s in npu_l0]))
        points = []
        for bg in LADDER:
            if bg not in gpu_by_bg or bg not in C_per_bg or np.isnan(C_per_bg[bg]):
                continue
            vals = [gpu_by_bg[bg].get(s) for s in sids if s in gpu_by_bg[bg]]
            vals = [v for v in vals if v is not None]
            if not vals: continue
            sap_gpu_g = float(np.mean(vals))
            delta = sap_npu_g - sap_gpu_g
            points.append({"bg": bg, "C": C_per_bg[bg],
                           "sap_gpu_g": sap_gpu_g, "delta": delta})
        pts = sorted(points, key=lambda x: x["C"])
        Cstar = None
        for i in range(len(pts) - 1):
            d1 = pts[i]["delta"]; d2 = pts[i + 1]["delta"]
            C1 = pts[i]["C"]; C2 = pts[i + 1]["C"]
            if d1 * d2 <= 0 and (d2 != d1):
                Cstar = C1 + (-d1) / (d2 - d1) * (C2 - C1); break
        result[grp] = {"sap_npu_g": sap_npu_g, "Cstar_ms": Cstar, "curve": pts}
    return result


def compute_gain(ms_df):
    """N=4 L1_light SizeAware vs SizeBlindRev, worst + mean."""
    if ms_df is None or not len(ms_df):
        return None
    sa = ms_df[(ms_df.n_streams == 4) & (ms_df.placement_name == "SizeAware")]
    sb = ms_df[(ms_df.n_streams == 4) & (ms_df.placement_name == "SizeBlindRev")]
    if not len(sa) or not len(sb): return None
    worst_gain = float(sa.iloc[0]["worst_sap"]) - float(sb.iloc[0]["worst_sap"])
    mean_gain = float(sa.iloc[0]["mean_sap"]) - float(sb.iloc[0]["mean_sap"])
    ratio = worst_gain / mean_gain if abs(mean_gain) > 1e-9 else float("inf")
    return {"sa_worst": float(sa.iloc[0]["worst_sap"]),
            "sb_worst": float(sb.iloc[0]["worst_sap"]),
            "sa_mean":  float(sa.iloc[0]["mean_sap"]),
            "sb_mean":  float(sb.iloc[0]["mean_sap"]),
            "worst_gain": worst_gain, "mean_gain": mean_gain,
            "worst_mean_ratio": ratio, "inverts": (worst_gain > 0)}


# ---------------------------------- table writers ---------------------------

def write_decomp_tex(family, label, family_results, mode_tag, fname, l_slice="L2_lm"):
    path = PAPER_TAB / fname
    with open(path, "w") as f:
        f.write(f"% Auto-generated by phase_b3_aggregate.py — {label} family\n")
        f.write(f"% NPU mode: {mode_tag} for single-stream Q+L. L block uses {l_slice}.\n")
        f.write("\\begin{tabular}{ll|ccc|ccc}\n\\toprule\n")
        l_label = l_slice.replace("_", "\\_")
        f.write("& & \\multicolumn{3}{c|}{$Q(s)$ (quantization)} & "
                f"\\multicolumn{{3}}{{c}}{{$L(s,\\text{{{l_label}}})$ (staleness)}} \\\\\n")
        f.write("Detector & params & small & medium & large & small & medium & large \\\\\n")
        f.write("\\midrule\n")
        for det in family:
            disp = det["display"]; pm = det["params_M"]
            if det.get("tbd"):
                f.write(f"{disp} & {pm}\\,M & TBD & TBD & TBD & TBD & TBD & TBD \\\\\n")
                continue
            r = family_results.get(det["name"])
            if r is None or r.get("QL") is None:
                f.write(f"{disp} & {pm}\\,M & --- & --- & --- & --- & --- & --- \\\\\n"); continue
            QL = r["QL"]; Q = QL["Q"]; L = QL["L"]
            def fmt(v): return "TBD" if v is None else f"${v:+.3f}$"
            f.write(f"{disp} & {pm}\\,M & "
                    f"{fmt(Q['s'])} & {fmt(Q['m'])} & {fmt(Q['l'])} & "
                    f"{fmt(L['s'])} & {fmt(L['m'])} & {fmt(L['l'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {path}")


def write_gain_tex(family, family_results, fname):
    path = PAPER_TAB / fname
    with open(path, "w") as f:
        f.write("% Auto-generated by phase_b3_aggregate.py.\n")
        f.write("% Worst-stream gain SizeAware - SizeBlindRev at N=4 bg L1_light, Composition A.\n")
        f.write("% Multi-stream phase runs in NPU single mode (global8 cannot host multiple\n")
        f.write("% instances); this is the same configuration v8 uses.\n")
        f.write("\\begin{tabular}{ll|cccc}\n\\toprule\n")
        f.write("Detector & family & inverts? & worst gain & mean gain & worst/mean \\\\\n")
        f.write("\\midrule\n")
        for det in family:
            disp = det["display"]
            fam = "YOLO11" if det["name"].startswith("yolo11") else "YOLOv8"
            if det.get("tbd"):
                f.write(f"{disp} & {fam} & TBD & TBD & TBD & TBD \\\\\n"); continue
            r = family_results.get(det["name"])
            if r is None or r.get("gain") is None:
                f.write(f"{disp} & {fam} & --- & --- & --- & --- \\\\\n"); continue
            g = r["gain"]
            inv = "yes" if g["inverts"] else "no"
            wg_s = f"\\textbf{{${g['worst_gain']:+.3f}$}}"
            mg_s = f"${g['mean_gain']:+.3f}$"
            ratio = g["worst_mean_ratio"]
            if ratio is None or ratio != ratio:
                ratio_s = "---"
            elif ratio == float("inf") or ratio == float("-inf"):
                ratio_s = "$\\infty$"
            else:
                ratio_s = f"${ratio:+.1f}\\times$"
            f.write(f"{disp} & {fam} & {inv} & {wg_s} & {mg_s} & {ratio_s} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {path}")


def render_cstar(family_results, out_path):
    """Two-panel figure: Δ_g(C) curves (left) and C* vs params (right)."""
    grp_colors = {"small-rich": "#3680c4",
                  "medium-mixed": "#3b9c4d",
                  "large-rich": "#c43b3b"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Δ_g(C) curves per (detector × group)
    ax = axes[0]
    det_list = [d for d in V11_FAMILY if not d.get("tbd")]
    alphas = np.linspace(0.4, 1.0, len(det_list))
    for det, alpha in zip(det_list, alphas):
        r = family_results.get(det["name"], {})
        cstar = r.get("cstar")
        if not cstar: continue
        for grp, info in cstar.items():
            xs = [p["C"] for p in info["curve"]]
            ys = [p["delta"] for p in info["curve"]]
            ax.plot(xs, ys, "-", color=grp_colors[grp], alpha=alpha,
                    linewidth=1.4, marker="o", markersize=4)
            if info.get("Cstar_ms") is not None:
                ax.scatter([info["Cstar_ms"]], [0], color=grp_colors[grp],
                           marker="*", s=70, alpha=alpha, zorder=5,
                           edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xscale("symlog", linthresh=20)
    ax.set_xlabel("scalar contention proxy C  (GPU eff E2E latency, ms)")
    ax.set_ylabel(r"$\Delta_g(C) = \mathrm{sAP}_{\mathrm{NPU},g} - \mathrm{sAP}_{\mathrm{GPU},g}(C)$")
    ax.set_title("Δ_g(C) curves across YOLO11 family  (alpha = capacity)")
    ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=c, label=g) for g, c in grp_colors.items()]
    ax.legend(handles=handles, fontsize=9, loc="best")

    # Panel B: C* vs detector params
    ax = axes[1]
    for grp, color in grp_colors.items():
        xs = []; ys = []; lbls = []
        for det in det_list:
            r = family_results.get(det["name"], {})
            cstar = r.get("cstar")
            if cstar and cstar.get(grp) and cstar[grp].get("Cstar_ms") is not None:
                xs.append(det["params_M"]); ys.append(cstar[grp]["Cstar_ms"])
                lbls.append(det["display"])
        if xs:
            ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=10, label=grp)
            for x, y, l in zip(xs, ys, lbls):
                ax.annotate(l.replace("YOLO11", "v11"),
                            xy=(x, y), xytext=(4, 4),
                            textcoords="offset points", fontsize=6, color=color)
    # TBD marker for v11n
    ax.axvline(2.6, color="gray", linestyle=":", alpha=0.4)
    yl = ax.get_ylim()
    ax.text(2.6, yl[1] * 0.95 if yl[1] > 0 else 50,
            "YOLO11n\n(TBD)", fontsize=7, color="gray", ha="center", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("detector capacity (parameters, M)")
    ax.set_ylabel("inversion threshold C* (ms)")
    ax.set_title("C* vs detector capacity (YOLO11 family, NPU global8)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle("YOLO11 family — NPU global8 single-stream sweep (sAP-gap C*)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    # Also copy to project root
    fig.savefig(FIG_PROJ / "gen_cstar.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"saved {FIG_PROJ / 'gen_cstar.pdf'}")


# ---------------------------------- summary CSVs ----------------------------

def write_summary_csvs(family_results, name_suffix=""):
    rows = []
    cstar_rows = []
    gain_rows = []
    for det in (V11_FAMILY + V8_FAMILY):
        if det.get("tbd"): continue
        r = family_results.get(det["name"])
        if r is None: continue
        QL = r.get("QL"); cstar = r.get("cstar"); gain = r.get("gain")
        family_tag = "YOLO11" if det["name"].startswith("yolo11") else "YOLOv8"
        if QL:
            row = {"detector": det["name"], "params_M": det["params_M"], "family": family_tag,
                   **{f"Q_{s}": QL["Q"][s] for s in ["s","m","l"]},
                   **{f"L_{s}": QL["L"][s] for s in ["s","m","l"]}}
            rows.append(row)
        if cstar:
            for grp, info in cstar.items():
                cstar_rows.append({"detector": det["name"], "params_M": det["params_M"],
                                    "family": family_tag, "group": grp,
                                    "sap_npu_g": info["sap_npu_g"],
                                    "Cstar_ms": info["Cstar_ms"]})
        if gain:
            gain_rows.append({"detector": det["name"], "params_M": det["params_M"],
                              "family": family_tag, **gain})
    pd.DataFrame(rows).to_csv(RES / "gen_decomp_v4.csv", index=False)
    pd.DataFrame(cstar_rows).to_csv(RES / "cstar_v4_family.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(RES / "gen_gain_v4.csv", index=False)
    print(f"saved {RES / 'gen_decomp_v4.csv'}, cstar_v4_family.csv, gen_gain_v4.csv")


# ---------------------------------- main ------------------------------------

def main():
    v11_results = {}
    v8_results = {}
    print("\n=== aggregating YOLO11 family (global8 single-stream, single multistream) ===")
    for det in V11_FAMILY:
        if det.get("tbd"): continue
        b = load_v11_baseline(det["name"])
        l = load_v11_ladder(det["name"])
        ms = load_v11_multistream(det["name"])
        v11_results[det["name"]] = {
            "QL": compute_QL(b, l),
            "cstar": compute_cstar(b, l),
            "gain": compute_gain(ms),
        }
    print("\n=== aggregating YOLOv8 family (single-mode single-stream + multistream) ===")
    for det in V8_FAMILY:
        b = load_v8_baseline(det["name"])
        l = load_v8_ladder(det["name"])
        ms = load_v8_multistream(det["name"])
        v8_results[det["name"]] = {
            "QL": compute_QL(b, l),
            "cstar": compute_cstar(b, l),
            "gain": compute_gain(ms),
        }
    # Combine for summary CSVs
    all_results = {**v11_results, **v8_results}
    write_summary_csvs(all_results)

    # Print summaries
    print("\n=== YOLO11 family Q(s), L(s, L2_lm) ===")
    for det in V11_FAMILY:
        if det.get("tbd"): continue
        QL = v11_results[det["name"]].get("QL")
        if QL is None: print(f"  {det['display']:<10s}  NO DATA"); continue
        Q = QL["Q"]; L = QL["L"]
        print(f"  {det['display']:<10s}  "
              f"Q s/m/l={Q['s']:+.3f}/{Q['m']:+.3f}/{Q['l']:+.3f}  "
              f"L s/m/l={L['s']:+.3f}/{L['m']:+.3f}/{L['l']:+.3f}")
    print("\n=== YOLO11 family C*_g (sAP-gap def, ms) ===")
    for det in V11_FAMILY:
        if det.get("tbd"): continue
        cstar = v11_results[det["name"]].get("cstar") or {}
        cs_s = (cstar.get("small-rich") or {}).get("Cstar_ms")
        cs_m = (cstar.get("medium-mixed") or {}).get("Cstar_ms")
        cs_l = (cstar.get("large-rich") or {}).get("Cstar_ms")
        print(f"  {det['display']:<10s}  "
              f"small-rich={cs_s}  medium-mixed={cs_m}  large-rich={cs_l}")
    print("\n=== YOLO11 family multistream gain N=4 L1_light ===")
    for det in V11_FAMILY:
        if det.get("tbd"): continue
        g = v11_results[det["name"]].get("gain")
        if g is None: print(f"  {det['display']:<10s}  NO DATA"); continue
        print(f"  {det['display']:<10s}  worst_gain={g['worst_gain']:+.3f}  "
              f"mean_gain={g['mean_gain']:+.3f}  ratio={g['worst_mean_ratio']:+.2f}  "
              f"inverts={g['inverts']}")

    # Write tables
    write_decomp_tex(V11_FAMILY, "YOLO11 family", v11_results,
                     mode_tag="\\texttt{global8}", fname="gen_decomp.tex")
    write_decomp_tex(V8_FAMILY, "YOLOv8 family", v8_results,
                     mode_tag="\\texttt{single}", fname="gen_decomp_appendix_v8.tex")
    write_gain_tex(V11_FAMILY, v11_results, "gen_gain.tex")
    write_gain_tex(V8_FAMILY, v8_results, "gen_gain_appendix_v8.tex")

    # Render figure
    render_cstar(v11_results, PAPER_FIG / "gen_cstar.pdf")


if __name__ == "__main__":
    main()
