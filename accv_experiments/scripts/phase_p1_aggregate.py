"""Phase P1 aggregator — turn per-detector CSVs into family tables/figure.

Inputs:
  results/p1_baseline_<det>.csv      per-detector single-stream baseline (GPU L0, NPU L0)
  results/p1_ladder_<det>.csv        per-detector GPU ladder (L1_light..L3_vlm)
  results/p1_multistream_<det>.csv   per-detector multistream (N=4, N=8)
  results/step_a_baseline.csv        YOLOv11s marker baseline (global8 mode)
  results/gen_decomp.csv             YOLOv11s ladder (L0 GPU and L1_light)
  results/cstar.csv                  YOLOv11s heavier ladder (L1_heavy, L2_lm, L3_vlm)
  results/step_g2_single_mode.csv    YOLOv11s multistream (N=4, N=8)
  results/step_e_size_classification.csv  size labels per sid

Outputs:
  results/gen_decomp_v2.csv          per-detector Q(s) and L(s,L1)
  results/cstar_v2_family.csv        per-detector per-group C*_g (sAP-gap def)
  results/gen_gain_v2.csv            per-detector worst gain at N=4 L1_light
  paper/tables/gen_decomp.tex        (regenerated, 5 v8 + v11s)
  paper/tables/gen_gain.tex          (regenerated, 5 v8 + v11s)
  paper/figures/gen_cstar.pdf        multi-capacity C* curves
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

DETECTORS = [
    {"name": "yolov8n",  "params_M": 3.2,  "display": "YOLOv8n"},
    {"name": "yolov8s",  "params_M": 11.2, "display": "YOLOv8s"},
    {"name": "yolov8m",  "params_M": 25.9, "display": "YOLOv8m"},
    {"name": "yolov8l",  "params_M": 43.7, "display": "YOLOv8l"},
    {"name": "yolov8x",  "params_M": 68.2, "display": "YOLOv8x"},
    {"name": "yolo11s",  "params_M": 9.4,  "display": "YOLOv11s (marker)"},  # reuse
]
LADDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
SIZE_GROUPS = {
    "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
    "medium-mixed": [0, 1, 6, 8, 10, 11, 17, 20],
    "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
}


# ---------------------------------- IO --------------------------------------

def load_baseline(det):
    if det == "yolo11s":
        df = pd.read_csv(RES / "step_a_baseline.csv")
        val = load_val()
        log_to_sid = {n: i for i, n in enumerate(val["sequences"])}
        df["sid"] = df["log_id"].map(log_to_sid)
        df["bg_level"] = "L0"  # step_a is L0 baseline
        df["eff_e2e_mean"] = df["infer_mean_ms"]
        return df  # uses sap_small/medium/large naming
    fp = RES / f"p1_baseline_{det}.csv"
    return pd.read_csv(fp) if fp.exists() else None


def load_ladder(det):
    """Return DataFrame with rows for GPU at every ladder level (L0..L3_vlm).
    L0 is taken from baseline (GPU device); the rest from p1_ladder/p1_baseline."""
    if det == "yolo11s":
        df1 = pd.read_csv(RES / "step_a_baseline.csv")
        val = load_val()
        log_to_sid = {n: i for i, n in enumerate(val["sequences"])}
        df1["sid"] = df1["log_id"].map(log_to_sid)
        df1 = df1[df1.device == "GPU"].copy()
        df1["bg_level"] = "L0"
        df1["eff_e2e_mean"] = df1["infer_mean_ms"]  # step_a stores infer; eff≈infer at L0
        # gen_decomp.csv has L1_light (col names sap_s/m/l + map_s/m/l + latency_mean)
        gd = pd.read_csv(RES / "gen_decomp.csv")
        l1 = gd[(gd.device == "GPU") & (gd.bg_level == "L1_light")].copy()
        l1["eff_e2e_mean"] = l1["latency_mean"]
        # cstar.csv (step_l) has L1_heavy/L2_lm/L3_vlm
        cs = pd.read_csv(RES / "cstar.csv")
        cs = cs[cs.bg_level.isin(["L1_heavy", "L2_lm", "L3_vlm"])]
        # Normalize columns
        def norm(df):
            keep = ["sid", "bg_level", "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
                    "map_5095", "map_s", "map_m", "map_l", "eff_e2e_mean", "frame_skip_pct"]
            # step_a uses sap_small etc.; remap if present
            if "sap_small" in df.columns:
                df = df.rename(columns={"sap_small": "sap_s",
                                        "sap_medium": "sap_m",
                                        "sap_large": "sap_l"})
            for c in keep:
                if c not in df.columns:
                    df[c] = np.nan
            return df[keep]
        return pd.concat([norm(df1), norm(l1), norm(cs)], ignore_index=True)
    # v8 family: baseline (L0) + ladder
    base = load_baseline(det)
    ldr = pd.read_csv(RES / f"p1_ladder_{det}.csv") if (RES / f"p1_ladder_{det}.csv").exists() else None
    if base is None:
        return None
    # baseline has both GPU and NPU rows; we want only GPU L0
    base_gpu = base[(base.device == "GPU") & (base.bg_level == "L0")].copy()
    cols = ["sid", "bg_level", "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
            "map_5095", "map_s", "map_m", "map_l", "eff_e2e_mean", "frame_skip_pct"]
    base_gpu = base_gpu[[c for c in cols if c in base_gpu.columns]]
    if ldr is not None:
        ldr_gpu = ldr[ldr.device == "GPU"][[c for c in cols if c in ldr.columns]]
        return pd.concat([base_gpu, ldr_gpu], ignore_index=True)
    return base_gpu


def load_multistream(det):
    """Return SizeAware vs SizeBlindRev rows at N=4, N=8 L1_light."""
    rows = []
    if det == "yolo11s":
        df = pd.read_csv(RES / "step_g2_single_mode.csv")
        # N=4 L1_light SizeAware / SizeBlindRev — find rows matching specs
        # SizeAware [NPU, NPU, GPU, GPU] = small→NPU
        # SizeBlindRev [GPU, GPU, NPU, NPU] = large→NPU
        SPEC_N4 = {"SizeAware": '["NPU", "NPU", "GPU", "GPU"]',
                   "SizeBlindRev": '["GPU", "GPU", "NPU", "NPU"]'}
        SPEC_N8 = {"SizeAware_NPU4":
                       '["NPU", "NPU", "NPU", "NPU", "GPU", "GPU", "GPU", "GPU"]',
                   "SizeBlindRev_NPU4":
                       '["GPU", "GPU", "GPU", "GPU", "NPU", "NPU", "NPU", "NPU"]'}
        for tag, specs in [(4, SPEC_N4), (8, SPEC_N8)]:
            for pname, spec in specs.items():
                sub = df[(df.n_streams == tag) & (df.bg_level == "L1") &
                         (df.placement_spec == spec)]
                if len(sub):
                    r = sub.iloc[0]
                    rows.append({"n_streams": tag, "bg_level": "L1_light",
                                  "placement_name": pname,
                                  "mean_sap": float(r["mean_sap"]),
                                  "worst_sap": float(r["worst_sap"])})
        return pd.DataFrame(rows)
    fp = RES / f"p1_multistream_{det}.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    return df[["n_streams", "bg_level", "placement_name", "mean_sap", "worst_sap"]]


# ---------------------------------- aggregate -------------------------------

def compute_Q_L_per_det(det):
    """Q(s) = mean over sids of (GPU L0 size-stratified sAP - NPU L0 sAP)
       L(s,L1) = mean over sids of (GPU L0 sAP - GPU L1_light sAP)
       All in size-stratified sAP units (per Table 1 convention)."""
    base = load_baseline(det)
    if base is None:
        return None
    sap_cols = {"small": ["sap_small", "sap_s"],
                "medium": ["sap_medium", "sap_m"],
                "large": ["sap_large", "sap_l"]}
    def pick(df, size):
        for c in sap_cols[size]:
            if c in df.columns:
                return df[c].astype(float)
        return None
    Q = {}
    g_l0 = base[(base.device == "GPU") & (base.bg_level == "L0")]
    n_l0 = base[(base.device == "NPU") & (base.bg_level == "L0")]
    # Pair by sid
    g_l0 = g_l0.sort_values("sid").reset_index(drop=True)
    n_l0 = n_l0.sort_values("sid").reset_index(drop=True)
    for s in ["small", "medium", "large"]:
        gv = pick(g_l0, s); nv = pick(n_l0, s)
        if gv is None or nv is None:
            Q[s] = None
        else:
            Q[s] = float((nv - gv).mean())  # NPU-GPU (negative = NPU loss)

    # L(s, L1) from baseline GPU L0 vs ladder GPU L1_light
    ldr = load_ladder(det)
    if ldr is None:
        return {"Q": Q, "L": {"small": None, "medium": None, "large": None}}
    g_l1 = ldr[ldr.bg_level == "L1_light"].sort_values("sid").reset_index(drop=True)
    L = {}
    for s in ["small", "medium", "large"]:
        gv0 = pick(g_l0, s); gv1 = pick(g_l1, s)
        if gv0 is None or gv1 is None:
            L[s] = None
        else:
            n = min(len(gv0), len(gv1))
            L[s] = float((gv0.iloc[:n] - gv1.iloc[:n]).mean())  # offline - streaming (positive = staleness loss)
    return {"Q": Q, "L": L}


def compute_cstar_per_det(det):
    """For each size group, compute C*_g via NEW sAP-gap definition.
       Returns dict: {group: {"sap_npu_g": x, "Cstar_ms": y, "delta_curve": [...]}}.
    """
    base = load_baseline(det)
    ldr = load_ladder(det)
    if base is None or ldr is None:
        return None
    # NPU L0 sAP per sid (overall sap_5095)
    npu_l0 = base[(base.device == "NPU") & (base.bg_level == "L0")][["sid", "sap_5095"]]
    npu_l0 = {int(r["sid"]): float(r["sap_5095"]) for _, r in npu_l0.iterrows()}
    # GPU per bg level per sid + eff E2E
    gpu_by_bg = {}
    for bg in LADDER:
        sub = ldr[ldr.bg_level == bg]
        if not len(sub):
            continue
        gpu_by_bg[bg] = {int(r["sid"]): float(r["sap_5095"]) for _, r in sub.iterrows()}
    # Mean C per bg (eff E2E)
    C_per_bg = {}
    for bg in LADDER:
        sub = ldr[ldr.bg_level == bg]
        if len(sub) and "eff_e2e_mean" in sub.columns:
            C_per_bg[bg] = float(sub["eff_e2e_mean"].mean())
    result = {}
    for grp, sids in SIZE_GROUPS.items():
        sap_npu_g = float(np.mean([npu_l0.get(s) for s in sids if s in npu_l0]))
        points = []
        for bg in LADDER:
            if bg not in gpu_by_bg or bg not in C_per_bg:
                continue
            vals = [gpu_by_bg[bg].get(s) for s in sids if s in gpu_by_bg[bg]]
            vals = [v for v in vals if v is not None]
            if not vals: continue
            sap_gpu_g = float(np.mean(vals))
            delta = sap_npu_g - sap_gpu_g
            points.append({"bg": bg, "C": C_per_bg[bg], "sap_gpu_g": sap_gpu_g, "delta": delta})
        # Interpolate C*
        pts = sorted(points, key=lambda x: x["C"])
        Cstar = None
        for i in range(len(pts) - 1):
            d1 = pts[i]["delta"]; d2 = pts[i + 1]["delta"]
            C1 = pts[i]["C"]; C2 = pts[i + 1]["C"]
            if d1 * d2 <= 0 and (d2 != d1):
                Cstar = C1 + (-d1) / (d2 - d1) * (C2 - C1)
                break
        result[grp] = {"sap_npu_g": sap_npu_g, "Cstar_ms": Cstar, "curve": pts}
    return result


def compute_gain_per_det(det):
    """SizeAware vs SizeBlindRev at N=4 L1_light."""
    ms = load_multistream(det)
    if ms is None or not len(ms):
        return None
    # Find N=4 L1_light SA + SBR
    sa = ms[(ms.n_streams == 4) & (ms.bg_level == "L1_light") &
            (ms.placement_name == "SizeAware")]
    sb = ms[(ms.n_streams == 4) & (ms.bg_level == "L1_light") &
            (ms.placement_name == "SizeBlindRev")]
    if not len(sa) or not len(sb):
        return None
    worst_gain = float(sa.iloc[0]["worst_sap"]) - float(sb.iloc[0]["worst_sap"])
    mean_gain = float(sa.iloc[0]["mean_sap"]) - float(sb.iloc[0]["mean_sap"])
    ratio = worst_gain / mean_gain if abs(mean_gain) > 1e-9 else float("inf")
    return {
        "sa_worst": float(sa.iloc[0]["worst_sap"]),
        "sb_worst": float(sb.iloc[0]["worst_sap"]),
        "sa_mean": float(sa.iloc[0]["mean_sap"]),
        "sb_mean": float(sb.iloc[0]["mean_sap"]),
        "worst_gain": worst_gain,
        "mean_gain": mean_gain,
        "worst_mean_ratio": ratio,
        "inverts": (worst_gain > 0),
    }


# ---------------------------------- write tables ----------------------------

def write_gen_decomp_tex(det_results):
    """Write paper/tables/gen_decomp.tex with v8 family + v11s marker."""
    out = PAPER_TAB / "gen_decomp.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_p1_aggregate.py\n")
        f.write("% Q(s) = single-stream L0 size-stratified sAP gap (GPU - NPU).\n")
        f.write("% L(s,L1) = GPU offline L0 sAP - GPU streaming L1_light sAP per size.\n")
        f.write("% Units: per-image sAP (matches tab:single-stream / Table 1).\n")
        f.write("% YOLOv11s row reuses step_a_baseline (global8 NPU mode) as a cross-family marker.\n")
        f.write("\\begin{tabular}{ll|ccc|ccc}\n\\toprule\n")
        f.write("& & \\multicolumn{3}{c|}{$Q(s)$ (quantization)} & "
                "\\multicolumn{3}{c}{$L(s,\\text{L1})$ (staleness)} \\\\\n")
        f.write("Detector & params & small & medium & large & small & medium & large \\\\\n")
        f.write("\\midrule\n")
        for det in DETECTORS:
            r = det_results.get(det["name"]) or {}
            disp = det["display"]; pm = det["params_M"]
            QL = r.get("QL")
            if QL is None:
                f.write(f"{disp} & {pm}\\,M & TBD & TBD & TBD & TBD & TBD & TBD \\\\\n")
                continue
            Q = QL["Q"]; L = QL["L"]
            def fmt(v): return "TBD" if v is None else f"${v:+.3f}$"
            f.write(f"{disp} & {pm}\\,M & "
                    f"{fmt(Q['small'])} & {fmt(Q['medium'])} & {fmt(Q['large'])} & "
                    f"{fmt(L['small'])} & {fmt(L['medium'])} & {fmt(L['large'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {out}")


def write_gen_gain_tex(det_results):
    out = PAPER_TAB / "gen_gain.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_p1_aggregate.py\n")
        f.write("% Worst-stream gain SizeAware - SizeBlindRev at N=4 bg L1_light\n")
        f.write("% on Composition A (sids 2, 22, 3, 21).\n")
        f.write("\\begin{tabular}{ll|ccc}\n\\toprule\n")
        f.write("Detector & family & inverts? & worst gain & worst/mean ratio \\\\\n")
        f.write("\\midrule\n")
        for det in DETECTORS:
            r = det_results.get(det["name"])
            disp = det["display"]
            family = "YOLOv11" if det["name"] == "yolo11s" else "YOLOv8"
            if r is None or r.get("gain") is None:
                f.write(f"{disp} & {family} & TBD & TBD & TBD \\\\\n")
                continue
            g = r["gain"]
            inv = "yes" if g["inverts"] else "no"
            ratio = g["worst_mean_ratio"]
            ratio_s = "n/a" if (ratio == float("inf") or ratio != ratio) else f"{ratio:+.1f}\\times"
            f.write(f"{disp} & {family} & {inv} & "
                    f"${g['worst_gain']:+.3f}$ & ${ratio_s}$ \\\\\n")
        f.write("--- & DETR    & --- & --- & --- (future work; no INT8 mxq) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {out}")


def render_cstar_figure(det_results):
    """Two-panel figure: (left) Δ_g(C) curves per detector grouped by group color;
    (right) C* vs detector params, per group."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    grp_colors = {"small-rich": "#3680c4",
                  "medium-mixed": "#3b9c4d",
                  "large-rich": "#c43b3b"}

    # Panel A: Δ_g(C) — one curve per (detector × group), color by group, alpha by detector
    ax = axes[0]
    det_alphas = np.linspace(0.35, 1.0, len(DETECTORS))
    for det, alpha in zip(DETECTORS, det_alphas):
        r = det_results.get(det["name"], {})
        cstar = r.get("cstar");
        if not cstar: continue
        for grp, info in cstar.items():
            curve = info.get("curve", [])
            if not curve: continue
            xs = [p["C"] for p in curve]
            ys = [p["delta"] for p in curve]
            ax.plot(xs, ys, "-", color=grp_colors[grp], alpha=alpha,
                    linewidth=1.4, marker="o", markersize=4,
                    label=f"{det['display']} {grp}")
            if info.get("Cstar_ms") is not None:
                ax.scatter([info["Cstar_ms"]], [0], color=grp_colors[grp],
                           marker="*", s=80, alpha=alpha, zorder=5,
                           edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xscale("symlog", linthresh=20)
    ax.set_xlabel("scalar contention proxy C  (GPU eff E2E latency, ms)")
    ax.set_ylabel(r"$\Delta_g(C) = \mathrm{sAP}_{\mathrm{NPU},g} - \mathrm{sAP}_{\mathrm{GPU},g}(C)$")
    ax.set_title("Δ_g(C) curves across detectors  (alpha = capacity)")
    ax.grid(alpha=0.3)
    # Custom legend: groups only
    handles = [plt.Line2D([], [], color=c, label=g) for g, c in grp_colors.items()]
    ax.legend(handles=handles, fontsize=9, loc="best")

    # Panel B: C* vs detector params per group
    ax = axes[1]
    for grp, color in grp_colors.items():
        xs = []; ys = []; lbls = []
        for det in DETECTORS:
            r = det_results.get(det["name"], {})
            cstar = r.get("cstar")
            if cstar and cstar.get(grp) and cstar[grp].get("Cstar_ms") is not None:
                xs.append(det["params_M"])
                ys.append(cstar[grp]["Cstar_ms"])
                lbls.append(det["display"])
        if xs:
            ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=10, label=grp)
            for x, y, l in zip(xs, ys, lbls):
                ax.annotate(l.replace("YOLOv", "v").replace(" (marker)", ""),
                            xy=(x, y), xytext=(4, 4), textcoords="offset points",
                            fontsize=6, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("detector capacity (parameters, M)")
    ax.set_ylabel("inversion threshold C* (ms)")
    ax.set_title("C* vs detector capacity per group (NEW sAP-gap definition)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle("Fig.~gen-cstar — YOLOv8 family + YOLOv11s marker (NEW C* via sAP-gap)",
                 fontsize=12)
    plt.tight_layout()
    out_paper = PAPER_FIG / "gen_cstar.pdf"
    out_proj  = FIG_PROJ / "gen_cstar.pdf"
    fig.savefig(out_paper, format="pdf", bbox_inches="tight")
    fig.savefig(out_proj,  format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_paper}")
    print(f"saved {out_proj}")


# ---------------------------------- main ------------------------------------

def main():
    det_results = {}
    summary_rows = []
    cstar_rows = []
    gain_rows = []
    for det in DETECTORS:
        name = det["name"]
        try:
            QL = compute_Q_L_per_det(name)
            cstar = compute_cstar_per_det(name)
            gain = compute_gain_per_det(name)
        except Exception as e:
            print(f"[aggregate] {name} fail: {e}")
            QL = cstar = gain = None
        det_results[name] = {"QL": QL, "cstar": cstar, "gain": gain}
        # summary rows
        if QL:
            summary_rows.append({
                "detector": name, "params_M": det["params_M"],
                **{f"Q_{s}": QL["Q"][s] for s in ["small", "medium", "large"]},
                **{f"L_{s}": QL["L"][s] for s in ["small", "medium", "large"]},
            })
        if cstar:
            for grp, info in cstar.items():
                cstar_rows.append({
                    "detector": name, "params_M": det["params_M"], "group": grp,
                    "sap_npu_g": info["sap_npu_g"],
                    "Cstar_ms": info["Cstar_ms"],
                })
        if gain:
            gain_rows.append({
                "detector": name, "params_M": det["params_M"],
                **gain,
            })

    pd.DataFrame(summary_rows).to_csv(RES / "gen_decomp_v2.csv", index=False)
    pd.DataFrame(cstar_rows).to_csv(RES / "cstar_v2_family.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(RES / "gen_gain_v2.csv", index=False)
    print(f"saved {RES / 'gen_decomp_v2.csv'}")
    print(f"saved {RES / 'cstar_v2_family.csv'}")
    print(f"saved {RES / 'gen_gain_v2.csv'}")

    write_gen_decomp_tex(det_results)
    write_gen_gain_tex(det_results)
    render_cstar_figure(det_results)

    print("\n=== gen_decomp_v2.csv ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\n=== cstar_v2_family.csv ===")
    print(pd.DataFrame(cstar_rows).to_string(index=False))
    print("\n=== gen_gain_v2.csv ===")
    print(pd.DataFrame(gain_rows).to_string(index=False))


if __name__ == "__main__":
    main()
