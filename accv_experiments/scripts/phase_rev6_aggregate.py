"""rev6 STEP 3 + STEP 4 — aggregate (absolute + relative) and emit paper artifacts.

Reads:
  results/p1r6_baseline_<det>.csv   for each yolo11{s,m,l,x}
  results/p1r6_ladder_<det>.csv
  results/p1r6_multistream_<det>.csv
  results/binary_plan_rev6.json     for binary-hash provenance per detector

Emits:
  results/gen_decomp_v5.csv         absolute Q, L per detector + relative
  results/relative_QL_v5.csv        relative Q, L per detector (primary view)
  results/cstar_v5.csv              sAP-gap C* per detector per group
  results/gen_gain_v5.csv           N=4 L1_light SizeAware vs SizeBlindRev
  results/relative_L_multistream_v5.csv   concurrency-induced L per detector
  paper/tables/gen_decomp.tex       primary: relative; absolute in parens
  paper/tables/gen_gain.tex         same 4-col layout, binary provenance footer
  paper/tables/gen_decomp_appendix_v8.tex  unchanged v8 supplement
  paper/tables/gen_gain_appendix_v8.tex    unchanged v8 supplement
  paper/figures/gen_cstar.pdf       v11 family rev6 binaries
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

RES = Path("accv_experiments/results")
PAPER_TAB = SCRIPT_DIR.parent.parent / "paper" / "tables"
PAPER_FIG = SCRIPT_DIR.parent.parent / "paper" / "figures"
FIG_PROJ  = SCRIPT_DIR.parent.parent / "figures"
PAPER_TAB.mkdir(parents=True, exist_ok=True)
PAPER_FIG.mkdir(parents=True, exist_ok=True)
FIG_PROJ.mkdir(parents=True, exist_ok=True)

PROBE = json.loads((RES / "binary_plan_rev6.json").read_text())

V11 = [
    {"name": "yolo11s", "params_M": 9.4,  "display": "YOLO11s",
     "binary": "legacy b2441f9d", "mode": "global8 (single-stream); single (multistream)"},
    {"name": "yolo11m", "params_M": 20.1, "display": "YOLO11m",
     "binary": "single " + PROBE["yolo11m"]["single_sha256"][:12], "mode": "single"},
    {"name": "yolo11l", "params_M": 25.3, "display": "YOLO11l",
     "binary": "single " + PROBE["yolo11l"]["single_sha256"][:12], "mode": "single"},
    {"name": "yolo11x", "params_M": 56.9, "display": "YOLO11x",
     "binary": "single " + PROBE["yolo11x"]["single_sha256"][:12], "mode": "single"},
]
LADDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
SIZE_GROUPS = {
    "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
    "medium-mixed": [0, 1, 6, 8, 10, 11, 17, 20],
    "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
}


# -------- IO --------

def load_baseline(det):
    return pd.read_csv(RES / f"p1r6_baseline_{det}.csv")


def load_ladder(det):
    base = load_baseline(det)
    g_l0 = base[(base.device == "GPU") & (base.bg_level == "L0")].copy()
    g_l0["bg_level"] = "L0"
    ldr = pd.read_csv(RES / f"p1r6_ladder_{det}.csv")
    cols = ["sid", "bg_level", "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
            "map_5095", "map_s", "map_m", "map_l", "eff_e2e_mean", "frame_skip_pct",
            "latency_mean"]
    return pd.concat([g_l0[[c for c in cols if c in g_l0.columns]],
                      ldr[[c for c in cols if c in ldr.columns]]],
                     ignore_index=True)


def load_multistream(det):
    return pd.read_csv(RES / f"p1r6_multistream_{det}.csv")


# -------- computations --------

def compute_QL(det):
    base = load_baseline(det)
    ldr = load_ladder(det)
    g_l0 = base[(base.device == "GPU") & (base.bg_level == "L0")].sort_values("sid").reset_index(drop=True)
    n_l0 = base[(base.device == "NPU") & (base.bg_level == "L0")].sort_values("sid").reset_index(drop=True)
    Q = {sz: float((n_l0[f"sap_{sz}"] - g_l0[f"sap_{sz}"]).mean()) for sz in ["s","m","l"]}
    g_l2 = ldr[ldr.bg_level == "L2_lm"].sort_values("sid").reset_index(drop=True)
    L = {}
    for sz in ["s","m","l"]:
        n = min(len(g_l0), len(g_l2))
        L[sz] = float((g_l0[f"sap_{sz}"].iloc[:n] - g_l2[f"sap_{sz}"].iloc[:n]).mean())
    return {"Q": Q, "L": L, "GPU_baseline": {sz: float(g_l0[f"sap_{sz}"].mean()) for sz in ["s","m","l"]}}


def compute_cstar(det):
    base = load_baseline(det)
    ldr = load_ladder(det)
    n_l0 = base[(base.device == "NPU") & (base.bg_level == "L0")]
    npu_sap_per_sid = {int(r["sid"]): float(r["sap_5095"]) for _, r in n_l0.iterrows()}
    gpu_by_bg = {}; C_per_bg = {}
    for bg in LADDER:
        sub = ldr[ldr.bg_level == bg]
        if not len(sub): continue
        gpu_by_bg[bg] = {int(r["sid"]): float(r["sap_5095"]) for _, r in sub.iterrows()}
        col = "eff_e2e_mean" if ("eff_e2e_mean" in sub.columns and sub["eff_e2e_mean"].notna().any()) else "latency_mean"
        C_per_bg[bg] = float(sub[col].mean())
    out = {}
    for grp, sids in SIZE_GROUPS.items():
        sap_npu_g = float(np.mean([npu_sap_per_sid[s] for s in sids if s in npu_sap_per_sid]))
        pts = []
        for bg in LADDER:
            if bg not in gpu_by_bg or bg not in C_per_bg: continue
            vals = [gpu_by_bg[bg][s] for s in sids if s in gpu_by_bg[bg]]
            if not vals: continue
            sap_gpu_g = float(np.mean(vals)); delta = sap_npu_g - sap_gpu_g
            pts.append({"bg": bg, "C": C_per_bg[bg], "sap_gpu_g": sap_gpu_g, "delta": delta})
        pts.sort(key=lambda x: x["C"])
        Cstar = None
        for i in range(len(pts) - 1):
            d1 = pts[i]["delta"]; d2 = pts[i+1]["delta"]
            C1 = pts[i]["C"]; C2 = pts[i+1]["C"]
            if d1 * d2 <= 0 and d1 != d2:
                Cstar = C1 + (-d1) / (d2 - d1) * (C2 - C1); break
        out[grp] = {"sap_npu_g": sap_npu_g, "Cstar_ms": Cstar, "curve": pts}
    return out


def compute_gain(det):
    ms = load_multistream(det)
    sa = ms[(ms.n_streams == 4) & (ms.placement_name == "SizeAware")]
    sb = ms[(ms.n_streams == 4) & (ms.placement_name == "SizeBlindRev")]
    if not len(sa) or not len(sb): return None
    wg = float(sa.iloc[0]["worst_sap"]) - float(sb.iloc[0]["worst_sap"])
    mg = float(sa.iloc[0]["mean_sap"]) - float(sb.iloc[0]["mean_sap"])
    ratio = wg / mg if abs(mg) > 1e-9 else float("inf")
    return {"sa_worst": float(sa.iloc[0]["worst_sap"]),
            "sb_worst": float(sb.iloc[0]["worst_sap"]),
            "sa_mean":  float(sa.iloc[0]["mean_sap"]),
            "sb_mean":  float(sb.iloc[0]["mean_sap"]),
            "worst_gain": wg, "mean_gain": mg,
            "worst_mean_ratio": ratio, "inverts": (wg > 0)}


def compute_multistream_L(det, gpu_baseline):
    """Concurrency-induced L per group at N=4 L1_light."""
    ms = load_multistream(det)
    ldr = load_ladder(det)
    n1_per_sid = {int(r["sid"]): {"sap_s": float(r["sap_s"]),
                                   "sap_m": float(r["sap_m"]),
                                   "sap_l": float(r["sap_l"])}
                  for _, r in ldr[ldr.bg_level == "L1_light"].iterrows()}
    def find_group(sid):
        for g, sids in SIZE_GROUPS.items():
            if sid in sids: return g
        return None
    rows = []
    for _, r in ms[ms.n_streams == 4].iterrows():
        placement = json.loads(r["placement_spec"])
        pname = r["placement_name"]
        for i, dev in enumerate(placement):
            if dev != "GPU": continue
            sid = int(r[f"s{i}_sid"])
            if sid not in n1_per_sid: continue
            for sz in ["s","m","l"]:
                n1 = n1_per_sid[sid][f"sap_{sz}"]
                n4 = float(r[f"s{i}_sap_{sz}"])
                L_multi = n1 - n4
                rel = L_multi / gpu_baseline[sz] if gpu_baseline[sz] else float("nan")
                rows.append({"detector": det, "placement": pname,
                             "sid": sid, "group": find_group(sid),
                             "size": sz, "L_multistream": L_multi,
                             "rel_L_multi": rel})
    return rows


# -------- table writers --------

def write_gen_decomp_tex():
    """Primary: relative Q and L. Absolute in parens."""
    out = PAPER_TAB / "gen_decomp.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev6_aggregate.py — YOLO11 family.\n")
        f.write("% Primary cells: relative_Q(s) = Q(s)/GPU_FP32_baseline(s);\n")
        f.write("%                relative_L(s, L2_lm) = L(s,L2_lm)/GPU_FP32_baseline_sAP(s).\n")
        f.write("% Each model unifies one mxq across phases (see RESULTS_STATUS.md rev 6).\n")
        f.write("\\begin{tabular}{ll|ccc|ccc}\n\\toprule\n")
        f.write("& & \\multicolumn{3}{c|}{relative $Q(s)$ (quantization)} & "
                "\\multicolumn{3}{c}{relative $L(s,\\text{L2\\_lm})$ (staleness)} \\\\\n")
        f.write("Detector & params & small & medium & large & small & medium & large \\\\\n")
        f.write("\\midrule\n")
        for det in V11:
            r = DET_RESULTS[det["name"]]
            QL = r["QL"]; Q = QL["Q"]; L = QL["L"]; b = QL["GPU_baseline"]
            def rel(v, base): return v / base if base else float("nan")
            def fmt(v): return f"${v*100:+.1f}\\%$"
            f.write(f"{det['display']} & {det['params_M']}\\,M & "
                    f"{fmt(rel(Q['s'],b['s']))} & {fmt(rel(Q['m'],b['m']))} & {fmt(rel(Q['l'],b['l']))} & "
                    f"{fmt(rel(L['s'],b['s']))} & {fmt(rel(L['m'],b['m']))} & {fmt(rel(L['l'],b['l']))} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {out}")


def write_gen_decomp_absolute_tex():
    """Companion table (absolute units) — kept as gen_decomp_absolute.tex."""
    out = PAPER_TAB / "gen_decomp_absolute.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated companion to gen_decomp.tex — absolute sAP units.\n")
        f.write("\\begin{tabular}{ll|ccc|ccc}\n\\toprule\n")
        f.write("& & \\multicolumn{3}{c|}{absolute $Q(s)$} & "
                "\\multicolumn{3}{c}{absolute $L(s,\\text{L2\\_lm})$} \\\\\n")
        f.write("Detector & params & small & medium & large & small & medium & large \\\\\n")
        f.write("\\midrule\n")
        for det in V11:
            r = DET_RESULTS[det["name"]]
            Q = r["QL"]["Q"]; L = r["QL"]["L"]
            def fmt(v): return f"${v:+.3f}$"
            f.write(f"{det['display']} & {det['params_M']}\\,M & "
                    f"{fmt(Q['s'])} & {fmt(Q['m'])} & {fmt(Q['l'])} & "
                    f"{fmt(L['s'])} & {fmt(L['m'])} & {fmt(L['l'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {out}")


def write_gen_gain_tex():
    out = PAPER_TAB / "gen_gain.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by phase_rev6_aggregate.py.\n")
        f.write("% N=4 bg L1_light Composition A. NPU phase = single mode per-detector;\n")
        f.write("% v11s uses legacy mxq, m/l/x use the published single-mode mxq.\n")
        f.write("\\begin{tabular}{ll|cccc}\n\\toprule\n")
        f.write("Detector & family & inverts? & worst gain & mean gain & worst/mean \\\\\n")
        f.write("\\midrule\n")
        for det in V11:
            r = DET_RESULTS[det["name"]]
            g = r["gain"]
            if g is None:
                f.write(f"{det['display']} & YOLO11 & TBD & TBD & TBD & TBD \\\\\n")
                continue
            inv = "yes" if g["inverts"] else "no"
            wg_s = f"\\textbf{{${g['worst_gain']:+.3f}$}}"
            mg_s = f"${g['mean_gain']:+.3f}$"
            ratio = g["worst_mean_ratio"]
            if ratio == float("inf") or ratio != ratio:
                ratio_s = "$\\infty$"
            else:
                ratio_s = f"${ratio:+.1f}\\times$"
            f.write(f"{det['display']} & YOLO11 & {inv} & {wg_s} & {mg_s} & {ratio_s} \\\\\n")
        f.write("\\midrule\n")
        f.write("RT-DETR / PicoDet & DETR/anchor-free & \\multicolumn{4}{c}{\\emph{future work (no INT8 NPU export available)}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {out}")


def render_cstar(out_path):
    grp_colors = {"small-rich":"#3680c4", "medium-mixed":"#3b9c4d", "large-rich":"#c43b3b"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    alphas = np.linspace(0.4, 1.0, len(V11))
    for det, alpha in zip(V11, alphas):
        cstar = DET_RESULTS[det["name"]]["cstar"]
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
    ax.set_title("Δ_g(C) curves across YOLO11 family (rev 6, unified binary per detector)")
    ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=c, label=g) for g, c in grp_colors.items()]
    ax.legend(handles=handles, fontsize=9, loc="best")
    ax = axes[1]
    for grp, color in grp_colors.items():
        xs = []; ys = []; lbls = []
        for det in V11:
            cstar = DET_RESULTS[det["name"]]["cstar"]
            if cstar.get(grp) and cstar[grp]["Cstar_ms"] is not None:
                xs.append(det["params_M"]); ys.append(cstar[grp]["Cstar_ms"])
                lbls.append(det["display"])
        if xs:
            ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=10, label=grp)
            for x, y, l in zip(xs, ys, lbls):
                ax.annotate(l.replace("YOLO11","v11"),
                            xy=(x, y), xytext=(4, 4), textcoords="offset points",
                            fontsize=6, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("detector capacity (parameters, M)")
    ax.set_ylabel("inversion threshold C* (ms)")
    ax.set_title("C* vs capacity (rev 6 unified-binary sweep)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.suptitle("YOLO11 family — rev 6 single-binary sweep (sAP-gap C*)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    fig.savefig(FIG_PROJ / "gen_cstar.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# -------- main --------

DET_RESULTS = {}

def main():
    summary = []
    cstar_rows = []
    gain_rows = []
    ms_L_rows = []
    rel_rows = []
    for det in V11:
        QL = compute_QL(det["name"])
        cstar = compute_cstar(det["name"])
        gain = compute_gain(det["name"])
        ms_L = compute_multistream_L(det["name"], QL["GPU_baseline"])
        DET_RESULTS[det["name"]] = {"QL": QL, "cstar": cstar, "gain": gain, "msL": ms_L}
        b = QL["GPU_baseline"]
        summary.append({
            "detector": det["name"], "params_M": det["params_M"],
            "binary": det["binary"], "mode": det["mode"],
            **{f"Q_{s}": QL["Q"][s] for s in ["s","m","l"]},
            **{f"L_{s}": QL["L"][s] for s in ["s","m","l"]},
            **{f"GPU_baseline_{s}": b[s] for s in ["s","m","l"]},
            **{f"rel_Q_{s}": QL["Q"][s] / b[s] if b[s] else None for s in ["s","m","l"]},
            **{f"rel_L_{s}": QL["L"][s] / b[s] if b[s] else None for s in ["s","m","l"]},
        })
        rel_rows.append({"detector": det["name"], "params_M": det["params_M"],
                          "binary": det["binary"],
                          **{f"rel_Q_{s}": QL["Q"][s] / b[s] for s in ["s","m","l"]},
                          **{f"rel_L_{s}": QL["L"][s] / b[s] for s in ["s","m","l"]}})
        for grp, info in cstar.items():
            cstar_rows.append({"detector": det["name"], "params_M": det["params_M"],
                                "group": grp, "sap_npu_g": info["sap_npu_g"],
                                "Cstar_ms": info["Cstar_ms"]})
        if gain:
            gain_rows.append({"detector": det["name"], "params_M": det["params_M"], **gain})
        ms_L_rows.extend(ms_L)
    pd.DataFrame(summary).to_csv(RES / "gen_decomp_v5.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(RES / "relative_QL_v5.csv", index=False)
    pd.DataFrame(cstar_rows).to_csv(RES / "cstar_v5.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(RES / "gen_gain_v5.csv", index=False)
    pd.DataFrame(ms_L_rows).to_csv(RES / "relative_L_multistream_v5.csv", index=False)
    print(f"\nsaved gen_decomp_v5, relative_QL_v5, cstar_v5, gen_gain_v5, "
          f"relative_L_multistream_v5 to {RES}")

    print("\n=== YOLO11 family Q + L (absolute) ===")
    for r in summary:
        print(f"  {r['detector']:<10s}  Q={r['Q_s']:+.4f}/{r['Q_m']:+.4f}/{r['Q_l']:+.4f}  "
              f"L={r['L_s']:+.4f}/{r['L_m']:+.4f}/{r['L_l']:+.4f}")
    print("\n=== YOLO11 family relative Q + L ===")
    for r in rel_rows:
        print(f"  {r['detector']:<10s}  rel_Q={r['rel_Q_s']*100:+.1f}%/{r['rel_Q_m']*100:+.1f}%/{r['rel_Q_l']*100:+.1f}%  "
              f"rel_L={r['rel_L_s']*100:+.1f}%/{r['rel_L_m']*100:+.1f}%/{r['rel_L_l']*100:+.1f}%")
    print("\n=== C* per detector per group (ms) ===")
    for r in cstar_rows:
        cs = r["Cstar_ms"]
        cs_s = f"{cs:.1f}" if cs is not None else "—"
        print(f"  {r['detector']:<10s} {r['group']:<14s}  sap_npu_g={r['sap_npu_g']:.3f}  C*={cs_s}")
    print("\n=== gen_gain (N=4 L1_light) ===")
    for r in gain_rows:
        print(f"  {r['detector']:<10s}  worst={r['worst_gain']:+.3f}  mean={r['mean_gain']:+.3f}  "
              f"ratio={r['worst_mean_ratio']:+.2f}  inverts={r['inverts']}")

    write_gen_decomp_tex()
    write_gen_decomp_absolute_tex()
    write_gen_gain_tex()
    render_cstar(PAPER_FIG / "gen_cstar.pdf")


if __name__ == "__main__":
    main()
