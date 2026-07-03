"""rev30 aggregation — derive ALL tables/figures from the single canonical CSV.

Input : results/rev30_clean_resnet/rev30_raw.csv  (+ util_samples.csv)
Output: rev30_canonical.csv (raw + per-cell GPU util sliced from util_samples),
        rev30_oracle_by_contention.csv, rev30_fig3a_points.csv, rev30_fig3b_points.csv,
        rev30_persize.csv, rev30_tabmain_L1CNN.tex, persize_sweep.pdf,
        rev30_fig3a_tikz.txt, rev30_fig3b_tikz.txt, rev30_boundary_report.md
No hand transcription: every emitted number comes from rev30_raw.csv.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("accv_experiments/results/rev30_clean_resnet")
RAW = RES / "rev30_raw.csv"
UTIL = RES / "util_samples.csv"
PERIOD = 1000.0 / 30.0
KS = [0, 1, 2, 3, 4, 6, 8]


def ratio_of(placement: str) -> int:
    return placement.count("N")


def load():
    df = pd.read_csv(RAW)
    df["ratio"] = df["placement"].apply(ratio_of)
    return df


# ---------- 1. per-cell GPU util sliced from continuous sampler ----------
def attach_util(df):
    cols = ["util_mean", "util_p50", "util_p95", "mem_used_mib", "power_w", "n_util_samples"]
    if not UTIL.exists():
        for c in cols: df[c] = np.nan
        return df
    u = pd.read_csv(UTIL)
    out = {c: [] for c in cols}
    for _, r in df.iterrows():
        w = u[(u["epoch"] >= r["t_start"] + 1.0) & (u["epoch"] <= r["t_end"])]  # drop 1s warm-up
        if len(w):
            out["util_mean"].append(round(w["util_gpu"].mean(), 1))
            out["util_p50"].append(round(w["util_gpu"].median(), 1))
            out["util_p95"].append(round(w["util_gpu"].quantile(0.95), 1))
            out["mem_used_mib"].append(int(w["mem_used_mib"].mean()))
            out["power_w"].append(round(w["power_w"].mean(), 1))
            out["n_util_samples"].append(len(w))
        else:
            for c in cols: out[c].append(np.nan)
    for c in cols: df[c] = out[c]
    return df


# ---------- 2. Fig 3b: Oracle by contention (N=4) + picked-ratio distribution ----------
def oracle_table(df):
    n4 = df[df["group"] == "N4"]
    rows, pick_dist = [], {}
    for k in KS:
        sub = n4[n4["k"] == k]
        if sub.empty: continue
        # mean worst per ratio
        mw = {r: sub[sub["ratio"] == r]["worst_sap"].mean() for r in range(5)}
        sw = {r: sub[sub["ratio"] == r]["worst_sap"].std(ddof=0) for r in range(5)}
        o_r = max(mw, key=mw.get)
        allgpu_skip = sub[sub["ratio"] == 0]["gpu_skip_pct"].mean()
        # per-rep pick (boundary instability)
        picks = []
        for rep, g in sub.groupby("rep"):
            wr = {r: g[g["ratio"] == r]["worst_sap"].mean() for r in range(5) if (g["ratio"] == r).any()}
            if len(wr) == 5:
                picks.append(max(wr, key=wr.get))
        pick_dist[k] = picks
        rows.append({
            "contention_level": f"RES{k}", "resnet_k": k,
            "gpu_skip_allgpu": round(allgpu_skip, 1),
            "allgpu_worst": round(mw[0], 4), "allgpu_worst_std": round(sw[0], 4),
            "allnpu_worst": round(mw[4], 4), "allnpu_worst_std": round(sw[4], 4),
            "oracle_worst": round(mw[o_r], 4), "oracle_worst_std": round(sw[o_r], 4),
            "oracle_ratio": o_r,
            "split1_worst": round(mw[1], 4), "split2_worst": round(mw[2], 4), "split3_worst": round(mw[3], 4),
            "pick_dist": json.dumps({int(r): picks.count(r) for r in range(5)}),
            "reps": len(sub["rep"].unique()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev30_oracle_by_contention.csv", index=False)
    return out, pick_dist


# ---------- 3. Fig 3a points: homogeneous N=2/4/8 ----------
def fig3a_table(df):
    rows = []
    for N, grp in [(2, "N2"), (4, "N4"), (8, "N8")]:
        g = df[df["group"] == grp]
        for k in sorted(g["k"].unique()):
            ag = g[(g["k"] == k) & (g["ratio"] == 0)]      # All-GPU
            an = g[(g["k"] == k) & (g["placement"].str.fullmatch("N+"))]  # All-NPU
            row = {"N": N, "k": k}
            if not ag.empty:
                row["gpu_skip_mean"] = round(ag["gpu_skip_pct"].mean(), 1)
                row["allgpu_worst"] = round(ag["worst_sap"].mean(), 4)
                row["allgpu_worst_std"] = round(ag["worst_sap"].std(ddof=0), 4)
            if not an.empty:
                row["allnpu_worst"] = round(an["worst_sap"].mean(), 4)
                row["allnpu_worst_std"] = round(an["worst_sap"].std(ddof=0), 4)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev30_fig3a_points.csv", index=False)
    return out


# ---------- 4. Fig 3b points (mean±std) ----------
def fig3b_points(oracle):
    cols = ["resnet_k", "gpu_skip_allgpu", "allgpu_worst", "allgpu_worst_std",
            "allnpu_worst", "allnpu_worst_std", "oracle_worst", "oracle_worst_std", "oracle_ratio"]
    oracle[cols].to_csv(RES / "rev30_fig3b_points.csv", index=False)
    return oracle[cols]


# ---------- 5. per-size loss (N=4 All-GPU) ----------
def persize_table(df):
    g = df[(df["group"] == "N4") & (df["ratio"] == 0)]
    base = g[g["k"] == 0]
    b_s, b_m, b_l = base["sap_small"].mean(), base["sap_medium"].mean(), base["sap_large"].mean()
    rows = []
    for k in KS:
        sub = g[g["k"] == k]
        if sub.empty: continue
        s, m, l = sub["sap_small"].mean(), sub["sap_medium"].mean(), sub["sap_large"].mean()
        rows.append({"k": k, "gpu_skip": round(sub["gpu_skip_pct"].mean(), 1),
                     "sap_small": round(s, 4), "sap_medium": round(m, 4), "sap_large": round(l, 4),
                     "loss_small": round(b_s - s, 4), "loss_medium": round(b_m - m, 4),
                     "loss_large": round(b_l - l, 4)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rev30_persize.csv", index=False)
    return out


# ---------- 6a. tab:main L1CNN row (auto-emit) ----------
def tabmain_row(oracle):
    r1 = oracle[oracle["resnet_k"] == 1].iloc[0]
    df = load()
    n4k1 = df[(df["group"] == "N4") & (df["k"] == 1)]
    ag = n4k1[n4k1["ratio"] == 0]; an = n4k1[n4k1["ratio"] == 4]
    o = n4k1[n4k1["ratio"] == int(r1["oracle_ratio"])]
    def mean_pair(sub): return sub["worst_sap"].mean(), sub["mean_sap"].mean()
    agw, agm = mean_pair(ag); anw, anm = mean_pair(an); ow, om = mean_pair(o)
    gskip = ag["gpu_skip_pct"].mean(); nskip = an["npu_skip_pct"].mean()
    # ResNet (L1CNN) | All-GPU w/m | All-NPU w/m | Oracle w/m | gpu/npu skip
    tex = (f"ResNet50 & {agw:.3f} & {agm:.3f} & {anw:.3f} & {anm:.3f} & "
           f"{ow:.3f} & {om:.3f} & {gskip:.0f}/{nskip:.0f} \\\\")
    note = (f"% auto-emitted by phase_rev30_aggregate.py from rev30_raw.csv (N=4,k=1)\n"
            f"% Oracle ratio (NPU streams) = {int(r1['oracle_ratio'])}; "
            f"All-GPU worst std={ag['worst_sap'].std(ddof=0):.4f}, "
            f"Oracle worst std={o['worst_sap'].std(ddof=0):.4f}, reps={len(ag)}\n")
    (RES / "rev30_tabmain_L1CNN.tex").write_text(note + tex + "\n")
    return tex


# ---------- 6b. figures ----------
def fig_persize(ps):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = ps["gpu_skip"]
    ax.plot(x, ps["loss_large"], "-o", label="large", color="tab:red")
    ax.plot(x, ps["loss_medium"], "-s", label="medium", color="tab:orange")
    ax.plot(x, ps["loss_small"], "-^", label="small", color="tab:blue")
    ax.set_xlabel("GPU frame skip (%)"); ax.set_ylabel("sAP loss vs no contention")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    # NO title (caption handled by tex)
    fig.tight_layout(); fig.savefig(RES / "persize_sweep.pdf"); plt.close(fig)


def fig3a_tikz(f3a):
    lines = []
    for N, color, mark in [(2, "red", "*"), (4, "orange", "square*"), (8, "brown", "triangle*")]:
        sub = f3a[(f3a["N"] == N) & (f3a["allgpu_worst"].notna())].sort_values("gpu_skip_mean")
        coords = "".join(f"({r.gpu_skip_mean},{r.allgpu_worst:.4f})" for r in sub.itertuples())
        lines.append(f"\\addplot[{color}, mark={mark}] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{All-GPU $N{{=}}{N}$}}")
    npu = f3a[f3a["allnpu_worst"].notna()]["allnpu_worst"].mean()
    lines.append(f"\\addplot[blue, dashed, mark=none] coordinates {{(0,{npu:.4f})(90,{npu:.4f})}};")
    lines.append("\\addlegendentry{All-NPU}")
    (RES / "rev30_fig3a_tikz.txt").write_text("\n".join(lines) + "\n")


def fig3b_tikz(f3b):
    def coords(col): return "".join(f"({r.gpu_skip_allgpu},{getattr(r, col):.4f})" for r in f3b.itertuples())
    txt = (f"\\addplot[orange, mark=square*] coordinates {{{coords('allgpu_worst')}}};\n"
           f"\\addlegendentry{{All-GPU}}\n"
           f"\\addplot[blue, mark=*] coordinates {{{coords('allnpu_worst')}}};\n"
           f"\\addlegendentry{{All-NPU}}\n"
           f"\\addplot[teal, mark=diamond*, densely dashed] coordinates {{{coords('oracle_worst')}}};\n"
           f"\\addlegendentry{{Oracle (split)}}\n")
    (RES / "rev30_fig3b_tikz.txt").write_text(txt)


# ---------- 7. boundary instability report ----------
def boundary_report(df, oracle, pick_dist):
    n4k1 = df[(df["group"] == "N4") & (df["k"] == 1)]
    ag = n4k1[n4k1["ratio"] == 0]; an = n4k1[n4k1["ratio"] == 4]
    o_r = int(oracle[oracle["resnet_k"] == 1]["oracle_ratio"].iloc[0])
    o = n4k1[n4k1["ratio"] == o_r]
    picks = pick_dist.get(1, [])
    dist = {r: picks.count(r) for r in range(5)}
    md = ["# rev30 boundary-instability report (N=4, k=1, 33.333 ms deadline)\n",
          f"Reps = {len(ag)}.\n",
          "| metric | mean | std | min | max |",
          "|---|---|---|---|---|",
          f"| GPU frame skip (%) | {ag['gpu_skip_pct'].mean():.1f} | {ag['gpu_skip_pct'].std(ddof=0):.1f} | {ag['gpu_skip_pct'].min():.1f} | {ag['gpu_skip_pct'].max():.1f} |",
          f"| All-GPU worst sAP | {ag['worst_sap'].mean():.4f} | {ag['worst_sap'].std(ddof=0):.4f} | {ag['worst_sap'].min():.4f} | {ag['worst_sap'].max():.4f} |",
          f"| All-NPU worst sAP | {an['worst_sap'].mean():.4f} | {an['worst_sap'].std(ddof=0):.4f} | {an['worst_sap'].min():.4f} | {an['worst_sap'].max():.4f} |",
          f"| Oracle worst sAP (r={o_r}) | {o['worst_sap'].mean():.4f} | {o['worst_sap'].std(ddof=0):.4f} | {o['worst_sap'].min():.4f} | {o['worst_sap'].max():.4f} |",
          f"| deadline margin (ms, GPU) | {ag['deadline_margin_ms_mean'].mean():.2f} | {ag['deadline_margin_ms_mean'].std(ddof=0):.2f} | {ag['deadline_margin_ms_mean'].min():.2f} | {ag['deadline_margin_ms_mean'].max():.2f} |",
          "",
          f"**Oracle picked-ratio distribution over {len(picks)} reps** (ratio = #streams on NPU): {dist}",
          "",
          "Interpretation: a wide GPU-skip spread and a multi-modal picked-ratio distribution at k=1",
          "show the *optimal placement itself* is unstable at the deadline boundary — a quantified",
          "finding, reported as mean±std, not a hidden weakness.",
          "",
          "## Cross-check against the old scattered runs (same nominal point)",
          f"- rev21/rev20 (old tab:main): Oracle 0.1086, All-GPU 0.0915, skip 35.0",
          f"- rev29 (old Fig3b): Oracle 0.0910, All-GPU 0.0887, skip 37.8",
          f"- rev30 k=1: Oracle {o['worst_sap'].mean():.4f}±{o['worst_sap'].std(ddof=0):.4f}, "
          f"All-GPU {ag['worst_sap'].mean():.4f}±{ag['worst_sap'].std(ddof=0):.4f}, "
          f"skip {ag['gpu_skip_pct'].mean():.1f}±{ag['gpu_skip_pct'].std(ddof=0):.1f}",
          f"- Old Oracle values inside rev30 [min,max]=[{o['worst_sap'].min():.4f},{o['worst_sap'].max():.4f}]? "
          f"0.1086:{'YES' if o['worst_sap'].min() <= 0.1086 <= o['worst_sap'].max() else 'NO'}  "
          f"0.0910:{'YES' if o['worst_sap'].min() <= 0.0910 <= o['worst_sap'].max() else 'NO'}",
          ""]
    (RES / "rev30_boundary_report.md").write_text("\n".join(md))


def main():
    df = load()
    df = attach_util(df)
    df.to_csv(RES / "rev30_canonical.csv", index=False)
    oracle, pick_dist = oracle_table(df)
    f3a = fig3a_table(df)
    f3b = fig3b_points(oracle)
    ps = persize_table(df)
    tex = tabmain_row(oracle)
    fig_persize(ps); fig3a_tikz(f3a); fig3b_tikz(f3b)
    boundary_report(df, oracle, pick_dist)
    print("=== rev30 aggregate done ===")
    print("tab:main L1CNN ->", tex)
    print("canonical rows:", len(df))
    print(oracle[["contention_level", "gpu_skip_allgpu", "allgpu_worst", "allnpu_worst",
                  "oracle_worst", "oracle_ratio", "pick_dist"]].to_string(index=False))


if __name__ == "__main__":
    main()
