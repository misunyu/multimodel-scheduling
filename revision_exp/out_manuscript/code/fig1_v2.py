"""Fig. 1(a)/(b) v2 in the original manuscript format. Writes fig1a_v2.* and fig1b_v2.* only (existing figures untouched).

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp/out_manuscript/code/fig1_v2.py

Sources (only): out_reprofiled/per_workload_x3.csv (workload list, All-GPU rows), revision_profiling/dataset/labels_test.csv
(F, T, D_macro per placement), out_manuscript/fig1b_data.csv (cross-check). Style: revision_exp/plots.py _style, 8.5 cm column.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "revision_exp"))

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import plots as PL  # noqa: E402  (style only)

OUT = REPO / "revision_exp" / "out_manuscript"
W = 8.5 / 2.54
ALPHA = 0.3
TOL = 1e-9
FAM7 = "gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19"
ORDER = ["gpt2", "mnasnet", "resnet50", "resnext50", "shufflenet-v2-12", "squeezenet1.0-12", "vgg19", "yolov4"]

lab = pd.read_csv(REPO / "revision_profiling" / "dataset" / "labels_test.csv")
pw = pd.read_csv(REPO / "revision_exp" / "out_reprofiled" / "per_workload_x3.csv")
f1b = pd.read_csv(OUT / "fig1b_data.csv")
for name in ["fig1a_v2", "fig1b_v2"]:
    for ext in ["pdf", "png", "csv"]:
        assert not (OUT / f"{name}.{ext}").exists(), f"{name}.{ext} exists; refusing to overwrite"


def axes_style(ax):
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.1)
    for s in ax.spines.values():
        s.set_linewidth(0.3)
    ax.tick_params(labelsize=6, width=0.3, length=2)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


# ---------------- Fig. 1(a) v2: All-GPU vs throughput-optimal (normalized throughput only) ----------------
def fig1a_v2():
    PL._style()
    groups = pw[(pw["N"] >= 3) & (pw["strategy"] == "All-GPU")][["group", "N", "selected"]]
    assert len(groups) == 12 and groups["N"].value_counts().sort_index().tolist() == [6, 1, 1, 2, 2]
    rows = []
    for _, g in groups.iterrows():
        fam = g["group"].replace("_x3.yaml", "")
        w = lab[(lab["family"] == fam) & (lab["multiplier"] == 3)]
        assert len(w) == 2 ** g["N"]
        ag = w[w["placement"] == "G" * g["N"]]
        assert len(ag) == 1
        ag = ag.iloc[0]
        assert int(g["selected"].split("_")[1]) == ag["comb_num"]  # per_workload_x3 All-GPU row = same placement
        fmax = w["F_mean"].max()
        assert abs(fmax - w["F_max"].iloc[0]) < TOL
        top = w[w["F_mean"] >= fmax - TOL].sort_values("comb_num")
        n_tied = len(top)  # exact F ties (same mean count): lowest combination number, as in the evaluator
        allgpu_in_tie = bool((top["combination"] == ag["combination"]).any())
        top = top.iloc[0]
        t_ag = ag["F_mean"] / fmax
        assert abs(t_ag - ag["throughput_norm"]) < TOL
        rows.append({"workload": fam, "N": int(g["N"]), "F_allgpu": ag["F_mean"], "F_max": fmax, "T_allgpu": t_ag, "T_throughput_optimal": 1.0,
                     "allgpu_combination": ag["combination"], "throughput_optimal_combination": top["combination"],
                     "throughput_optimal_placement": top["placement"], "n_F_max_tied": n_tied, "allgpu_attains_F_max": allgpu_in_tie,
                     "S_macro_allgpu": ag["S_macro_a0.3"], "S_macro_throughput_optimal": top["S_macro_a0.3"],
                     "S_macro_oracle": w["S_macro_a0.3"].max()})
    d = pd.DataFrame(rows).sort_values(["N", "T_allgpu"]).reset_index(drop=True)
    xs = []
    for n, g in d.groupby("N", sort=True):
        k = len(g)
        off = np.zeros(1) if k == 1 else np.linspace(-0.3, 0.3, k)
        xs += list(n + off)
    d["x"] = xs
    d.round(6).to_csv(OUT / "fig1a_v2.csv", index=False)

    fig, ax = plt.subplots(figsize=(W, W * 0.62))
    for _, r in d.iterrows():
        ax.plot([r["x"], r["x"]], [r["T_allgpu"], 1.0], color="0.55", linewidth=0.5, zorder=1)
    ax.scatter(d["x"], d["T_allgpu"], marker="v", s=12, color="tab:orange", edgecolor="black", linewidth=0.3, zorder=3, label="All-GPU")
    ax.scatter(d["x"], np.ones(len(d)), marker="D", s=10, color="black", linewidth=0.3, zorder=3, label="Throughput-optimal")
    ax.set_xticks([3, 4, 5, 6, 7])
    ax.set_xlim(2.5, 7.5)
    lo = d["T_allgpu"].min()
    ax.set_ylim(np.floor((lo - 0.01) * 50) / 50, 1.01)
    ax.set_xlabel("Number of application instances (N)", fontsize=6)
    ax.set_ylabel("Normalized throughput", fontsize=6)
    axes_style(ax)
    ax.legend(loc="lower left", fontsize=5.5, frameon=False, handletextpad=0.3, borderaxespad=0.3)
    fig.tight_layout()
    save(fig, "fig1a_v2")
    return d


# ---------------- Fig. 1(b) v2: T and D of the two placements ----------------
def fig1b_v2():
    PL._style()
    w = lab[(lab["family"] == FAM7) & (lab["multiplier"] == 3)].copy()
    assert len(w) == 128
    w["S"] = w["throughput_norm"] - ALPHA * w["drop_rate_norm"]
    assert (w["S"] - w["S_macro_a0.3"]).abs().max() < 1e-12
    picks = {"Throughput-maximizing": "combination_108", "Drop-aware (max $T-0.3D$)": "combination_079"}
    fmax = w["F_mean"].max()
    a = w[w["F_mean"] >= fmax - TOL]
    assert a["combination"].tolist() == ["combination_108"], "combination_108 is not the unique F maximum"
    smax = w["S"].max()
    b = w[w["S"] >= smax - TOL]
    assert b["combination"].tolist() == ["combination_079"], "combination_079 is not the unique S_macro maximum"
    rows, check = [], []
    for role, comb in picks.items():
        r = w[w["combination"] == comb].iloc[0]
        ref = f1b[f1b["combination"] == comb].iloc[0]
        for col, rc in [("F_mean", "F_mean_req_s"), ("throughput_norm", "T"), ("drop_rate_norm", "D_macro")]:
            diff = abs(r[col] - ref[rc])
            check.append({"combination": comb, "field": col, "labels_test": r[col], "fig1b_data": ref[rc], "abs_diff": diff})
            assert diff < 1e-5, f"{comb} {col} mismatch {r[col]} vs {ref[rc]}"
        inst = f1b[(f1b["combination"] == comb)].drop_duplicates("instance").set_index("instance")
        cfg = "".join(inst.loc[m, "device"][0] for m in ORDER if m in inst.index)
        assert cfg == r["placement"], (cfg, r["placement"])
        rows.append({"role": role.split(" (")[0], "combination": comb, "placement(GC per sorted model)": r["placement"], "F": r["F_mean"],
                     "T": r["throughput_norm"], "D_macro": r["drop_rate_norm"], "S": r["S"]})
    expect = {"combination_108": (1.000, 0.102), "combination_079": (0.981, 0.014)}
    for r in rows:
        assert round(r["T"], 3) == expect[r["combination"]][0] and round(r["D_macro"], 3) == expect[r["combination"]][1]
    d = pd.DataFrame(rows)
    d.round(6).to_csv(OUT / "fig1b_v2.csv", index=False)

    colors = [("skyblue", "//"), ("lightcoral", "..")]
    labels = list(picks)
    width = 0.36

    def bars(ax, col, xc, fmt="{:.3f}"):
        for i, (r, (c, h)) in enumerate(zip(rows, colors)):
            x = xc + (i - 0.5) * width
            ax.bar(x, r[col], width, color=c, alpha=0.6, hatch=h, edgecolor="black", linewidth=0.3, label=labels[i])
            ax.text(x, r[col], fmt.format(r[col]), ha="center", va="bottom", fontsize=5.5)

    # single grouped panel first; split if a D bar is too short to read at print size
    fig_h = W * 0.55
    ax_h_cm = fig_h * 2.54 * 0.75
    min_d_mm = min(r["D_macro"] for r in rows) / 1.1 * ax_h_cm * 10
    split = min_d_mm < 1.0
    if not split:
        fig, ax = plt.subplots(figsize=(W, fig_h))
        bars(ax, "T", 0)
        bars(ax, "D_macro", 1)
        ax.set_xticks([0, 1], ["Normalized throughput $T$", "Mean drop fraction $D$"])
        ax.set_ylim(0, 1.12)
        axes_style(ax)
        ax.legend(loc="upper right", fontsize=5.5, frameon=False, handletextpad=0.3)
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, fig_h))
        bars(a1, "T", 0)
        bars(a2, "D_macro", 0)
        a1.set_ylim(0, 1.15)
        a2.set_ylim(0, max(r["D_macro"] for r in rows) * 1.2)
        a1.set_xticks([0], ["Normalized throughput $T$"])
        a2.set_xticks([0], ["Mean drop fraction $D$"])
        for ax in (a1, a2):
            axes_style(ax)
        h, l = a1.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=5.5, frameon=False, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "fig1b_v2")
    return d, pd.DataFrame(check), {"split_panels": bool(split), "min_D_bar_height_mm_single_panel": min_d_mm}


if __name__ == "__main__":
    a = fig1a_v2()
    b, chk, lay = fig1b_v2()
    print(a.to_string())
    print(b.to_string())
    print(chk.to_string())
    print(json.dumps(lay))
