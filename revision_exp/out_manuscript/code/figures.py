"""Manuscript figures 1(a), 1(b), 3, 4 from the fixed results (no recomputation of any model).

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp/out_manuscript/code/figures.py

Style: revision_exp/plots.py (_style: serif / Times New Roman, thin lines), single-column width 8.5 cm.
Sources: out_reprofiled/{per_workload_x3,oracle_definitions,oracle_definitions_instances,fig3_data}.csv,
         revision_profiling/dataset/{labels_test,labels_instances_all}.csv, out_phase3/fig4_noclip/fig4_data.csv.
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
P2 = REPO / "revision_exp" / "out_reprofiled"
DS = REPO / "revision_profiling" / "dataset"
W = 8.5 / 2.54
ALPHA = 0.3
ABBR = {"gpt2": "g", "mnasnet": "m", "resnet50": "r", "resnext50": "x", "shufflenet-v2-12": "s", "squeezenet1.0-12": "q", "vgg19": "v", "yolov4": "y"}
NAME = {"gpt2": "GPT-2", "mnasnet": "MnasNet", "resnet50": "ResNet50", "resnext50": "ResNeXt50", "shufflenet-v2-12": "ShuffleNet", "squeezenet1.0-12": "SqueezeNet",
        "vgg19": "VGG19", "yolov4": "YOLOv4"}


def axes_style(ax):
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.1)
    for s in ax.spines.values():
        s.set_linewidth(0.3)
    ax.tick_params(labelsize=6, width=0.3, length=2)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def fam(group):
    return group.replace("_x3.yaml", "")


def label(family):
    return "+".join(ABBR[m] for m in family.split("+"))


def fig1a():
    p = pd.read_csv(P2 / "per_workload_x3.csv")
    p = p[p["N"] >= 3]
    gpu = p[p["strategy"] == "All-GPU"].set_index("group")
    orc = p[p["strategy"] == "Oracle"].set_index("group")
    od = pd.read_csv(P2 / "oracle_definitions.csv").set_index("group")
    rows = []
    for g in orc.index:
        f = fam(g)
        rows.append({"workload": f, "label": label(f), "N": int(orc.loc[g, "N"]), "models": ";".join(f.split("+")),
                     "S_allgpu": float(gpu.loc[g, "S_selected"]), "S_oracle": float(orc.loc[g, "S_selected"]),
                     "oracle_combination": od.loc[g, "oracle_macro(b)"], "oracle_placement": od.loc[g, "placement_macro(b)"],
                     "n_oracle": int(od.loc[g, "n_oracle_macro(b)"]), "allgpu_is_oracle": bool(abs(orc.loc[g, "S_selected"] - gpu.loc[g, "S_selected"]) <= 1e-9),
                     "diff": float(orc.loc[g, "S_selected"] - gpu.loc[g, "S_selected"])})
    d = pd.DataFrame(rows).sort_values(["N", "label"]).reset_index(drop=True)
    assert len(d) == 12 and list(d["N"]) == [3, 3, 3, 3, 3, 3, 4, 5, 6, 6, 7, 7], list(d["N"])
    d.round(6).to_csv(OUT / "fig1a_data.csv", index=False)
    PL._style()
    fig, ax = plt.subplots(figsize=(W, 2.3))
    x = np.arange(len(d))
    for i, r in d.iterrows():
        ax.plot([i, i], [r["S_allgpu"], r["S_oracle"]], color="0.6", linewidth=0.6, zorder=1)
    ax.scatter(x, d["S_allgpu"], marker="v", s=9, color="tab:orange", label="All-GPU", zorder=3, linewidths=0)
    ax.scatter(x, d["S_oracle"], marker="D", s=7, color="black", label="Oracle", zorder=4, linewidths=0)
    prev = None
    for i, n in enumerate(d["N"]):
        if prev is not None and n != prev:
            ax.axvline(i - 0.5, color="0.85", linewidth=0.4, zorder=0)
        prev = n
    for n in sorted(d["N"].unique()):
        idx = np.where(d["N"] == n)[0]
        ax.text(idx.mean(), 1.012, f"N={n}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], rotation=60, ha="right", fontsize=5)
    ax.set_ylabel("Measured score", fontsize=6)
    lo = min(d["S_allgpu"].min(), d["S_oracle"].min())
    ax.set_ylim(np.floor(lo * 20) / 20 - 0.01, 1.035)
    ax.set_xlim(-0.6, len(d) - 0.4)
    axes_style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=5.5, frameon=False, handletextpad=0.3)
    save(fig, "fig1a")
    return d


def fig1b():
    lab = pd.read_csv(DS / "labels_test.csv")
    inst = pd.read_csv(DS / "labels_instances_all.csv")
    f = "gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19"
    w = lab[lab["family"] == f].copy()
    assert len(w) == 128 and (w["n_reps"] == 3).all()
    w["S_agg"] = w["throughput_norm"] - ALPHA * w["drop_agg_norm"]
    w["S_macro"] = w["throughput_norm"] - ALPHA * w["drop_rate_norm"]
    pick = {"(i) aggregate-score optimum": w.loc[w["S_agg"].idxmax()], "(ii) macro-drop-score optimum": w.loc[w["S_macro"].idxmax()],
            "(iii) throughput optimum": w.loc[w["F_mean"].idxmax()]}
    for k, r in pick.items():
        tied = w[w["S_agg" if k.startswith("(i)") else "S_macro" if k.startswith("(ii)") else "F_mean"] >= (r["S_agg"] if k.startswith("(i)") else r["S_macro"] if k.startswith("(ii)") else r["F_mean"]) - 1e-9]
        assert len(tied) == 1, f"tie for {k}"
    od = pd.read_csv(P2 / "oracle_definitions.csv").set_index("group").loc[f + "_x3.yaml"]
    assert pick["(i) aggregate-score optimum"]["combination"] == od["oracle_agg(a)"] and pick["(ii) macro-drop-score optimum"]["combination"] == od["oracle_macro(b)"]
    ii = inst[(inst["family"] == f) & (inst["rate"] == "x3")]
    rows = []
    for k, r in pick.items():
        sub = ii[ii["combination"] == r["combination"]].sort_values("instance_index")
        base = {"placement_role": k, "combination": r["combination"], "placement(GC per sorted model)": r["placement"],
                "F_mean_req_s": r["F_mean"], "T": r["throughput_norm"], "D_agg": r["drop_agg_norm"], "D_macro": r["drop_rate_norm"],
                "S_agg": r["S_agg"], "S_macro": r["S_macro"], "Lambda": r["Lambda_sched"]}
        for _, s in sub.iterrows():
            rows.append({**base, "instance": s["model"], "device": s["device"].upper(), "lambda_i": s["lambda_i"], "generated_i": s["generated_i"],
                         "completed_i": s["completed_i"], "dropped_i": s["dropped_i"], "q_i": s["q_i_i"]})
    d = pd.DataFrame(rows)
    d.round(6).to_csv(OUT / "fig1b_data.csv", index=False)
    same_iii = pick["(iii) throughput optimum"]["combination"] == pick["(i) aggregate-score optimum"]["combination"]
    a, b = d[d["placement_role"].str.startswith("(i)")], d[d["placement_role"].str.startswith("(ii)")]
    facts = {"workload": f, "agg_optimum": pick["(i) aggregate-score optimum"]["combination"], "macro_optimum": pick["(ii) macro-drop-score optimum"]["combination"],
             "throughput_optimum": pick["(iii) throughput optimum"]["combination"], "throughput_optimum_equals_agg_optimum": bool(same_iii),
             "T_agg_opt": float(a["T"].iloc[0]), "T_macro_opt": float(b["T"].iloc[0]), "T_diff(agg-macro)": float(a["T"].iloc[0] - b["T"].iloc[0]),
             "max_q_in_macro_opt": float(b["q_i"].max()), "max_q_instance_in_macro_opt": b.loc[b["q_i"].idxmax(), "instance"],
             "gpt2_q_in_agg_opt": float(a.loc[a["instance"] == "gpt2", "q_i"].iloc[0]), "gpt2_device_in_agg_opt": a.loc[a["instance"] == "gpt2", "device"].iloc[0],
             "D_agg_agg_opt": float(a["D_agg"].iloc[0]), "D_agg_macro_opt": float(b["D_agg"].iloc[0]), "D_agg_diff(macro-agg)": float(b["D_agg"].iloc[0] - a["D_agg"].iloc[0]),
             "D_macro_agg_opt": float(a["D_macro"].iloc[0]), "D_macro_macro_opt": float(b["D_macro"].iloc[0]),
             "S_agg": {"agg_opt": float(a["S_agg"].iloc[0]), "macro_opt": float(b["S_agg"].iloc[0])},
             "S_macro": {"agg_opt": float(a["S_macro"].iloc[0]), "macro_opt": float(b["S_macro"].iloc[0])},
             "F": {"agg_opt": float(a["F_mean_req_s"].iloc[0]), "macro_opt": float(b["F_mean_req_s"].iloc[0])}}
    PL._style()
    fig, ax = plt.subplots(figsize=(W, 2.1))
    models = list(a["instance"])
    x = np.arange(len(models))
    wd = 0.38
    ax.bar(x - wd / 2, a["q_i"], wd, color="lightcoral", alpha=0.7, hatch="..", edgecolor="black", linewidth=0.3,
           label=r"max $T - 0.3\,D_{\mathrm{agg}}$ (aggregate drop)")
    ax.bar(x + wd / 2, b["q_i"], wd, color="skyblue", alpha=0.7, hatch="//", edgecolor="black", linewidth=0.3,
           label=r"max $T - 0.3\,D_{\mathrm{macro}}$ (per-instance drop)")
    short = {"gpt2": "GPT-2", "mnasnet": "Mnas", "resnet50": "R50", "resnext50": "RX50", "shufflenet-v2-12": "Shuffle", "squeezenet1.0-12": "Squeeze", "vgg19": "VGG19"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{short[m]}\n{a['device'].iloc[i][0]} | {b['device'].iloc[i][0]}" for i, m in enumerate(models)], fontsize=5)
    ax.set_xlabel("instance (device: aggregate optimum | per-instance optimum)", fontsize=5.5)
    ax.set_ylabel(r"Drop fraction $q_i$", fontsize=6)
    ax.set_ylim(0, 1)
    axes_style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=5, frameon=False, handletextpad=0.3)
    save(fig, "fig1b")
    return d, facts


def fig3():
    d = pd.read_csv(P2 / "fig3_data.csv")
    d = d[d["N>="].between(3, 7)].sort_values("N>=")
    assert np.allclose(d["All-GPU"], d["Latency-based"]), "All-GPU and Latency-based differ: they cannot share one series"
    out = d[["N>=", "workloads", "Oracle", "Ours", "All-GPU", "Latency-based", "Best-of-5", "Best-of-5_std"]]
    out.to_csv(OUT / "fig3_data.csv", index=False)
    PL._style()
    fig, ax = plt.subplots(figsize=(W, 2.3))
    x = d["N>="].values
    ax.plot(x, d["Oracle"], marker="D", label="Oracle", color="black", linewidth=0.5, markersize=2, linestyle="--")
    ax.plot(x, d["Ours"], marker="o", label="Ours", color="tab:red", linewidth=0.5, markersize=2)
    ax.plot(x, d["All-GPU"], marker="v", label="All-GPU (= Latency-based)", color="tab:orange", linewidth=0.5, markersize=2, linestyle=":")
    ax.plot(x, d["Best-of-5"], marker="^", label="Best-of-5", color="forestgreen", linewidth=0.5, markersize=2)
    ax.fill_between(x, d["Best-of-5"] - d["Best-of-5_std"], d["Best-of-5"] + d["Best-of-5_std"], color="forestgreen", alpha=0.15, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥{n} ({w})" for n, w in zip(d["N>="], d["workloads"])], fontsize=6)
    ax.set_xlabel("Number of application instances (workloads)", fontsize=6)
    ax.set_ylabel("Average score", fontsize=6)
    axes_style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, fontsize=5, frameon=False, columnspacing=0.8, handletextpad=0.3)
    save(fig, "fig3")
    return out


def fig4():
    d = pd.read_csv(REPO / "revision_exp" / "out_phase3" / "fig4_noclip" / "fig4_data.csv").sort_values("alpha")
    out = d[["alpha", "regret_multi", "regret_single", "top5_hits_multi", "top5_hits_single", "top1_hits_multi", "top1_hits_single",
             "avg_score_multi", "avg_score_single", "oracle_avg", "workloads"]]
    out.to_csv(OUT / "fig4_data.csv", index=False)
    PL._style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.9))
    al = d["alpha"].values
    a1.plot(al, d["regret_multi"], marker="o", markersize=2, linewidth=0.6, color="tab:blue", label="Multi-objective")
    a1.plot(al, d["regret_single"], marker="s", markersize=2, linewidth=0.6, color="tab:red", linestyle="--", label="Single-objective")
    a1.set_ylim(0, max(d["regret_multi"].max(), d["regret_single"].max()) * 1.15)
    a1.set_xlabel(r"$\alpha$", fontsize=6)
    a1.set_ylabel("Regret", fontsize=6)
    a1.set_xticks(al)
    a1.set_xticklabels([f"{v:.1f}" if i % 3 == 0 else "" for i, v in enumerate(al)], fontsize=5.5)
    x = np.arange(len(al))
    wd = 0.38
    a2.bar(x - wd / 2, d["top5_hits_multi"], wd, color="skyblue", alpha=0.6, hatch="//", edgecolor="black", linewidth=0.3, label="Multi-objective")
    a2.bar(x + wd / 2, d["top5_hits_single"], wd, color="lightcoral", alpha=0.6, hatch="..", edgecolor="black", linewidth=0.3, label="Single-objective")
    a2.set_ylim(0, 12)
    a2.set_yticks([0, 3, 6, 9, 12])
    a2.set_xticks(x)
    a2.set_xticklabels([f"{v:.1f}" if i % 3 == 0 else "" for i, v in enumerate(al)], fontsize=5.5)
    a2.set_xlabel(r"$\alpha$", fontsize=6)
    a2.set_ylabel("Top-5 hits (of 12)", fontsize=6)
    for a in (a1, a2):
        axes_style(a)
    a2.grid(axis="x", visible=False)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=5.5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig4")
    le = int((d["regret_multi"] <= d["regret_single"] + 1e-12).sum())
    facts = {"alphas": len(d), "multi_regret_le_single": le, "max_abs_regret_diff": float((d["regret_multi"] - d["regret_single"]).abs().max()),
             "alpha_of_max_diff": float(d.loc[(d["regret_multi"] - d["regret_single"]).abs().idxmax(), "alpha"]),
             "top1_multi_range": [int(d["top1_hits_multi"].min()), int(d["top1_hits_multi"].max())],
             "top1_single_range": [int(d["top1_hits_single"].min()), int(d["top1_hits_single"].max())],
             "top5_multi_range": [int(d["top5_hits_multi"].min()), int(d["top5_hits_multi"].max())],
             "top5_single_range": [int(d["top5_hits_single"].min()), int(d["top5_hits_single"].max())],
             "alphas_multi_regret_gt_single": [float(a) for a, m, s in zip(d["alpha"], d["regret_multi"], d["regret_single"]) if m > s + 1e-12]}
    return out, facts


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a = fig1a()
    _, fb = fig1b()
    fig3()
    _, f4 = fig4()
    (OUT / "figure_facts.json").write_text(json.dumps({"fig1b": fb, "fig4": f4}, indent=1, default=str))
    print(a.to_string())
    print(json.dumps({"fig1b": fb, "fig4": f4}, indent=1, default=str))
