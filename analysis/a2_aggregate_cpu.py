"""A2 Task 1 aggregation + figure (mirror image of Fig 3).

Reads analysis/a2_cpu_sweep.csv, aggregates mean+/-std over reps per
(point, placement), and draws the CPU-localized contention sweep in the paper's
Fig 3 style: All-GPU vs All-NPU worst-stream sAP, x = NPU deadline-miss (%),
with host CPU util annotated; the realistic anchor (TinyLLaMA CPU) as a separate
marker. Prints the crossover coordinate if the curves cross.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "analysis/a2_cpu_sweep.csv"
PDF = ROOT / "analysis/a2_cpu_sweep.pdf"
OUT_AGG = ROOT / "analysis/a2_cpu_sweep_agg.csv"


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def agg(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["point"], r["cotenant"], int(r["c"]), r["placement"])].append(r)
    out = []
    for (point, cot, c, pl), rs in groups.items():
        def col(name):
            return np.array([float(x[name]) for x in rs if x.get(name) not in (None, "")])
        def ms(name):
            a = col(name)
            return (float(a.mean()), float(a.std(ddof=0))) if len(a) else (float("nan"), 0.0)
        w_m, w_s = ms("worst_sap")
        out.append({
            "point": point, "cotenant": cot, "c": c, "placement": pl, "n": len(rs),
            "worst_sap_mean": round(w_m, 5), "worst_sap_std": round(w_s, 5),
            "mean_sap_mean": round(ms("mean_sap")[0], 5),
            "sap_small": round(ms("sap_small")[0], 5),
            "sap_medium": round(ms("sap_medium")[0], 5),
            "sap_large": round(ms("sap_large")[0], 5),
            "gpu_skip_mean": round(ms("gpu_skip_pct")[0], 2),
            "npu_skip_mean": round(ms("npu_skip_pct")[0], 2),
            "cpu_pct_mean": round(ms("cpu_pct_mean")[0], 1),
            "cpu_nbusy_mean": round(ms("cpu_nbusy_mean")[0], 1),
            "gpu_util_mean": round(ms("gpu_util_mean")[0], 1),
        })
    out.sort(key=lambda r: (r["placement"], r["c"]))
    return out


def main():
    rows = load()
    a = agg(rows)
    with open(OUT_AGG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(a[0].keys()))
        w.writeheader()
        for r in a:
            w.writerow(r)
    print(f"wrote {OUT_AGG}\n")

    # console table
    print(f"{'point':14s} {'place':7s} {'c':>3s} {'cpu%':>5s} {'nbusy':>5s} "
          f"{'npuDM%':>7s} {'gpuDM%':>7s} {'worst':>7s} {'small':>6s} {'large':>6s}")
    for r in a:
        print(f"{r['point']:14s} {r['placement']:7s} {r['c']:3d} "
              f"{r['cpu_pct_mean']:5.1f} {r['cpu_nbusy_mean']:5.1f} "
              f"{r['npu_skip_mean']:7.2f} {r['gpu_skip_mean']:7.2f} "
              f"{r['worst_sap_mean']:7.4f} {r['sap_small']:6.3f} {r['sap_large']:6.3f}")

    sweep = [r for r in a if r["cotenant"] == "cpu_stress"]
    gpu = {r["c"]: r for r in sweep if r["placement"] == "AllGPU"}
    npu = {r["c"]: r for r in sweep if r["placement"] == "AllNPU"}
    cs = sorted(gpu)

    # monotonicity + crossover
    npu_worst = [npu[c]["worst_sap_mean"] for c in cs]
    gpu_worst = [gpu[c]["worst_sap_mean"] for c in cs]
    mono = all(npu_worst[i] >= npu_worst[i + 1] - 1e-4 for i in range(len(npu_worst) - 1))
    print(f"\nAll-NPU worst vs c: {[round(x,4) for x in npu_worst]}  monotonic_decreasing={mono}")
    print(f"All-GPU worst vs c: {[round(x,4) for x in gpu_worst]}")
    cross = None
    for i in range(len(cs) - 1):
        d0 = gpu_worst[i] - npu_worst[i]
        d1 = gpu_worst[i + 1] - npu_worst[i + 1]
        if d0 * d1 < 0:
            cross = (cs[i], cs[i + 1])
    if cross:
        c0, c1 = cross
        print(f"CROSSOVER between c={c0} and c={c1} "
              f"(NPU DM {npu[c0]['npu_skip_mean']:.1f}%->{npu[c1]['npu_skip_mean']:.1f}%, "
              f"CPU {npu[c0]['cpu_pct_mean']:.0f}%->{npu[c1]['cpu_pct_mean']:.0f}%)")
    else:
        print("No AllGPU/AllNPU crossover in swept range.")

    # ---- figure: mirror of Fig 3 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    # common contention coordinate = host CPU util %
    xg = [gpu[c]["cpu_pct_mean"] for c in cs]
    xnpu = [npu[c]["cpu_pct_mean"] for c in cs]
    ax.errorbar(xnpu, [npu[c]["worst_sap_mean"] for c in cs],
                yerr=[npu[c]["worst_sap_std"] for c in cs],
                marker="o", color="#EE6677", lw=1.8, capsize=3, label="All-NPU")
    ax.errorbar(xg, [gpu[c]["worst_sap_mean"] for c in cs],
                yerr=[gpu[c]["worst_sap_std"] for c in cs],
                marker="s", color="#4477AA", lw=1.8, capsize=3, label="All-GPU")
    for c in cs:
        ax.annotate(f"c={c}", (npu[c]["cpu_pct_mean"], npu[c]["worst_sap_mean"]),
                    textcoords="offset points", xytext=(3, -10), fontsize=6, color="#555555")
    an_n = [r for r in a if r["cotenant"] == "tinyllama_cpu" and r["placement"] == "AllNPU"]
    an_g = [r for r in a if r["cotenant"] == "tinyllama_cpu" and r["placement"] == "AllGPU"]
    if an_n:
        ax.scatter([an_n[0]["cpu_pct_mean"]], [an_n[0]["worst_sap_mean"]],
                   marker="*", s=180, color="#EE6677", edgecolor="black", zorder=5,
                   label="anchor: LLM-CPU (NPU)")
    if an_g:
        ax.scatter([an_g[0]["cpu_pct_mean"]], [an_g[0]["worst_sap_mean"]],
                   marker="*", s=180, color="#4477AA", edgecolor="black", zorder=5,
                   label="anchor: LLM-CPU (GPU)")
    ax.set_xlabel("host CPU utilization (%)")
    ax.set_ylabel("worst-stream sAP")
    ax.set_title("(a) worst sAP vs host-CPU contention")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    # panel (b): deadline-miss rate vs CPU% for each path
    ax2.plot(xnpu, [npu[c]["npu_skip_mean"] for c in cs], "o-", color="#EE6677",
             lw=1.8, label="All-NPU DM")
    ax2.plot(xg, [gpu[c]["gpu_skip_mean"] for c in cs], "s-", color="#4477AA",
             lw=1.8, label="All-GPU DM")
    ax2.set_xlabel("host CPU utilization (%)")
    ax2.set_ylabel("deadline-miss rate (%)")
    ax2.set_title("(b) deadline-miss vs host-CPU contention")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PDF)
    print(f"\nwrote {PDF}")


if __name__ == "__main__":
    main()
