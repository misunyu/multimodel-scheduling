"""Fig. 3 v2: same data and style as fig3 (code/figures.py fig3), height x0.8 (2.3 -> 1.84 in), width unchanged. Writes fig3_v2.{pdf,png} only.

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp/out_manuscript/code/fig3_v2.py
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import figures as FG  # noqa: E402  (helpers and style only; its __main__ is not run)

for ext in ["pdf", "png"]:
    assert not (FG.OUT / f"fig3_v2.{ext}").exists(), f"fig3_v2.{ext} exists; refusing to overwrite"
d = pd.read_csv(FG.OUT / "fig3_data.csv").sort_values("N>=")  # the data plotted in fig3.pdf
FG.PL._style()
fig, ax = plt.subplots(figsize=(FG.W, 2.3 * 0.8))
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
FG.axes_style(ax)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=5, frameon=False, columnspacing=0.8, handletextpad=0.3,
          borderaxespad=0.2)  # anchored to the axes top so the shorter axes keep the legend clear of the plot
FG.save(fig, "fig3_v2")
