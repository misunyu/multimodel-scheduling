"""Commented appendix artifacts:
  tab:gpu-util   -> appendix_gpu_util/     (rev30_canonical per-k All-GPU util/mem/power)
  fig:util-vs-dm -> appendix_util_vs_dm/   (analysis/a3_util_vs_dm.pdf + per-k util vs DM)
Both are commented-out in main_vision.tex; packaged for supplementary completeness."""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RES, RD, ANALYSIS, backup_stale

R30 = RES / "rev30_clean_resnet"
KS = [0, 1, 2, 3, 4, 6, 8]


def gpu_util():
    PKG = RD / "appendix_gpu_util"; PKG.mkdir(exist_ok=True)
    c = pd.read_csv(R30 / "rev30_canonical.csv")
    c["ratio"] = c.placement.apply(lambda s: s.count("N"))
    g = c[(c.group == "N4") & (c.ratio == 0)]
    agg = g.groupby("k").agg(gpu_dm=("gpu_skip_pct", "mean"), util_mean=("util_mean", "mean"),
                             util_p95=("util_p95", "mean"), mem=("mem_used_mib", "mean"),
                             power=("power_w", "mean")).round(1).reindex(KS)
    agg.to_csv(PKG / "gpu_util_by_k.csv")
    tex = [r"% tab:gpu-util (commented in paper). regen_appendix.py",
           r"\begin{tabular}{lccccc}", r"\toprule",
           r"$k$ & GPU DM & util mean & util p95 & mem (MiB) & power (W) \\", r"\midrule"]
    for k in KS:
        r = agg.loc[k]
        tex.append(f"{k} & ${r.gpu_dm:.1f}\\%$ & ${r.util_mean:.0f}\\%$ & ${r.util_p95:.0f}\\%$ & "
                   f"${int(r.mem)}$ & ${int(r.power)}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (PKG / "gpu_util.tex").write_text("\n".join(tex) + "\n")
    print("appendix_gpu_util: util_mean by k =", list(agg.util_mean.values))


def util_vs_dm():
    PKG = RD / "appendix_util_vs_dm"; PKG.mkdir(exist_ok=True)
    pdf = ANALYSIS / "a3_util_vs_dm.pdf"
    if not pdf.exists():
        subprocess.run([sys.executable, str(ANALYSIS / "a3_evaluate.py")], check=True, cwd=ANALYSIS)
    backup_stale(PKG / "a3_util_vs_dm.pdf")
    shutil.copy(pdf, PKG / "a3_util_vs_dm.pdf")
    (PKG / "README.txt").write_text(
        "fig:util-vs-dm (commented, main_vision.tex L725). Source analysis/a3_evaluate.py.\n"
        "Left axis GPU util (flat ~68%), right axis GPU deadline-miss (24->67%).\n")
    print("appendix_util_vs_dm: copied a3_util_vs_dm.pdf")


def main():
    gpu_util()
    util_vs_dm()
    return True


if __name__ == "__main__":
    main()
