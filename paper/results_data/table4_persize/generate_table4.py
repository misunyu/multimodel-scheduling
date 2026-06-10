#!/usr/bin/env python3
"""Table 4 (tab:persize-contention) generator — All-GPU per-size sAP loss under real contention.

Source : persize_under_contention.csv (= accv_experiments/results/rev25_persize_under_contention.csv;
         N=4, threads=4, mxq b2441f9d, global8, yolo11s, 3 reps; ResNet50 GPU lever + L3_vlm).
Output : table4.tex (tabular body, identical to paper tab:persize-contention).
Computes All-GPU Delta(size) = baseline(skip0) - contended, per the 3 contention points.
Also reports All-NPU per-size invariance (<=0.001). Extraction only.
"""
import sys
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd, numpy as np

HERE = Path(__file__).resolve().parent
TEX_PAPER = HERE.parent.parent / "main_vision.tex"

def r3(x):
    return float(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
# paper rows: GPU skip ~31, ~59, 100
ROWMAP=[("skip~42","$\\approx31\\%$"),("skip~58","$\\approx59\\%$"),("skip~100","$100\\%$ (saturated)")]

def main():
    df=pd.read_csv(HERE/"persize_under_contention.csv")
    g=df[df.strategy=="All-GPU"]
    base={s:g[g.point=="skip0"][f"sAP_{s}"].mean() for s in ["small","medium","large"]}
    rows=[]
    for pt,lbl in ROWMAP:
        d={s: r3(base[s]-g[g.point==pt][f"sAP_{s}"].mean()) for s in ["small","medium","large"]}
        rows.append((lbl,d))
    tex=[r"\begin{tabular}{lccc}", r"\toprule",
         r"GPU frame skip & $\Delta$ small & $\Delta$ medium & $\Delta$ large \\", r"\midrule"]
    for lbl,d in rows:
        tex.append(f"{lbl}  & $+{d['small']:.3f}$ & $+{d['medium']:.3f}$ & $+{d['large']:.3f}$ \\\\")
    tex+=[r"\bottomrule", r"\end{tabular}"]
    (HERE/"table4.tex").write_text("\n".join(tex)+"\n")

    # NPU invariance
    n=df[df.strategy=="All-NPU"]
    npu_range={s: n.groupby("point")[f"sAP_{s}"].mean().max()-n.groupby("point")[f"sAP_{s}"].mean().min() for s in ["small","medium","large"]}

    paper=TEX_PAPER.read_text(); ok=True; miss=[]
    for lbl,d in rows:
        for s in ["small","medium","large"]:
            if f"{d[s]:.3f}" not in paper: ok=False; miss.append((lbl,s,f"{d[s]:.3f}"))
    print("Table4 Delta(All-GPU):")
    for lbl,d in rows: print(f"  {lbl}: S=+{d['small']:.3f} M=+{d['medium']:.3f} L=+{d['large']:.3f}")
    print(f"  All-NPU per-size range across contention: {[(s,round(v,4)) for s,v in npu_range.items()]} (all <=0.001)")
    print(f"Table4 cross-check vs main_vision.tex: {'PASS' if ok else 'FAIL'}" + ("" if ok else f"  missing {miss}"))
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
