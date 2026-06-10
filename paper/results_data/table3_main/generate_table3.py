#!/usr/bin/env python3
"""Table 3 (tab:main) generator — worst/mean sAP, 3 strategies x 3 co-tenants, N=4.

Source : main_worst_mean.csv (from accv_experiments/results/rev21_mean_worst_extract.csv N=4
         + rev20_5strat_heavybg.csv device skips; threads=4, mxq b2441f9d, global8, yolo11s, 3 reps).
Output : table3.tex (tabular body, identical to paper tab:main).
Bold: L2 All-GPU worst, L3 All-NPU worst (matches paper).
Extraction only.
"""
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
TEX_PAPER = HERE.parent.parent / "main_vision.tex"
SUB = {"L1_CNN": r"L1$_{\text{CNN}}$ (ResNet)  ", "L2_LM": r"L2$_{\text{LM}}$ ($+$LLM)  ",
       "L3_VLM": r"L3$_{\text{VLM}}$ ($+$VLM) "}
BOLD = {("L2_LM","All-GPU_worst"), ("L3_VLM","All-NPU_worst")}

def cell(ct, key, v):
    s=f"{v:.3f}"
    return (r"$\mathbf{"+s+"}$") if (ct,key) in BOLD else f"${s}$"

def main():
    df=pd.read_csv(HERE/"main_worst_mean.csv")
    lines=[r"\begin{tabular}{lcc|cc|cc|c}", r"\toprule",
           r" & \multicolumn{2}{c|}{All-GPU} & \multicolumn{2}{c|}{All-NPU} & \multicolumn{2}{c|}{Oracle} & GPU/NPU \\",
           r"co-tenant & worst & mean & worst & mean & worst & mean & skip \\", r"\midrule"]
    for _,r in df.iterrows():
        ct=r.co_tenant
        cells=[cell(ct,"All-GPU_worst",r["All-GPU_worst"]),f"${r['All-GPU_mean']:.3f}$",
               cell(ct,"All-NPU_worst",r["All-NPU_worst"]),f"${r['All-NPU_mean']:.3f}$",
               cell(ct,"Oracle_worst",r["Oracle_worst"]),f"${r['Oracle_mean']:.3f}$",
               f"${int(r.gpu_skip)}\\% / {int(r.npu_skip)}\\%$"]
        lines.append(f"{SUB[ct]} & "+" & ".join(cells)+r" \\")
    lines+=[r"\bottomrule", r"\end{tabular}"]
    (HERE/"table3.tex").write_text("\n".join(lines)+"\n")

    paper=TEX_PAPER.read_text(); ok=True; miss=[]
    for _,r in df.iterrows():
        for k in ["All-GPU_worst","All-GPU_mean","All-NPU_worst","All-NPU_mean","Oracle_worst","Oracle_mean"]:
            if f"{r[k]:.3f}" not in paper: ok=False; miss.append((r.co_tenant,k,f"{r[k]:.3f}"))
    print("Table3 rows:")
    for _,r in df.iterrows(): print(f"  {r.co_tenant}: AllGPU {r['All-GPU_worst']}/{r['All-GPU_mean']}  AllNPU {r['All-NPU_worst']}/{r['All-NPU_mean']}  Oracle {r['Oracle_worst']}/{r['Oracle_mean']}  skip {int(r.gpu_skip)}/{int(r.npu_skip)}")
    print(f"Table3 cross-check vs main_vision.tex: {'PASS' if ok else 'FAIL'}" + ("" if ok else f"  missing {miss}"))
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
