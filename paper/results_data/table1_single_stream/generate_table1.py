#!/usr/bin/env python3
"""Table 1 (tab:single-stream) generator — per-size single-camera sAP, threads=4.

Source : single_stream.csv (from accv_experiments/results/rev19_table1_threads4.csv;
         threads=4 both devices, mxq b2441f9d, global8, yolo11s, 24 logs, 3 reps, frame skip ~0).
Output : table1.tex (tabular body, identical to paper tab:single-stream).
Cross-validates per-size sAP / diff / rel% against paper/main_vision.tex.
Extraction only. p-values (<0.001 / n.s.) carried as in paper (significance of the paired test).
"""
import sys, json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd

HERE = Path(__file__).resolve().parent
TEX_PAPER = HERE.parent.parent / "main_vision.tex"
PVAL = {"small": "$p<0.001$", "medium": "$p<0.001$", "large": "n.s."}

def r3(x):  # round half up to 3 decimals (paper convention)
    return float(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def main():
    df = pd.read_csv(HERE / "single_stream.csv").set_index("size")
    meta = json.loads((HERE / "meta.json").read_text())
    def row(size, label):
        r = df.loc[size]
        dv = r3(float(r["diff"])); rel = float(r["rel_pct"]); gpu=r3(float(r["gpu"])); npu=r3(float(r["npu"]))
        diff = f"${'+' if dv>=0 else '-'}{abs(dv):.3f}$"
        relstr = f"({'+' if rel>=0 else '$-$'}{abs(rel):.1f}\\%, {PVAL[size]})"
        return f"AP$_{{\\text{{{label}}}}}$ & {gpu:.3f} & {npu:.3f} & {diff} {relstr} \\\\"
    tex = [r"\begin{tabular}{lccc}", r"\toprule",
           r"Metric & GPU & NPU & NPU$-$GPU \\", r"\midrule",
           f"infer mean (ms)      & 8.5 & {meta['npu_infer_ms']:.1f} & --- \\\\",
           r"frame skip (\%)      & $\sim\!0$ & $\sim\!0$ & --- \\",
           row("small","small"), row("medium","medium"), row("large","large"),
           r"\bottomrule", r"\end{tabular}"]
    (HERE / "table1.tex").write_text("\n".join(tex) + "\n")

    paper = TEX_PAPER.read_text(); ok = True; miss = []
    for size in ["small","medium","large"]:
        r = df.loc[size]
        for tok in [f"{r3(float(r['gpu'])):.3f}", f"{r3(float(r['npu'])):.3f}", f"{abs(r3(float(r['diff']))):.3f}", f"{abs(float(r['rel_pct'])):.1f}\\%"]:
            if tok not in paper: ok=False; miss.append((size,tok))
    print("Table1 per-size:", {s: (df.loc[s].gpu, df.loc[s].npu, df.loc[s].diff, df.loc[s].rel_pct) for s in df.index})
    print(f"  note: infer GPU measured {meta['gpu_infer_ms']} ms (paper rounds 8.55->8.5 from rev18); per-size sAP exact.")
    print(f"Table1 cross-check vs main_vision.tex: {'PASS' if ok else 'FAIL'}" + ("" if ok else f"  missing {miss}"))
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
