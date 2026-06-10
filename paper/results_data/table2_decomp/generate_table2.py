#!/usr/bin/env python3
"""Table 2 (tab:decomp) generator — causal separation quantization vs staleness.

Source : decomp.csv (quant rel% = threads=4 NPU-GPU per-size gap; staleness dAP =
         threads24 gap - threads4 gap, from accv_experiments/results/rev18_postproc_levers.csv)
         bitident_summary.json (rev22_bitident.csv: 20 frames, diff=0).
Output : table2.tex (tabular body, identical to paper tab:decomp).
NOTE: offline mAP is NOT thread-invariant (it penalizes skip as zero recall); causal
      separation rests on bit-identical per-processed-frame detections + skip 0 vs 52% contrast.
Extraction only.
"""
import sys, json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
TEX_PAPER = HERE.parent.parent / "main_vision.tex"

def main():
    df = pd.read_csv(HERE / "decomp.csv").set_index("size")
    bi = json.loads((HERE / "bitident_summary.json").read_text())
    def q(s):
        v=df.loc[s,"quant_rel_pct"]; return f"${'+' if v>=0 else '-'}{abs(v):.1f}\\%$"
    def st(s):
        v=df.loc[s,"staleness_dAP"]
        return r"$\approx 0$" if abs(v)<0.005 else f"${'+' if v>=0 else '-'}{abs(v):.3f}$"
    tex=[r"\begin{tabular}{lccc}", r"\toprule",
         r"loss component & small & medium & large \\", r"\midrule",
         f"quantization (staleness off) & {q('small')} & {q('medium')} & {q('large')} \\\\",
         f"staleness (off $\\to$ on, $\\Delta$sAP) & {st('small')} & {st('medium')} & {st('large')} \\\\",
         r"\bottomrule", r"\end{tabular}"]
    (HERE/"table2.tex").write_text("\n".join(tex)+"\n")

    paper=TEX_PAPER.read_text(); ok=True; miss=[]
    for s in ["small","medium","large"]:
        for tok in [f"{abs(df.loc[s,'quant_rel_pct']):.1f}\\%"]:
            if tok not in paper: ok=False; miss.append(tok)
    for v in [0.030,0.098]:
        if f"{v:.3f}" not in paper: ok=False; miss.append(f"{v:.3f}")
    bi_ok = bi["all_identical"] and bi["max_box_diff"]==0.0 and bi["max_score_diff"]==0.0
    print("Table2 quant rel%:", dict(df.quant_rel_pct), " staleness dAP:", dict(df.staleness_dAP))
    print(f"  bit-identical: {bi['n_frames']} frames, all_identical={bi['all_identical']}, max diff={bi['max_box_diff']}")
    print(f"Table2 cross-check vs main_vision.tex: {'PASS' if (ok and bi_ok) else 'FAIL'}" + ("" if ok else f"  missing {miss}"))
    return ok and bi_ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
