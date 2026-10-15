"""Family reconstruction after an LM decision (spec v2.2 section 3.4) and Phase 2 run counts.
  python3 -B code/family_reconstruct.py   -> family_reconstruction.csv, family_scenarios.csv
Rule: drop the excluded models' instances; N = 0 -> removed; equal to a family that is already kept -> removed (duplicate);
otherwise kept. Unchanged original families are registered first, so a reduced family that equals one of them is the duplicate.
Runs: all 2^N placements x 4 rates (1,2,3,4x) + 2 extra repeats of the 3x runs for N >= 3 (D6: 3 measurements in total).
"""
import sys

sys.dont_write_bytecode = True
import csv
from pathlib import Path

from gen_schedules import existing_families

ROOT = Path(__file__).resolve().parent.parent
SCEN = {"both_included": [], "exclude_tinyllama": ["tiny-llama-chat-onnx"], "exclude_gpt2": ["gpt2"],
        "exclude_both": ["gpt2", "tiny-llama-chat-onnx"]}
fam = existing_families()
rows, summ = [], []
for sc, excl in SCEN.items():
    kept = {}
    status = {}
    for f, models in sorted(fam.items()):
        if not set(models) & set(excl):
            kept[tuple(models)] = f
            status[f] = ("kept_unchanged", models, "")
    for f, models in sorted(fam.items(), key=lambda kv: (len(kv[1]), kv[0])):
        if f in status:
            continue
        red = tuple(m for m in models if m not in excl)
        if not red:
            status[f] = ("removed_empty", [], "")
        elif red in kept:
            status[f] = ("removed_duplicate", list(red), kept[red])
        else:
            kept[red] = f
            status[f] = ("kept_modified", list(red), "")
    ndist = {}
    cand = runs = rep_runs = 0
    for f in sorted(fam):
        st, red, dup = status[f]
        n = len(red)
        k = st.startswith("kept")
        if k:
            ndist[n] = ndist.get(n, 0) + 1
            cand += 2 ** n
            runs += 4 * 2 ** n
            rep_runs += 2 * 2 ** n if n >= 3 else 0
        rows.append({"scenario": sc, "original_family": f, "original_n": len(fam[f]), "status": st, "new_models": "+".join(red), "new_n": n,
                     "duplicate_of": dup, "placements": 2 ** n if k else 0})
    summ.append({"scenario": sc, "excluded": "+".join(excl) or "-", "families": sum(ndist.values()),
                 **{f"N={n}": ndist.get(n, 0) for n in range(1, 9)}, "families_N>=3": sum(v for n, v in ndist.items() if n >= 3),
                 "placements_sum_2^N": cand, "runs_4_rates": runs, "extra_repeat_runs_3x_N>=3": rep_runs, "runs_total": runs + rep_runs})
for name, data in (("family_reconstruction.csv", rows), ("family_scenarios.csv", summ)):
    with open(ROOT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
for s in summ:
    print(s)
