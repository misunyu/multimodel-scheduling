"""Phase 2 section 1 (section 5.4 repeat variation): beta 0.1, 2x, LM-free N=6, 8 candidates, 3 repeat blocks.

  python3 -B code/analyze_repeat54.py   -> pilot/repeat_5_4/repeat_5_4_{candidates,pairs,top1}.csv, summary.json

Definitions (fixed before the runs were looked at, Phase 2 v1.2 section 1):
  F_max, R_max  = max over the 8 pilot candidates of the 3-repeat mean of F and of the aggregate drop rate R
  S_rep (b)     = F_rep / F_max - alpha * D_macro_rep          (alpha = 0.3)
  S_rep (a)     = F_rep / F_max - alpha * R_rep / R_max ;  S_rep (c) = F_rep / F_max - alpha * D_max_rep
  SE            = std(ddof=1) / sqrt(3)
  independent   SE_diff = sqrt(SE_A^2 + SE_B^2);  paired SE_diff = SE of the per-repeat differences (same repeat index)
  rank-uncertain pair (decision rule) : |dS| < sqrt(SE_A^2 + SE_B^2).  The older SE_A + SE_B criterion is listed for reference.
"best measured among pilot candidates" is used instead of "oracle": only 8 of the 64 placements were measured.
"""
import sys

sys.dont_write_bytecode = True

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_pilot import load_runs, write_csv

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "pilot" / "repeat_5_4"
ALPHA = 0.3
THRESHOLD_PAIRS = 7   # of 28 (25 %)


def se(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def main():
    runs, invalid = load_runs(DIR / "runs")
    by = defaultdict(dict)
    for r in runs:
        by[r["placement"]][r["repeat"]] = r
    cands = sorted(by)
    reps = sorted({k for v in by.values() for k in v})
    assert len(cands) == 8 and reps == [0, 1, 2] and all(len(by[c]) == 3 for c in cands), (len(cands), reps, len(invalid))
    F = {c: np.array([by[c][k]["system"]["throughput_fps"] for k in reps]) for c in cands}
    R = {c: np.array([by[c][k]["system"]["drop_rate_fps"] for k in reps]) for c in cands}
    Dm = {c: np.array([by[c][k]["system"]["drop_macro"] for k in reps]) for c in cands}
    Dx = {c: np.array([by[c][k]["system"]["drop_max"] for k in reps]) for c in cands}
    fmax = max(F[c].mean() for c in cands)
    rmax = max(R[c].mean() for c in cands)
    S = {"b": {c: F[c] / fmax - ALPHA * Dm[c] for c in cands},
         "a": {c: F[c] / fmax - ALPHA * (R[c] / rmax if rmax > 0 else 0 * R[c]) for c in cands},
         "c": {c: F[c] / fmax - ALPHA * Dx[c] for c in cands}}
    best = {d: max(cands, key=lambda c: (S[d][c].mean(), c)) for d in S}
    Sb = S["b"]

    rows = []
    for c in cands:
        rows.append({"placement": c, "combination": by[c][0]["combination"], "n_gpu": c.count("G"),
                     "F_mean": round(F[c].mean(), 3), "F_se": round(se(F[c]), 3), "F_rel_range": round((F[c].max() - F[c].min()) / F[c].mean(), 5),
                     "D_macro_mean": round(Dm[c].mean(), 5), "D_macro_se": round(se(Dm[c]), 5),
                     "S_mean": round(Sb[c].mean(), 5), "S_se": round(se(Sb[c]), 5),
                     **{f"S_rep{k}": round(float(Sb[c][i]), 5) for i, k in enumerate(reps)},
                     "S_agg_mean(a)": round(S["a"][c].mean(), 5), "S_max_mean(c)": round(S["c"][c].mean(), 5),
                     "D_agg_mean": round(float((R[c] / rmax).mean()) if rmax > 0 else 0.0, 5), "D_max_mean": round(Dx[c].mean(), 5),
                     "best_measured(b)": c == best["b"], "best_measured(a)": c == best["a"], "best_measured(c)": c == best["c"]})
    rows.sort(key=lambda r: -r["S_mean"])
    write_csv(DIR / "repeat_5_4_candidates.csv", rows)

    pairs = []
    for a, b in itertools.combinations(cands, 2):
        d = Sb[a].mean() - Sb[b].mean()
        se_ind = float(np.hypot(se(Sb[a]), se(Sb[b])))
        se_pair = se(Sb[a] - Sb[b])
        pairs.append({"A": a, "B": b, "dS": round(d, 5), "abs_dS": round(abs(d), 5), "SE_A": round(se(Sb[a]), 5), "SE_B": round(se(Sb[b]), 5),
                      "SE_independent": round(se_ind, 5), "SE_paired": round(se_pair, 5),
                      "uncertain_independent": bool(abs(d) < se_ind), "uncertain_paired": bool(abs(d) < se_pair),
                      "uncertain_sum_rule(reference)": bool(abs(d) <= se(Sb[a]) + se(Sb[b])),
                      "involves_best_measured": best["b"] in (a, b)})
    pairs.sort(key=lambda r: r["abs_dS"])
    write_csv(DIR / "repeat_5_4_pairs.csv", pairs)

    top = []
    for i, k in enumerate(reps):
        row = {"rep": k}
        for d in ("b", "a", "c"):
            # per repeat, with that repeat's own normalisation (what a single-run campaign would see)
            f = {c: F[c][i] for c in cands}
            fm = max(f.values())
            rm = max(R[c][i] for c in cands)
            pen = {"b": {c: Dm[c][i] for c in cands}, "a": {c: (R[c][i] / rm if rm > 0 else 0.0) for c in cands}, "c": {c: Dx[c][i] for c in cands}}[d]
            s = {c: f[c] / fm - ALPHA * pen[c] for c in cands}
            order = sorted(cands, key=lambda c: (-s[c], c))
            row[f"top1({d})"] = order[0]
            row[f"top1_S({d})"] = round(s[order[0]], 5)
            row[f"margin_to_2nd({d})"] = round(s[order[0]] - s[order[1]], 5)
            row[f"second({d})"] = order[1]
        top.append(row)
    top.append({"rep": "mean of 3", **{f"top1({d})": best[d] for d in S}, **{f"top1_S({d})": round(S[d][best[d]].mean(), 5) for d in S}})
    write_csv(DIR / "repeat_5_4_top1.csv", top)

    n_ind = sum(p["uncertain_independent"] for p in pairs)
    n_pair = sum(p["uncertain_paired"] for p in pairs)
    n_sum = sum(p["uncertain_sum_rule(reference)"] for p in pairs)
    qd = defaultdict(list)
    for c in cands:
        for k in reps:
            for inst in by[c][k]["instances"]:
                qd[(c, inst["name"])].append(inst["window"]["q_i"])
    qr = np.array([max(v) - min(v) for v in qd.values()])
    frr = np.array([r["F_rel_range"] for r in rows])
    summ = {"runs_valid": len(runs), "runs_invalid": len(invalid), "candidates": len(cands), "pairs": len(pairs),
            "F_max_mean": fmax, "R_max_mean": rmax, "alpha": ALPHA,
            "uncertain_pairs_independent": n_ind, "uncertain_pairs_paired": n_pair, "uncertain_pairs_sum_rule": n_sum,
            "threshold_pairs": THRESHOLD_PAIRS, "decision": "scenario 1 (approved range)" if n_ind < THRESHOLD_PAIRS else "report and ask for approval of scenario 2",
            "best_measured": best, "best_measured_same_for_abc": len(set(best.values())) == 1,
            "top1_per_rep(b)": [t["top1(b)"] for t in top[:3]], "top1_stable(b)": len({t["top1(b)"] for t in top[:3]}) == 1,
            "top1_per_rep(a)": [t["top1(a)"] for t in top[:3]], "top1_per_rep(c)": [t["top1(c)"] for t in top[:3]],
            "F_rel_range_median": float(np.median(frr)), "F_rel_range_max": float(frr.max()),
            "S_se_median": float(np.median([r["S_se"] for r in rows])), "S_se_max": float(max(r["S_se"] for r in rows)),
            "q_i_abs_range_median": float(np.median(qr)), "q_i_abs_range_max": float(qr.max()),
            "best_vs_rest": [p for p in pairs if p["involves_best_measured"]]}
    (DIR / "summary.json").write_text(json.dumps(summ, indent=1))
    for r in rows:
        print(r["placement"], r["F_mean"], r["F_se"], r["D_macro_mean"], r["S_mean"], r["S_se"], r["S_agg_mean(a)"], r["S_max_mean(c)"])
    for t in top:
        print(t)
    print(json.dumps({k: v for k, v in summ.items() if k != "best_vs_rest"}, indent=1))
    for p in pairs:
        if p["uncertain_independent"] or p["uncertain_paired"]:
            print("UNCERTAIN", p)
    # per-instance q_i of the best-measured candidates when the definitions disagree
    if len(set(best.values())) > 1:
        for d, c in best.items():
            print("best", d, c, {i["name"]: (round(np.mean([by[c][k]["instances"][j]["window"]["q_i"] for k in reps]), 4), round(i["rate"], 1))
                                 for j, i in enumerate(by[c][0]["instances"])})


if __name__ == "__main__":
    main()
