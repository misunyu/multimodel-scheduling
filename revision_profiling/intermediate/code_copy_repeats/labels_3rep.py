"""Phase 4b: 3-repeat labels, repeat-deviation statistics and per-repeat oracles. New file.

The candidate labels are built by the ORIGINAL code/recompute_scores.py functions (load_runs, build_tables), unchanged — the same code that
built the Phase 2 3x labels (Phase 2 v1.2 section 3.1):
  per candidate: F_bar = mean_r F_r, D_bar_macro = mean_r D_macro_r (D_macro_r = run value system.drop_macro = mean_i q_{i,r}; checked below),
  per workload: F_max = max_cand F_bar, T = F_bar / F_max, S = T - 0.3 D_bar_macro.
  per repeat (stored next to the mean): S_r = F_r / F_max - 0.3 D_macro_r with the COMMON F_max of the 3-repeat means (recompute_scores.build_tables,
  column S_macro_a0.3_rep<r>); the SE columns are sample std (ddof=1) / sqrt(n_reps) (recompute_scores.se) -> S_macro_a0.3_se is the SE of these S_r.
Per-repeat oracle (Phase 2 code/integrity.py rule): each repeat with its OWN normalisation, S_r' = F_r / max_cand F_r - 0.3 D_macro_r
  (what a single-run campaign would have produced); oracle set = candidates with S_r' >= max - 1e-9; representative id = lowest combination number.

This module also holds the PRECHECK section 0-1 (apply to the Phase 2 3x raw repeats and compare with Phase 2 dataset/labels_test.csv).
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IROOT = HERE.parent
ROOT = IROOT.parent
sys.path.insert(0, str(ROOT / "code"))
import recompute_scores as RS  # noqa: E402  (original, unchanged)

assert Path(RS.__file__).resolve().parent == ROOT / "code"
ALPHA = 0.3
TOL = 1e-9
REP_DIRS = [IROOT / "results", IROOT / "results_rep1", IROOT / "results_rep2"]


def build(dirs):
    valid, invalid = RS.load_runs(dirs)
    cand, inst, wl = RS.build_tables(valid)
    cand["workload"] = cand["schedule_file"].str[:-5]
    return valid, invalid, cand, inst, wl


def dmacro_from_instances(cand, inst):
    """D_macro per repeat recomputed as mean_i q_{i,r} from the instance table, and its 3-repeat mean. Returns the max abs differences."""
    out = {}
    g = inst.groupby(["family", "multiplier", "combination"])
    for k in (0, 1, 2):
        q = g[f"q_i_i_rep{k}"].mean().rename(f"q_rep{k}")
        m = cand.merge(q.reset_index(), on=["family", "multiplier", "combination"], how="left", validate="one_to_one")
        ok = m[f"D_macro_rep{k}"].notna()
        out[f"rep{k}"] = float((m.loc[ok, f"D_macro_rep{k}"] - m.loc[ok, f"q_rep{k}"]).abs().max()) if ok.any() else None
    qm = g[[f"q_i_i_rep{k}" for k in (0, 1, 2)]].mean()
    qbar = qm.mean(axis=1, skipna=True).rename("qbar").reset_index()
    m = cand.merge(qbar, on=["family", "multiplier", "combination"], how="left", validate="one_to_one")
    out["mean"] = float((m["D_macro_mean"] - m["qbar"]).abs().max())
    return out


def se_stats(df):
    """Distribution of the 3-repeat SE of F, D_macro and S (S_r with the common F_max)."""
    rows = []
    for name, col in (("F (fps)", "F_se"), ("F relative (SE / F_bar)", "_F_rel_se"), ("D_macro", "D_macro_se"), ("S", "S_macro_a0.3_se")):
        x = (df["F_se"] / df["F_mean"]) if col == "_F_rel_se" else df[col]
        rows.append({"metric": name, "candidates": int(x.notna().sum()), "median": float(x.median()), "p90": float(x.quantile(0.9)), "max": float(x.max())})
    return pd.DataFrame(rows)


def rep_oracles(cand, reps=(0, 1, 2)):
    """Per workload: oracle set of every repeat (own normalisation, tol 1e-9), common set, representative ids, mean-label oracle set."""
    rows = []
    for wk, sub in cand.groupby("workload"):
        sub = sub.sort_values("comb_num")
        cn = sub["comb_num"].values
        s_mean = sub["S_macro_a0.3"].values
        mset = set(cn[s_mean >= s_mean.max() - TOL])
        row = {"workload": wk, "family": sub["family"].iloc[0], "N": int(sub["N"].iloc[0]), "multiplier": float(sub["multiplier"].iloc[0]),
               "candidates": len(sub), "oracle_mean_label": "combination_%03d" % min(mset), "n_oracle_mean_label": len(mset)}
        sets, reps_ids = [], []
        for k in reps:
            F, Dm = sub[f"F_rep{k}"].values, sub[f"D_macro_rep{k}"].values
            if np.isnan(F).any():
                sets.append(None)
                reps_ids.append(None)
                row[f"oracle_rep{k}"] = None
                row[f"n_oracle_rep{k}"] = None
                continue
            s = F / F.max() - ALPHA * Dm
            o = set(cn[s >= s.max() - TOL])
            sets.append(o)
            reps_ids.append(min(o))
            row[f"oracle_rep{k}"] = "combination_%03d" % min(o)
            row[f"n_oracle_rep{k}"] = len(o)
            row[f"oracle_set_rep{k}"] = ";".join(str(x) for x in sorted(o))
        have = [x for x in sets if x is not None]
        common = set.intersection(*have) if len(have) == len(reps) else set()
        row["common_optimal_exists"] = bool(common)
        row["common_optimal_set"] = ";".join(str(x) for x in sorted(common))
        row["representative_ids_equal"] = len(have) == len(reps) and len(set(reps_ids)) == 1
        row["mean_label_oracle_in_common"] = bool(mset & common)
        rows.append(row)
    return pd.DataFrame(rows)


def phase2_x3_dirs():
    return sorted(p for p in (ROOT / "results").glob("*/x3") if p.is_dir())


def precheck_phase2():
    """PRECHECK 0-1: the label code on the Phase 2 3x raw repeats vs Phase 2 dataset/labels_test.csv (1e-9), SE median 0.0010,
    per-repeat oracles vs Phase 2 results/integrity/oracle_x3.csv."""
    valid, invalid, cand, inst, wl = build(phase2_x3_dirs())
    ref = pd.read_csv(ROOT / "dataset" / "labels_test.csv")
    m = cand.merge(ref, on=["schedule_file", "combination"], suffixes=("", "_p2"), validate="one_to_one")
    cols = ["F_mean", "D_macro_mean", "throughput_norm", "S_macro_a0.3", "F_max", "F_se", "D_macro_se", "S_macro_a0.3_se"] + \
           [f"{c}_rep{k}" for c in ("F", "D_macro", "S_macro_a0.3") for k in (0, 1, 2)]
    diffs = {}
    for c in cols:
        a, b = m[c].values.astype(float), m[f"{c}_p2"].values.astype(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        diffs[c] = float(np.nanmax(d)) if not np.isnan(d).any() else float("inf")
    n3 = m[(m["N"] >= 3)]
    se = se_stats(n3)
    orc = rep_oracles(cand[cand["N"] >= 3])
    p2o = pd.read_csv(ROOT / "results" / "integrity" / "oracle_x3.csv")
    oc = orc.merge(p2o[["family", "oracle_rep0", "oracle_rep1", "oracle_rep2", "reps_equal_to_each_other"]], on="family", suffixes=("", "_p2"), validate="one_to_one")
    same_ids = all(int(str(oc.loc[i, f"oracle_rep{k}"]).rsplit("_", 1)[1]) == int(oc.loc[i, f"oracle_rep{k}_p2"]) for i in oc.index for k in (0, 1, 2))
    dq = dmacro_from_instances(cand, inst)
    se_s = se.set_index("metric").loc["S"]
    res = {
        "runs_loaded": len(valid), "invalid": len(invalid), "candidates": len(cand), "candidates_matched_with_phase2_labels_test": len(m),
        "rows_phase2_labels_test": len(ref), "max_abs_diff": diffs,
        "D_macro_run_value_vs_mean_i_q_i": dq,
        "se_stats_N>=3": se.to_dict(orient="records"),
        "rep_oracle_ids_equal_phase2_oracle_x3": same_ids,
        "workloads_rep_ids_all_equal": int(orc["representative_ids_equal"].sum()),
        "workloads_rep_ids_all_equal_phase2": int(p2o["reps_equal_to_each_other"].sum()),
        "workloads_common_optimal": int(orc["common_optimal_exists"].sum()),
        "oracles": orc,
    }
    res["checks"] = {
        "Phase 2 3x: 22 workloads, 504 candidates, all rows matched": len(m) == len(ref) == len(cand) == 504,
        "F_bar, D_bar_macro, T, S, F_max, SE and per-repeat columns equal Phase 2 labels_test.csv (max abs diff <= 1e-9)": all(v <= 1e-9 for v in diffs.values()),
        "D_macro_r (run value) = mean_i q_{i,r}, and D_bar_macro = mean_r mean_i q_{i,r} (<= 1e-12)": all(v is not None and v <= 1e-12 for v in dq.values()),
        "S SE median (3x N>=3, 480 candidates) = 0.0010 (Phase 2 INTEGRITY.md / REPORT_PHASE2.md); p90 0.0027, max 0.0130":
            se_s["candidates"] == 480 and round(se_s["median"], 4) == 0.0010 and round(se_s["p90"], 4) == 0.0027 and round(se_s["max"], 4) == 0.0130,
        "per-repeat oracle ids (own normalisation) = Phase 2 results/integrity/oracle_x3.csv, 'all three equal' 8/12":
            same_ids and int(orc["representative_ids_equal"].sum()) == int(p2o["reps_equal_to_each_other"].sum()) == 8,
    }
    return res
