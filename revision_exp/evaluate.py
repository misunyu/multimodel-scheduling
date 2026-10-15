"""Single evaluator used for every strategy.

Definitions
  S           = T - alpha * D                     (measured, labels unchanged)
  oracle set  = {candidates with S >= max(S) - 1e-9} inside a workload
  selection   = descending s_hat, ties -> ascending combination number
  Top-1       = selected candidate in the oracle set
  Top-5       = any of the 5 first candidates of that order in the oracle set (only for strategies that score every
                candidate; N/A for single-output strategies)
  Avg score   = workload mean of the measured S of the selected candidate
  Regret      = workload mean of max(S) - S_selected

select() never sees measured quantities; score_selection() never sees predictions.
"""
import sys

sys.dont_write_bytecode = True

import math

import numpy as np
import pandas as pd

TIE_TOL = 1e-9


def select(ids, s_hat):
    """ids: DataFrame[group, comb_num] (no measured columns). s_hat: array aligned with ids.
    Returns {group: [comb_num ordered best-first]} and {group: number of candidates tied with the first}."""
    assert set(ids.columns) == {"group", "comb_num"}, "select() must not receive measured columns"
    s_hat = np.asarray(s_hat, dtype=float)
    assert len(s_hat) == len(ids) and np.isfinite(s_hat).all()
    order, ties = {}, {}
    for g, idx in ids.groupby("group", sort=True).indices.items():
        c = ids["comb_num"].values[idx]
        s = s_hat[idx]
        o = np.lexsort((c, -s))
        order[g] = [int(x) for x in c[o]]
        ties[g] = int((s == s[o[0]]).sum())
    return order, ties


def single_choice(ids, chosen):
    """For single-output strategies: {group: comb_num or None} -> order dict with one element (or empty)."""
    return {g: ([int(chosen[g])] if chosen.get(g) is not None else []) for g in sorted(ids["group"].unique())}


def legacy_top5_groups(ids, s_hat):
    """Reference only: candidates whose s_hat is within the 5 highest DISTINCT values (old rule)."""
    out = {}
    s_hat = np.asarray(s_hat, dtype=float)
    for g, idx in ids.groupby("group", sort=True).indices.items():
        s = s_hat[idx]
        u = sorted(set(s.tolist()), reverse=True)
        thr = u[min(4, len(u) - 1)]
        out[g] = set(int(x) for x in ids["comb_num"].values[idx][s >= thr - 1e-7])
    return out


def score_selection(cand, order, alpha, strategy, top5=True, ties=None, legacy_top5=None):
    """cand: DataFrame[group, comb_num, combination, N, T, D]. order: {group: [comb_num, ...]}."""
    rows = []
    for g, sub in cand.groupby("group", sort=True):
        sub = sub.sort_values("comb_num")
        assert sub["comb_num"].is_unique, f"duplicate candidate in workload {g}"
        S = (sub["T"] - alpha * sub["D"]).values
        mx = S.max()
        oracle = set(int(c) for c, s in zip(sub["comb_num"], S) if s >= mx - TIE_TOL)
        o = order.get(g, [])
        base = {"strategy": strategy, "group": g, "N": int(sub["N"].iloc[0]), "n_candidates": len(sub),
                "n_candidates_le5": len(sub) <= 5, "oracle_S": float(mx), "n_oracle": len(oracle),
                "oracle_combination": "combination_" + str(min(oracle))}
        if not o:
            rows.append({**base, "evaluated": False})
            continue
        pos = {int(c): i for i, c in enumerate(sub["comb_num"])}
        assert all(c in pos for c in o), f"selected candidate not among measured candidates of {g}"
        i = pos[o[0]]
        r = {**base, "evaluated": True, "selected": "combination_" + str(o[0]), "S_selected": float(S[i]),
             "T_selected": float(sub["T"].values[i]), "D_selected": float(sub["D"].values[i]),
             "regret": float(mx - S[i]), "top1": o[0] in oracle,
             "top5": (any(c in oracle for c in o[:5]) if top5 else None),
             "n_tied_with_first": (ties or {}).get(g), "selected_all_gpu": bool(sub["all_gpu"].values[i]) if "all_gpu" in sub else None}
        if legacy_top5 is not None:
            r["top5_legacy_rule"] = bool(oracle & legacy_top5[g])
        rows.append(r)
    return pd.DataFrame(rows)


def aggregate(per, min_n=3, label=None):
    p = per[(per["N"] >= min_n)]
    e = p[p["evaluated"]]
    n = len(e)
    out = {"strategy": label or per["strategy"].iloc[0], "min_N": min_n, "workloads": n,
           "workloads_not_evaluated": int(len(p) - n),
           "not_evaluated_list": ";".join(p[~p["evaluated"]]["group"]) if len(p) - n else "",
           "avg_score": float(e["S_selected"].mean()) if n else float("nan"),
           "top1_hits": int(e["top1"].sum()) if n else 0, "top1_n": n,
           "regret": float(e["regret"].mean()) if n else float("nan"),
           "oracle_avg_same_workloads": float(e["oracle_S"].mean()) if n else float("nan"),
           "groups_le5_candidates": int(p["n_candidates_le5"].sum())}
    if n and e["top5"].notna().all():
        out["top5_hits"], out["top5_n"] = int(e["top5"].astype(bool).sum()), n
    else:
        out["top5_hits"], out["top5_n"] = None, None
    if "top5_legacy_rule" in e and n and e["top5_legacy_rule"].notna().all():
        out["top5_legacy_hits"] = int(e["top5_legacy_rule"].astype(bool).sum())
    if "selected_all_gpu" in e and n:
        out["selected_all_gpu"] = int(e["selected_all_gpu"].astype(bool).sum())
    return out


def frac(h, n):
    return "N/A" if h is None or n in (None, 0) or (isinstance(h, float) and math.isnan(h)) else f"{int(h)}/{int(n)} ({int(h) / int(n):.2f})"


def mae_table(df, t_hat, d_hat, label):
    rows = []
    for name, m in (("N>=1", np.ones(len(df), bool)), ("N>=3", (df["N"] >= 3).values)):
        t_err = np.abs(np.asarray(t_hat)[m] - df["T"].values[m])
        d_err = np.abs(np.asarray(d_hat)[m] - df["D"].values[m])
        pos = df["D"].values[m] > 0.01
        rows.append({"model": label, "subset": name, "candidates": int(m.sum()), "mae_T": float(t_err.mean()), "mae_D": float(d_err.mean()),
                     "mae_D_where_D>0.01": float(d_err[pos].mean()) if pos.any() else float("nan"), "candidates_D>0.01": int(pos.sum())})
    return rows


# ---------------------------------------------------------------------------------------------------- legacy rules
def legacy_evaluate(df_csv_order, pred_score, alpha):
    """Re-implementation of the OLD rules (deploy_selector_xgb_suite.py:1210-1356) for the replay check:
    groups in csv order with duplicates kept; oracle = isclose(rel 1e-7) to max; Top-1 = any predicted-best (ties) is oracle;
    Top-5 = oracle within the 5 highest distinct predicted values; reported score = first row of a stable descending sort."""
    d = df_csv_order.assign(_p=np.asarray(pred_score, dtype=float))
    rows = []
    for g, sub in d.groupby("group_id_legacy", sort=False):
        act = (sub["T"] - alpha * sub["D"]).values
        pred = sub["_p"].values
        names = sub["combination"].values
        mx = act.max()
        best_names = set(n for n, a in zip(names, act) if math.isclose(a, mx, rel_tol=1e-7))
        order = sorted(range(len(sub)), key=lambda i: pred[i], reverse=True)  # stable, like sorted(..., reverse=True)
        pmx = pred[order[0]]
        pred_best = [names[i] for i in order if math.isclose(pred[i], pmx, rel_tol=1e-7)]
        u = sorted(set(pred.tolist()), reverse=True)
        thr = u[min(4, len(u) - 1)]
        top5_names = set(n for n, p in zip(names, pred) if p >= thr - 1e-7)
        rows.append({"group": sub["group"].iloc[0], "N": int(sub["N"].iloc[0]), "n_rows": len(sub), "oracle_S": float(mx),
                     "S_selected": float(act[order[0]]), "selected": names[order[0]],
                     "top1": any(n in best_names for n in pred_best), "top5": bool(best_names & top5_names),
                     "regret": float(max(0.0, mx - act[order[0]])), "n_pred_best_ties": len(pred_best)})
    return pd.DataFrame(rows)


def self_test():
    """Unit tests required by the task: perfect scores -> 18/18 style result; ties resolved by combination number."""
    cand = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 6, "comb_num": [1, 2, 3, 4, 1, 2, 3, 4, 5, 6],
                         "combination": [f"combination_{i}" for i in [1, 2, 3, 4, 1, 2, 3, 4, 5, 6]], "N": [3] * 4 + [4] * 6,
                         "T": [0.2, 1.0, 1.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], "D": [0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0.5]})
    ids = cand[["group", "comb_num"]]
    S = (cand["T"] - 0.3 * cand["D"]).values
    o, t = select(ids, S)
    per = score_selection(cand, o, 0.3, "oracle", ties=t)
    a = aggregate(per)
    assert a["top1_hits"] == 2 and a["top5_hits"] == 2 and abs(a["regret"]) < 1e-12 and abs(a["avg_score"] - a["oracle_avg_same_workloads"]) < 1e-12
    assert o["a"][0] == 2 and t["a"] == 2, "tie must go to the lower combination number"
    o2, _ = select(ids, np.zeros(len(cand)))            # all tied -> combination_1 everywhere
    assert o2["a"][0] == 1 and o2["b"][0] == 1
    per2 = score_selection(cand, o2, 0.3, "flat")
    assert not per2["top1"].any() and per2.loc[per2.group == "a", "top5"].iloc[0] and not per2.loc[per2.group == "b", "top5"].iloc[0]
    per3 = score_selection(cand, single_choice(ids, {"a": 3, "b": None}), 0.3, "single", top5=False)
    a3 = aggregate(per3)
    assert a3["workloads"] == 1 and a3["workloads_not_evaluated"] == 1 and a3["top5_hits"] is None and a3["top1_hits"] == 1
    try:
        select(cand[["group", "comb_num", "T"]], S)
        raise SystemExit("select() accepted a measured column")
    except AssertionError:
        pass
    return "evaluate.self_test: 5 checks passed"
