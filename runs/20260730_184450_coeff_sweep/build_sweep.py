"""Appendix D coefficient sweep (alpha x beta) on MEASURED scores, plus checks A-E.

Read-only over the stored 540-window collection. No hardware, no re-collection, no
predictor output (this appendix is measured-score sensitivity, not model sensitivity),
no writes outside this run directory.

Definitions (fixed before running; not revised in light of results):
  baseline    = (alpha=0.3, beta=1.0); the beta term applies only to sets containing a
                generative model (llama1b / qwen2_vl), per score_combo.
  optimum     = tie set: every placement whose score is within 1e-9 of the group max
                (same rule as runs/20260730_172053_coverage).
  group change: tie set at (alpha,beta) has EMPTY intersection with the baseline tie set.
  set change  : any of that set's 3 rate groups changed at that grid point.

Grids:
  PRIMARY  -- pre-registered, recovered from SCORE_WEIGHTS_REVIEW.md section 2
              alpha in {0, 0.15, 0.3, 0.45, 0.6}, beta in {0, 0.25, 0.5, 0.75, 1.0}
  DECLARED -- the grid declared in the task prompt (alpha step 0.1)
  Both are computed and reported; PRIMARY is the headline. Neither was adjusted after
  seeing results.

Usage: python runs/<ts>_coeff_sweep/build_sweep.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from xgboost_model.deploy_selector_xgb_suite import (  # noqa: E402
    _norm_exec, _normalize_targets, score_combo)

P_SETS = "working_sets.yaml"
P_WIN = {pl: f"xgboost_model/full_collection_540/cpu_{pl}/performance_{pl}_full540.json"
         for pl in ("gpu", "npu")}

GEN_MODELS = {"llama1b", "qwen2_vl"}
BASE_ALPHA, BASE_BETA = 0.3, 1.0
TIE_EPS = 1e-9

# --- grids (see module docstring; fixed a priori) ---
PRIMARY_ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6]
PRIMARY_BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DECLARED_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DECLARED_BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

SETS = yaml.safe_load((ROOT / P_SETS).read_text())["sets"]
SID_OF = {",".join(sorted(v)): k for k, v in SETS.items()}
ORDER = ["S1", "S2", "S3", "S4", "base1", "base2", "base3", "base4", "base5",
         "S5", "S6", "S7", "S9", "S8", "S10"]
HAS_GEN = {s: any(m in GEN_MODELS for m in SETS[s]) for s in ORDER}


def placement_vector(window):
    """Platform-neutral placement: (model, 'cpu'|'accel') sorted by model name."""
    return tuple(sorted((v["model"],
                         "cpu" if _norm_exec(v["execution"]) == "cpu" else "accel")
                        for v in window["models"].values()))


def cpu_label(pv):
    """Compact human label: which models sit on the CPU (everything else is accel)."""
    cpu = [m for m, d in pv if d == "cpu"]
    return "all-accel" if not cpu else "cpu:" + "+".join(sorted(cpu))


def load_rows(platform):
    """One row per measured window: raw measured totals + placement + set/rate."""
    rows = []
    for w in json.loads((ROOT / P_WIN[platform]).read_text()):
        models = sorted(v["model"] for v in w["models"].values())
        sid = SID_OF[",".join(models)]
        t = w["total"]
        rows.append({
            "sid": sid, "rate": w["rate_factor"], "combo": w["combination"],
            "workload": w["workload"], "models": ",".join(models),
            "placement": placement_vector(w), "has_gen": HAS_GEN[sid],
            "y1": float(t["total_throughput_fps"]),
            "y2": float(t["deadline_miss_rate"]),
            "y3": float(t.get("total_tokens_per_s", 0.0)),
        })
    return rows


def normalized_rows(rows):
    """Same rows with y1/y2/y3 replaced by the suite's within-group [0,1] normalization.

    Used ONLY for the secondary robustness variant (report section F). Group key is
    (models, workload, rate_factor) -- exactly _normalize_targets' key.
    """
    Y = pd.DataFrame({"y1_total_throughput_fps": [r["y1"] for r in rows],
                      "y2_deadline_miss_rate": [r["y2"] for r in rows],
                      "y3_total_tokens_per_s": [r["y3"] for r in rows]})
    M = pd.DataFrame({"models": [r["models"] for r in rows],
                      "workload": [r["workload"] for r in rows],
                      "rate_factor": [r["rate"] for r in rows]})
    Yn = _normalize_targets(Y, M)
    out = []
    for i, r in enumerate(rows):
        q = dict(r)
        q["y1"] = float(Yn["y1_total_throughput_fps"].iloc[i])
        q["y2"] = float(Yn["y2_deadline_miss_rate"].iloc[i])
        q["y3"] = float(Yn["y3_total_tokens_per_s"].iloc[i])
        out.append(q)
    return out


def tie_sets(rows, alpha, beta):
    """(sid, rate) -> frozenset of tie-optimal placements at these coefficients."""
    g = defaultdict(list)
    for r in rows:
        s = score_combo(r["y1"], r["y2"], r["y3"] if r["has_gen"] else 0.0,
                        r["has_gen"], alpha, beta)
        g[(r["sid"], r["rate"])].append((s, r["placement"]))
    out = {}
    for k, items in g.items():
        best = max(s for s, _ in items)
        out[k] = frozenset(p for s, p in items if abs(s - best) < TIE_EPS)
    return out


def sweep(rows_by_pl, alphas, betas):
    """Grid sweep -> (csv row dicts, per-set change counts, per-gridpoint detail)."""
    base = {pl: tie_sets(rows_by_pl[pl], BASE_ALPHA, BASE_BETA) for pl in rows_by_pl}
    csv_rows, detail = [], {}
    set_counts = {pl: {s: 0 for s in ORDER} for pl in rows_by_pl}
    group_counts = {pl: defaultdict(int) for pl in rows_by_pl}
    n_points = 0
    for alpha in alphas:
        for beta in betas:
            n_points += 1
            rec = {"alpha": alpha, "beta": beta}
            for pl in ("gpu", "npu"):
                ts = tie_sets(rows_by_pl[pl], alpha, beta)
                changed_groups, changed_sets = [], set()
                for k in sorted(ts):
                    if not (ts[k] & base[pl][k]):
                        changed_groups.append(k)
                        changed_sets.add(k[0])
                        group_counts[pl][f"{k[0]}@{k[1]}"] += 1
                for s in changed_sets:
                    set_counts[pl][s] += 1
                ordered = [s for s in ORDER if s in changed_sets]
                rec[f"n_changed_sets_{pl}"] = len(ordered)
                rec[f"changed_sets_{pl}"] = ";".join(ordered)
                rec[f"n_changed_groups_{pl}"] = len(changed_groups)
                detail.setdefault(f"{alpha}|{beta}", {})[pl] = {
                    "changed_sets": ordered,
                    "changed_groups": [f"{k[0]}@{k[1]}" for k in changed_groups],
                }
            csv_rows.append(rec)
    return csv_rows, set_counts, dict(group_counts), detail, n_points, base


COLS = ["alpha", "beta", "n_changed_sets_gpu", "n_changed_sets_npu",
        "changed_sets_gpu", "changed_sets_npu",
        "n_changed_groups_gpu", "n_changed_groups_npu"]


def write_csv(path, rows):
    lines = [",".join(COLS)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in COLS))
    path.write_text("\n".join(lines) + "\n")


def optimum_map(rows, alphas, betas):
    """(sid, rate) -> {(alpha,beta): compact optimum label} -- material for section D."""
    m = defaultdict(dict)
    for alpha in alphas:
        for beta in betas:
            ts = tie_sets(rows, alpha, beta)
            for k, s in ts.items():
                m[f"{k[0]}@{k[1]}"][f"{alpha}|{beta}"] = "|".join(
                    sorted(cpu_label(p) for p in s))
    return dict(m)


def main():
    raw = {pl: load_rows(pl) for pl in ("gpu", "npu")}
    facts = {"grid_provenance": {
        "primary": {"source": "SCORE_WEIGHTS_REVIEW.md section 2 (beta sweep table: "
                              "0/0.25/0.5/0.75/1.0; alpha sweep table: 0/0.15/0.3/0.45/0.6)",
                    "alphas": PRIMARY_ALPHAS, "betas": PRIMARY_BETAS,
                    "n_points": len(PRIMARY_ALPHAS) * len(PRIMARY_BETAS)},
        "declared": {"source": "task prompt (used because its alpha list differs from "
                               "the pre-registered one; reported alongside, not instead)",
                     "alphas": DECLARED_ALPHAS, "betas": DECLARED_BETAS,
                     "n_points": len(DECLARED_ALPHAS) * len(DECLARED_BETAS)},
    }, "definitions": {
        "baseline": {"alpha": BASE_ALPHA, "beta": BASE_BETA},
        "tie_eps": TIE_EPS,
        "tie_rule": "optimum = all placements within 1e-9 of the group max; a group "
                    "CHANGED iff its tie set does not intersect the baseline tie set",
        "score_basis": "raw measured window totals (score_combo inputs); no predictor",
    }}

    # ---------- primary + declared sweeps (raw measured scores) ----------
    prim, prim_sets, prim_groups, prim_detail, n_prim, base_ts = sweep(
        raw, PRIMARY_ALPHAS, PRIMARY_BETAS)
    decl, decl_sets, decl_groups, decl_detail, n_decl, _ = sweep(
        raw, DECLARED_ALPHAS, DECLARED_BETAS)
    write_csv(OUT / "sweep_setcounts.csv", prim)
    write_csv(OUT / "sweep_setcounts_declared_grid.csv", decl)

    for tag, rows_, sc, gc, det, npt in (("primary", prim, prim_sets, prim_groups,
                                          prim_detail, n_prim),
                                         ("declared", decl, decl_sets, decl_groups,
                                          decl_detail, n_decl)):
        facts[tag] = {
            "n_grid_points": npt,
            "gpu_max_changed_sets": max(r["n_changed_sets_gpu"] for r in rows_),
            "npu_max_changed_sets": max(r["n_changed_sets_npu"] for r in rows_),
            "gpu_violations": [{"alpha": r["alpha"], "beta": r["beta"],
                                "sets": r["changed_sets_gpu"]}
                               for r in rows_ if r["n_changed_sets_gpu"] > 0],
            "npu_argmax_points": [{"alpha": r["alpha"], "beta": r["beta"],
                                   "sets": r["changed_sets_npu"]}
                                  for r in rows_
                                  if r["n_changed_sets_npu"] ==
                                  max(x["n_changed_sets_npu"] for x in rows_)],
            "per_set_change_counts": sc,
            "per_group_change_counts": {pl: dict(gc[pl]) for pl in ("gpu", "npu")},
        }

    # ---------- A: baseline self-check ----------
    b = next(r for r in prim if r["alpha"] == BASE_ALPHA and r["beta"] == BASE_BETA)
    facts["A_baseline_selfcheck"] = {
        "alpha": b["alpha"], "beta": b["beta"],
        "n_changed_sets_gpu": b["n_changed_sets_gpu"],
        "n_changed_sets_npu": b["n_changed_sets_npu"],
        "ok": b["n_changed_sets_gpu"] == 0 and b["n_changed_sets_npu"] == 0}
    facts["A_gpu_zero_everywhere"] = {
        "primary": facts["primary"]["gpu_max_changed_sets"] == 0,
        "declared": facts["declared"]["gpu_max_changed_sets"] == 0}
    facts["A_npu_max_is_7"] = {
        "primary": facts["primary"]["npu_max_changed_sets"] == 7,
        "declared": facts["declared"]["npu_max_changed_sets"] == 7}

    # ---------- C: beta-sensitive groups at alpha=0.3, beta 1.0 vs 0.5 ----------
    c_diff = {}
    for pl in ("gpu", "npu"):
        t10 = tie_sets(raw[pl], 0.3, 1.0)
        t05 = tie_sets(raw[pl], 0.3, 0.5)
        c_diff[pl] = [{"group": f"{k[0]}@{k[1]}",
                       "beta1_0": sorted(cpu_label(p) for p in t10[k]),
                       "beta0_5": sorted(cpu_label(p) for p in t05[k])}
                      for k in sorted(t10) if not (t10[k] & t05[k])]
    facts["C_beta_sensitive_groups"] = c_diff
    facts["C_counts"] = {pl: len(v) for pl, v in c_diff.items()}

    # ---------- D: optimum map (material for the appendix caption) ----------
    facts["D_optimum_map_npu"] = optimum_map(raw["npu"], PRIMARY_ALPHAS, PRIMARY_BETAS)
    facts["D_optimum_map_gpu"] = optimum_map(raw["gpu"], PRIMARY_ALPHAS, PRIMARY_BETAS)

    # ---------- E: reconciliation with the published divergence figures ----------
    # Cross-platform disagreement at the two betas, same rule as the coverage run.
    recon = {}
    for beta in (1.0, 0.5):
        tg, tn = tie_sets(raw["gpu"], 0.3, beta), tie_sets(raw["npu"], 0.3, beta)
        recon[str(beta)] = {"n_groups": len(tg),
                            "n_disagree": sum(1 for k in tg if not (tg[k] & tn[k]))}
    facts["E_divergence_recheck"] = recon
    facts["E_matches_platform_divergence_md"] = (recon["1.0"]["n_disagree"] == 29
                                                 and recon["0.5"]["n_disagree"] == 30)

    # ---------- F: secondary robustness variant on normalized measured targets ----------
    # The raw-total score mixes units (y1 ~ 100-230 fps, y2 in [0,1], y3 ~ 0-50 tok/s),
    # so alpha*y2 <= 0.6 can barely move a raw argmax. Recomputed on the suite's
    # within-group [0,1] normalization to show whether alpha-insensitivity is a scale
    # artifact. SECONDARY -- does not replace the primary numbers above.
    norm = {pl: normalized_rows(raw[pl]) for pl in ("gpu", "npu")}
    nrows, nsets, _, _, nn, _ = sweep(norm, PRIMARY_ALPHAS, PRIMARY_BETAS)
    write_csv(OUT / "sweep_setcounts_normalized.csv", nrows)
    facts["F_normalized_variant"] = {
        "note": "secondary; group min-max/Fmax normalization of MEASURED targets via "
                "deploy_selector_xgb_suite._normalize_targets (not predictor output)",
        "n_grid_points": nn,
        "gpu_max_changed_sets": max(r["n_changed_sets_gpu"] for r in nrows),
        "npu_max_changed_sets": max(r["n_changed_sets_npu"] for r in nrows),
        "gpu_violations": [{"alpha": r["alpha"], "beta": r["beta"],
                            "sets": r["changed_sets_gpu"]}
                           for r in nrows if r["n_changed_sets_gpu"] > 0],
        "per_set_change_counts": nsets,
    }
    # magnitude evidence for the scale caveat
    facts["F_raw_scale"] = {
        pl: {"y1_min": min(r["y1"] for r in raw[pl]), "y1_max": max(r["y1"] for r in raw[pl]),
             "y2_min": min(r["y2"] for r in raw[pl]), "y2_max": max(r["y2"] for r in raw[pl]),
             "y3_max": max(r["y3"] for r in raw[pl])}
        for pl in ("gpu", "npu")}

    (OUT / "sweep_facts.json").write_text(
        json.dumps(facts, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in facts.items()
                      if not k.startswith("D_optimum_map")},
                     indent=2, ensure_ascii=False, default=str)[:6000])
    return facts


if __name__ == "__main__":
    main()
