"""Score-basis audit + recomputation of headline numbers on the paper's declared basis.

Read-only over the stored 540-window collection. No re-collection, no re-training, no
manuscript or existing-artifact edits, no writes outside this run directory.

DECLARED BASIS (paper Eq. (1)), fixed before running:
  group = (set, rate). Within a group:
    y1 = r1 / max(r1);  y3 = r3 / max(r3);  y2 = (r2 - min r2) / (max r2 - min r2),
    and y2 = 0 when max == min.
  S = y1 - 0.3*y2 + beta*y3   (beta term only for sets containing a generative model)
  beta = 1.0 primary, 0.5 secondary.  r1/r2/r3 = full540 measured WINDOW TOTALS.
  tie rule: 1e-9 tie set; intersecting tie sets = agreement.

y3 masking convention (inherited from deploy_selector_xgb_suite._normalize_targets, as
instructed): max(r3) is taken over EVERY row of the group, not only over y3-valid rows
(y3_valid = "some generative view sits on an accelerator"). Masked candidates therefore
participate in the denominator. Their measured r3 is near-zero, so this is checked
empirically below (facts["y3_mask_convention"]) rather than assumed harmless.

Two r1 conventions exist in the codebase and are BOTH computed here:
  totals  -- r1 = total.total_throughput_fps (sum over ALL views). The declared basis.
  vision  -- r1 = sum of throughput_fps over VISION views only, which is what
             featurize_window actually feeds the predictor pipeline.

Usage: python runs/<ts>_score_basis_audit/build_basis_audit.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from xgboost_model.deploy_selector_xgb_suite import _is_vision, _norm_exec  # noqa: E402

P_WIN = {pl: f"xgboost_model/full_collection_540/cpu_{pl}/performance_{pl}_full540.json"
         for pl in ("gpu", "npu")}
P_DIV = "xgboost_model/full_collection_540/analysis/platform_divergence.csv"

GEN_MODELS = {"llama1b", "qwen2_vl"}
ALPHA = 0.3
TIE_EPS = 1e-9
PRIMARY_ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6]
PRIMARY_BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BASE_ALPHA, BASE_BETA = 0.3, 1.0

SETS = yaml.safe_load((ROOT / "working_sets.yaml").read_text())["sets"]
SID_OF = {",".join(sorted(v)): k for k, v in SETS.items()}
ORDER = ["S1", "S2", "S3", "S4", "base1", "base2", "base3", "base4", "base5",
         "S5", "S6", "S7", "S9", "S8", "S10"]
HAS_GEN = {s: any(m in GEN_MODELS for m in SETS[s]) for s in ORDER}
# mode == exhaustive sets, established in runs/20260730_172053_coverage
EXHAUSTIVE = ["S1", "S2", "S3", "S4", "base1", "base2", "base3", "base4", "base5", "S5"]


def cpu_label(pv):
    cpu = [m for m, d in pv if d == "cpu"]
    return "all-accel" if not cpu else "cpu:" + "+".join(sorted(cpu))


def load_rows(platform):
    rows = []
    for w in json.loads((ROOT / P_WIN[platform]).read_text()):
        models = sorted(v["model"] for v in w["models"].values())
        sid = SID_OF[",".join(models)]
        pv = tuple(sorted((v["model"],
                           "cpu" if _norm_exec(v["execution"]) == "cpu" else "accel")
                          for v in w["models"].values()))
        vision_fps = sum(float(v.get("throughput_fps") or 0.0)
                         for v in w["models"].values() if _is_vision(v["model"]))
        y3_valid = any(_norm_exec(v["execution"]) in ("gpu", "npu")
                       and not _is_vision(v["model"]) for v in w["models"].values())
        t = w["total"]
        rows.append({
            "sid": sid, "rate": w["rate_factor"], "combo": w["combination"],
            "placement": pv, "label": cpu_label(pv), "has_gen": HAS_GEN[sid],
            "r1_totals": float(t["total_throughput_fps"]),
            "r1_vision": vision_fps,
            "r2": float(t["deadline_miss_rate"]),
            "r3": float(t.get("total_tokens_per_s", 0.0)),
            "y3_valid": y3_valid,
        })
    return rows


def normalize(rows, r1_key="r1_totals"):
    """Declared-basis group normalization; returns rows with n1/n2/n3 added."""
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[(r["sid"], r["rate"])].append(i)
    out = [dict(r) for r in rows]
    for _, idxs in g.items():
        m1 = max(out[i][r1_key] for i in idxs)
        m3 = max(out[i]["r3"] for i in idxs)
        lo = min(out[i]["r2"] for i in idxs)
        hi = max(out[i]["r2"] for i in idxs)
        for i in idxs:
            out[i]["n1"] = out[i][r1_key] / m1 if m1 > 1e-9 else 0.0
            out[i]["n3"] = out[i]["r3"] / m3 if m3 > 1e-9 else 0.0
            out[i]["n2"] = (out[i]["r2"] - lo) / (hi - lo) if (hi - lo) > 1e-9 else 0.0
    return out


def score(r, alpha, beta, basis):
    if basis == "normalized":
        s = r["n1"] - alpha * r["n2"]
        return s + beta * r["n3"] if r["has_gen"] else s
    s = r["r1_totals"] - alpha * r["r2"]
    return s + beta * r["r3"] if r["has_gen"] else s


def tie_sets(rows, alpha, beta, basis):
    g = defaultdict(list)
    for r in rows:
        g[(r["sid"], r["rate"])].append((score(r, alpha, beta, basis), r["placement"]))
    out = {}
    for k, items in g.items():
        best = max(s for s, _ in items)
        out[k] = frozenset(p for s, p in items if abs(s - best) < TIE_EPS)
    return out


def divergence(rows_by_pl, alpha, beta, basis, keys=None):
    tg = tie_sets(rows_by_pl["gpu"], alpha, beta, basis)
    tn = tie_sets(rows_by_pl["npu"], alpha, beta, basis)
    ks = sorted(k for k in tg if keys is None or k[0] in keys)
    dis = [k for k in ks if not (tg[k] & tn[k])]
    gen = [k for k in ks if HAS_GEN[k[0]]]
    vis = [k for k in ks if not HAS_GEN[k[0]]]
    return {
        "n_groups": len(ks), "n_disagree": len(dis),
        "gen": {"n": len(gen), "d": sum(1 for k in dis if HAS_GEN[k[0]])},
        "vision": {"n": len(vis), "d": sum(1 for k in dis if not HAS_GEN[k[0]])},
        "disagree_groups": [f"{k[0]}@{k[1]}" for k in dis],
        "n_groups_with_ties": sum(1 for k in ks if len(tg[k]) > 1 or len(tn[k]) > 1),
    }, tg, tn


def sweep_setcounts(rows_by_pl, basis, alphas, betas):
    base = {pl: tie_sets(rows_by_pl[pl], BASE_ALPHA, BASE_BETA, basis)
            for pl in rows_by_pl}
    csv_rows, per_set = [], {pl: {s: 0 for s in ORDER} for pl in rows_by_pl}
    for alpha in alphas:
        for beta in betas:
            rec = {"alpha": alpha, "beta": beta}
            for pl in ("gpu", "npu"):
                ts = tie_sets(rows_by_pl[pl], alpha, beta, basis)
                changed = {k[0] for k in sorted(ts) if not (ts[k] & base[pl][k])}
                ngroups = sum(1 for k in ts if not (ts[k] & base[pl][k]))
                for s in changed:
                    per_set[pl][s] += 1
                ordered = [s for s in ORDER if s in changed]
                rec[f"n_changed_sets_{pl}"] = len(ordered)
                rec[f"changed_sets_{pl}"] = ";".join(ordered)
                rec[f"n_changed_groups_{pl}"] = ngroups
            csv_rows.append(rec)
    return csv_rows, per_set


SWEEP_COLS = ["alpha", "beta", "n_changed_sets_gpu", "n_changed_sets_npu",
              "changed_sets_gpu", "changed_sets_npu",
              "n_changed_groups_gpu", "n_changed_groups_npu"]


def main():
    raw = {pl: load_rows(pl) for pl in ("gpu", "npu")}
    norm = {pl: normalize(raw[pl], "r1_totals") for pl in raw}
    norm_vision = {pl: normalize(raw[pl], "r1_vision") for pl in raw}
    facts = {}

    # ---------- basis discrepancy evidence (part 1) ----------
    facts["r1_convention_gap"] = {
        pl: {"n_rows_differing": sum(1 for r in raw[pl]
                                     if abs(r["r1_totals"] - r["r1_vision"]) > 1e-9),
             "max_abs_diff_fps": max(abs(r["r1_totals"] - r["r1_vision"])
                                     for r in raw[pl]),
             "note": "total.total_throughput_fps (all views) vs featurize_window's "
                     "vision-only y1"}
        for pl in raw}
    # does including masked rows in the y3 denominator change it?
    mask_effect = {}
    for pl in raw:
        g = defaultdict(list)
        for r in raw[pl]:
            g[(r["sid"], r["rate"])].append(r)
        changed = 0
        for k, rs in g.items():
            all_max = max(r["r3"] for r in rs)
            val = [r["r3"] for r in rs if r["y3_valid"]]
            if val and abs(max(val) - all_max) > 1e-9:
                changed += 1
        mask_effect[pl] = {"groups_where_denominator_differs": changed,
                           "n_groups": len(g)}
    facts["y3_mask_convention"] = {
        "rule": "_normalize_targets takes max(r3) over ALL rows of the group; y3_valid "
                "is used only to mask TRAINING rows, never the normalization denominator",
        "empirical_effect": mask_effect}

    # ---------- part 1 cross-check: 29/33 gen, 0/12 vision on the RAW basis ----------
    raw_div = {}
    for beta in (1.0, 0.5):
        d, _, _ = divergence(raw, ALPHA, beta, "raw")
        raw_div[str(beta)] = d
    facts["part1_raw_basis_divergence"] = raw_div
    facts["part1_reproduces_29_45_and_30_45"] = (raw_div["1.0"]["n_disagree"] == 29
                                                 and raw_div["0.5"]["n_disagree"] == 30)
    facts["part1_reproduces_29of33_and_0of12"] = (
        raw_div["1.0"]["gen"] == {"n": 33, "d": 29}
        and raw_div["1.0"]["vision"] == {"n": 12, "d": 0})

    # ---------- part 2a: 45-group divergence on the declared basis ----------
    norm_div, tg_by_beta, tn_by_beta = {}, {}, {}
    for beta in (1.0, 0.5):
        d, tg, tn = divergence(norm, ALPHA, beta, "normalized")
        norm_div[str(beta)] = d
        tg_by_beta[beta], tn_by_beta[beta] = tg, tn
    facts["part2a_declared_basis_divergence"] = norm_div

    # per-group difference vs the raw basis (beta=1.0 and 0.5)
    diffs = {}
    for beta in (1.0, 0.5):
        rd, rg, rn = divergence(raw, ALPHA, beta, "raw")
        nd = norm_div[str(beta)]
        rset, nset = set(rd["disagree_groups"]), set(nd["disagree_groups"])
        diffs[str(beta)] = {
            "raw_only": sorted(rset - nset), "normalized_only": sorted(nset - rset),
            "n_raw": len(rset), "n_normalized": len(nset)}
    facts["part2a_group_level_diff_vs_raw"] = diffs

    # beta-sensitive groups on the declared basis (own-optimum move, per platform)
    beta_sens = {}
    for pl, tsrc in (("gpu", tg_by_beta), ("npu", tn_by_beta)):
        beta_sens[pl] = [f"{k[0]}@{k[1]}" for k in sorted(tsrc[1.0])
                         if not (tsrc[1.0][k] & tsrc[0.5][k])]
    # and groups whose AGREEMENT verdict flips between the two betas
    flip = sorted(set(norm_div["1.0"]["disagree_groups"])
                  ^ set(norm_div["0.5"]["disagree_groups"]))
    facts["part2a_beta_sensitive"] = {"own_optimum_moves": beta_sens,
                                      "agreement_verdict_flips": flip}

    # ---------- part 2b: exhaustive-only ----------
    exh = {}
    for beta in (1.0, 0.5):
        d, _, _ = divergence(norm, ALPHA, beta, "normalized", keys=EXHAUSTIVE)
        r, _, _ = divergence(raw, ALPHA, beta, "raw", keys=EXHAUSTIVE)
        exh[str(beta)] = {"declared_basis": d, "raw_basis": r}
    facts["part2b_exhaustive"] = exh

    # ---------- part 2c/2d: coefficient sweep on the declared basis ----------
    sw, per_set = sweep_setcounts(norm, "normalized", PRIMARY_ALPHAS, PRIMARY_BETAS)
    (OUT / "sweep_declared_basis.csv").write_text(
        "\n".join([",".join(SWEEP_COLS)]
                  + [",".join(str(r[c]) for c in SWEEP_COLS) for r in sw]) + "\n")
    b0 = [r for r in sw if r["beta"] == 0.0]
    bpos = [r for r in sw if r["beta"] > 0.0]
    facts["part2c_sweep"] = {
        "grid": {"alphas": PRIMARY_ALPHAS, "betas": PRIMARY_BETAS, "n_points": len(sw)},
        "baseline": {"alpha": BASE_ALPHA, "beta": BASE_BETA},
        "gpu_max_changed_sets": max(r["n_changed_sets_gpu"] for r in sw),
        "npu_max_changed_sets": max(r["n_changed_sets_npu"] for r in sw),
        "per_set_change_counts": per_set,
        "prev_F_variant_reference": {"gpu_max": 9, "npu_max": 11,
                                     "source": "runs/20260730_184450_coeff_sweep "
                                               "sweep_facts.json F_normalized_variant"},
    }
    facts["part2c_matches_prev_F"] = (
        facts["part2c_sweep"]["gpu_max_changed_sets"] == 9
        and facts["part2c_sweep"]["npu_max_changed_sets"] == 11)
    facts["part2d_beta_split"] = {
        "beta_zero_column": {
            "n_points": len(b0),
            "gpu_max": max(r["n_changed_sets_gpu"] for r in b0),
            "npu_max": max(r["n_changed_sets_npu"] for r in b0)},
        "beta_positive_columns": {
            "n_points": len(bpos),
            "gpu_max": max(r["n_changed_sets_gpu"] for r in bpos),
            "npu_max": max(r["n_changed_sets_npu"] for r in bpos)},
        "note": "reported separately without a recommendation on the paper's range",
    }

    # sensitivity of part 2a to the r1 convention (totals vs vision-only)
    vis_div = {}
    for beta in (1.0, 0.5):
        d, _, _ = divergence(norm_vision, ALPHA, beta, "normalized")
        vis_div[str(beta)] = {"n_disagree": d["n_disagree"], "gen": d["gen"],
                              "vision": d["vision"],
                              "disagree_groups": d["disagree_groups"]}
    facts["part2a_r1_vision_variant"] = vis_div

    # ---------- per-group CSV ----------
    lines = ["basis,beta,set,rate,has_gen,gpu_best,npu_best,agree"]
    for basis, rws in (("raw", raw), ("declared_normalized", norm)):
        for beta in (1.0, 0.5):
            tg = tie_sets(rws["gpu"], ALPHA, beta,
                          "raw" if basis == "raw" else "normalized")
            tn = tie_sets(rws["npu"], ALPHA, beta,
                          "raw" if basis == "raw" else "normalized")
            for k in sorted(tg):
                lines.append(
                    f"{basis},{beta},{k[0]},{k[1]},{int(HAS_GEN[k[0]])},"
                    f"\"{'|'.join(sorted(cpu_label(p) for p in tg[k]))}\","
                    f"\"{'|'.join(sorted(cpu_label(p) for p in tn[k]))}\","
                    f"{int(bool(tg[k] & tn[k]))}")
    (OUT / "divergence_by_group.csv").write_text("\n".join(lines) + "\n")

    (OUT / "basis_facts.json").write_text(
        json.dumps(facts, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(facts, indent=2, ensure_ascii=False, default=str)[:7000])
    return facts


if __name__ == "__main__":
    main()
