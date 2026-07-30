"""Appendix sampling-coverage table + verifications A-E for the MLForSys appendix.

Read-only over already-collected artifacts. No hardware, no re-collection, no writes
outside this run directory.

Inputs (all paths recorded in run_manifest.json):
  working_sets.yaml                                    -- set definitions, strata config
  model_registry.py                     DEVICE_CONSTRAINTS -- the VLM no-CPU constraint
  xgboost_model/gen_collection_schedules.py            -- candidate generation + strata
  xgboost_model/schedules/collection/collect_cpu_{gpu,npu}.yaml(.meta.json)
                                                       -- the scheduled placements
  xgboost_model/full_collection_540/cpu_{gpu,npu}/performance_{gpu,npu}_full540.json
                                                       -- the measured windows

Outputs: appendix_coverage.csv, appendix_coverage_by_rate.csv (only if rate levels
diverge), coverage_report.md, coverage_facts.json.

Usage: python runs/<ts>_coverage/build_coverage.py
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import model_registry as reg  # noqa: E402
from xgboost_model.deploy_selector_xgb_suite import _norm_exec, score_combo  # noqa: E402

# ---- source paths (verbatim, for source_ref / manifest) ----
P_SETS = "working_sets.yaml"
P_REGISTRY = "model_registry.py"
P_GEN = "xgboost_model/gen_collection_schedules.py"
P_SCHED = {pl: f"xgboost_model/schedules/collection/collect_cpu_{pl}.yaml" for pl in ("gpu", "npu")}
P_META = {pl: P_SCHED[pl] + ".meta.json" for pl in ("gpu", "npu")}
P_WIN = {pl: f"xgboost_model/full_collection_540/cpu_{pl}/performance_{pl}_full540.json"
         for pl in ("gpu", "npu")}

# Score policy: identical to xgboost_model/deploy_selector_xgb_suite.py:score_combo
# and analysis_common.ALPHA/BETA (= 0.3 / 1.0).
ALPHA = 0.3
BETAS = (1.0, 0.5)
GEN_MODELS = {"llama1b", "qwen2_vl"}
# Tie rule for measured-best argmax: the 1e-9 tie set convention of
# full_collection_540/scripts/analysis_common.py:ranking_metrics (see report §C).
TIE_EPS = 1e-9

W = yaml.safe_load((ROOT / P_SETS).read_text())
SETS = W["sets"]
STRAT_SETS = set(W["stratified_sets"])
STRAT_TARGET = {int(k): v for k, v in W["stratified_target_by_free"].items()}
SID_OF = {",".join(sorted(v)): k for k, v in SETS.items()}
# Report order: by set size then id (matches compare_platforms_v2.ORDER).
ORDER = ["S1", "S2", "S3", "S4", "base1", "base2", "base3", "base4", "base5",
         "S5", "S6", "S7", "S9", "S8", "S10"]


def free_models(models):
    """Models that may go on the CPU -- the free placement axis.

    Same predicate as gen_collection_schedules.free_models(): a model is free iff
    "cpu" in model_registry.allowed_devices(model).
    """
    return [m for m in models if "cpu" in reg.allowed_devices(m)]


def placement_vector(window):
    """Canonical placement vector: (model, 'cpu'|'accel') sorted by model name.

    Platform-neutral: the accelerator token (GPU vs NPU) is folded to 'accel' so the
    two platforms' placement sets are directly comparable (verification A).
    """
    out = []
    for v in window["models"].values():
        dev = _norm_exec(v["execution"])
        out.append((v["model"], "cpu" if dev == "cpu" else "accel"))
    return tuple(sorted(out))


def load_rows(platform):
    """One row per measured window, with placement vector and both-beta scores."""
    windows = json.loads((ROOT / P_WIN[platform]).read_text())
    rows = []
    for w in windows:
        models = sorted(v["model"] for v in w["models"].values())
        sid = SID_OF.get(",".join(models))
        has_gen = any(m in GEN_MODELS for m in models)
        t = w["total"]
        row = {"sid": sid, "rate": w["rate_factor"], "combo": w["combination"],
               "placement": placement_vector(w), "has_gen": has_gen,
               "n_models": len(models)}
        for beta in BETAS:
            row[f"score_b{beta}"] = score_combo(
                t["total_throughput_fps"], t["deadline_miss_rate"],
                t.get("total_tokens_per_s", 0.0) if has_gen else 0.0,
                has_gen, ALPHA, beta)
        rows.append(row)
    return rows


def scheduled_placements(platform):
    """Placements as WRITTEN INTO the schedule yaml (independent of the measurements)."""
    doc = yaml.safe_load((ROOT / P_SCHED[platform]).read_text())
    meta = json.loads((ROOT / P_META[platform]).read_text())
    by_set_rate = defaultdict(set)
    for combo, entry in doc.items():
        m = meta[combo]
        pv = tuple(sorted((e["model"],
                           "cpu" if _norm_exec(e["execution"]) == "cpu" else "accel")
                          for e in entry.values()))
        by_set_rate[(m["set"], m["rate_factor"])].add(pv)
    return by_set_rate


def enumerate_feasible(models):
    """Full constrained candidate set, regenerated from the generator's own logic.

    gen_collection_schedules.all_placements(): free models take cpu/accel; every
    non-free model (i.e. qwen2_vl) is pinned to the accelerator.
    """
    free = free_models(models)
    fixed = [m for m in models if m not in free]
    out = set()
    for combo in itertools.product(["cpu", "accel"], repeat=len(free)):
        p = dict(zip(free, combo))
        for m in fixed:
            p[m] = "accel"
        out.add(tuple(sorted(p.items())))
    return out


def best_placements(rows, beta):
    """(sid, rate) -> tie-aware set of measured-argmax placements.

    Tie-aware: ALL placements within TIE_EPS of the group max are returned, so a group
    counts as agreeing across platforms iff some placement is optimal on both.
    """
    g = defaultdict(list)
    for r in rows:
        g[(r["sid"], r["rate"])].append(r)
    out = {}
    for k, rs in g.items():
        best = max(r[f"score_b{beta}"] for r in rs)
        out[k] = {r["placement"] for r in rs
                  if abs(r[f"score_b{beta}"] - best) < TIE_EPS}
    return out


def main():
    rows = {pl: load_rows(pl) for pl in ("gpu", "npu")}
    sched = {pl: scheduled_placements(pl) for pl in ("gpu", "npu")}
    facts = {}

    # measured placement set per (platform, set) and per (platform, set, rate)
    meas_set = {pl: defaultdict(set) for pl in rows}
    meas_set_rate = {pl: defaultdict(set) for pl in rows}
    n_win = {pl: defaultdict(int) for pl in rows}
    for pl, rs in rows.items():
        for r in rs:
            meas_set[pl][r["sid"]].add(r["placement"])
            meas_set_rate[pl][(r["sid"], r["rate"])].add(r["placement"])
            n_win[pl][r["sid"]] += 1

    # ---------- B3: do the 3 rate levels share one placement set? ----------
    rate_uniform = {}
    for pl in rows:
        for sid in ORDER:
            per_rate = {rate: pset for (s, rate), pset in meas_set_rate[pl].items()
                        if s == sid}
            same = len({frozenset(p) for p in per_rate.values()}) == 1
            rate_uniform[(pl, sid)] = {"n_rates": len(per_rate), "identical": same,
                                       "sizes": sorted(len(p) for p in per_rate.values())}
    b3_all_identical = all(v["identical"] for v in rate_uniform.values())
    facts["B3_rate_levels_share_placement_set"] = b3_all_identical
    facts["B3_detail"] = {f"{pl}/{sid}": v for (pl, sid), v in rate_uniform.items()}

    # ---------- artifact 1: appendix_coverage.csv ----------
    csv_rows = []
    per_set = {}
    for sid in ORDER:
        models = SETS[sid]
        N = len(models)
        free = free_models(models)
        feasible = enumerate_feasible(models)
        n_pow2 = 2 ** N
        n_feasible = len(feasible)
        mg, mn = len(meas_set["gpu"][sid]), len(meas_set["npu"][sid])
        mode = "exhaustive" if (mg == n_feasible and mn == n_feasible) else "sampled"
        if sid in STRAT_SETS:
            strata_key = ("n_accel = #free-placed models on the accelerator "
                          f"(target {STRAT_TARGET[len(free)]} of {n_feasible})")
        else:
            strata_key = "n/a (exhaustive enumeration, no sampling)"
        csv_rows.append({
            "set_id": sid, "N": N, "n_pow2": n_pow2, "n_feasible": n_feasible,
            "n_measured_gpu": mg, "n_measured_npu": mn,
            "coverage_gpu": round(mg / n_feasible, 4),
            "coverage_npu": round(mn / n_feasible, 4),
            "mode": mode, "strata_key": strata_key,
            "source_ref": (
                f"{P_SETS}:sets.{sid} (N, models); "
                f"{P_GEN}:free_models+all_placements via {P_REGISTRY}:DEVICE_CONSTRAINTS "
                f"(n_feasible); "
                f"{P_WIN['gpu']}:[].models[].execution (n_measured_gpu); "
                f"{P_WIN['npu']}:[].models[].execution (n_measured_npu); "
                + (f"{P_SETS}:stratified_sets+stratified_target_by_free (strata_key)"
                   if sid in STRAT_SETS else f"{P_GEN}:combos_for (strata_key)")),
        })
        per_set[sid] = {"N": N, "n_free": len(free), "n_pow2": n_pow2,
                        "n_feasible": n_feasible, "mode": mode,
                        "has_vlm": "qwen2_vl" in models,
                        "n_measured": {"gpu": mg, "npu": mn},
                        "n_windows": {"gpu": n_win["gpu"][sid], "npu": n_win["npu"][sid]}}

    cols = ["set_id", "N", "n_pow2", "n_feasible", "n_measured_gpu", "n_measured_npu",
            "coverage_gpu", "coverage_npu", "mode", "strata_key", "source_ref"]

    def write_csv(path, rows_):
        lines = [",".join(cols)]
        for r in rows_:
            lines.append(",".join(
                '"' + str(r[c]).replace('"', '""') + '"' if isinstance(r[c], str)
                and ("," in r[c] or '"' in r[c]) else str(r[c]) for c in cols))
        path.write_text("\n".join(lines) + "\n")

    write_csv(OUT / "appendix_coverage.csv", csv_rows)
    facts["artifact1_rows"] = len(csv_rows)
    facts["artifact1_granularity"] = "set (15 rows)" if b3_all_identical else "(set, rate) (45 rows)"

    # If B3 failed, the appendix table must be per (set, rate) -- emit that instead.
    by_rate_path = None
    if not b3_all_identical:
        rr = []
        for sid in ORDER:
            models = SETS[sid]
            feasible = enumerate_feasible(models)
            for rate in sorted({r for (s, r) in meas_set_rate["gpu"] if s == sid}):
                base = dict(next(c for c in csv_rows if c["set_id"] == sid))
                mg = len(meas_set_rate["gpu"][(sid, rate)])
                mn = len(meas_set_rate["npu"][(sid, rate)])
                base.update({"set_id": f"{sid}@rate{rate}",
                             "n_measured_gpu": mg, "n_measured_npu": mn,
                             "coverage_gpu": round(mg / len(feasible), 4),
                             "coverage_npu": round(mn / len(feasible), 4),
                             "mode": "exhaustive" if mg == mn == len(feasible) else "sampled"})
                rr.append(base)
        by_rate_path = OUT / "appendix_coverage_by_rate.csv"
        write_csv(by_rate_path, rr)

    # ---------- A: cross-platform candidate-set identity ----------
    a_detail = {}
    for sid in ORDER:
        sym = meas_set["gpu"][sid] ^ meas_set["npu"][sid]
        a_detail[sid] = {"symdiff_size": len(sym),
                         "gpu_only": sorted(map(str, meas_set["gpu"][sid] - meas_set["npu"][sid])),
                         "npu_only": sorted(map(str, meas_set["npu"][sid] - meas_set["gpu"][sid]))}
    # also at (set, rate) granularity and against the schedule yamls
    a_rate_sym = {f"{sid}@{rate}": len(meas_set_rate["gpu"][(sid, rate)] ^
                                       meas_set_rate["npu"][(sid, rate)])
                  for (sid, rate) in meas_set_rate["gpu"]}
    a_sched_sym = {}
    for key in sched["gpu"]:
        a_sched_sym[f"{key[0]}@{key[1]}"] = len(sched["gpu"][key] ^ sched["npu"].get(key, set()))
    facts["A_symdiff_by_set"] = {k: v["symdiff_size"] for k, v in a_detail.items()}
    facts["A_all_zero"] = all(v["symdiff_size"] == 0 for v in a_detail.values())
    facts["A_symdiff_by_set_rate_all_zero"] = all(v == 0 for v in a_rate_sym.values())
    facts["A_schedule_yaml_symdiff_all_zero"] = all(v == 0 for v in a_sched_sym.values())
    facts["A_detail"] = a_detail

    # measured vs scheduled (did the runner measure exactly what was scheduled?)
    meas_vs_sched = {}
    for pl in rows:
        bad = {f"{k[0]}@{k[1]}": len(sched[pl][k] ^ meas_set_rate[pl][k])
               for k in sched[pl] if sched[pl][k] != meas_set_rate[pl].get(k, set())}
        meas_vs_sched[pl] = bad
    facts["A_measured_equals_scheduled"] = {pl: (len(b) == 0) for pl, b in meas_vs_sched.items()}
    facts["A_measured_vs_scheduled_mismatches"] = meas_vs_sched

    # ---------- B: feasible-count definition ----------
    b_cpu_violations = {pl: sum(1 for r in rows[pl]
                                if ("qwen2_vl", "cpu") in r["placement"]) for pl in rows}
    b_pow2_check = {}
    for sid in ORDER:
        d = per_set[sid]
        b_pow2_check[sid] = {
            "n_feasible": d["n_feasible"], "two_pow_N_minus_1": 2 ** (d["N"] - 1),
            "holds": d["n_feasible"] == 2 ** (d["N"] - 1),
            "has_vlm": d["has_vlm"], "n_free": d["n_free"]}
    facts["B_qwen2_vl_cpu_rows"] = b_cpu_violations
    facts["B_pow2_check"] = b_pow2_check
    facts["B_pow2_holds_for_all"] = all(v["holds"] for v in b_pow2_check.values())
    facts["B_pow2_holds_iff_has_vlm"] = all(v["holds"] == v["has_vlm"]
                                            for v in b_pow2_check.values())

    # ---------- C: divergence restricted to exhaustive groups ----------
    exh_sids = [sid for sid in ORDER if per_set[sid]["mode"] == "exhaustive"]
    c = {}
    for beta in BETAS:
        bg, bn = best_placements(rows["gpu"], beta), best_placements(rows["npu"], beta)
        groups = sorted(k for k in bg if k[0] in exh_sids)
        rec = {"n_groups": len(groups), "n_disagree": 0, "disagree_groups": [],
               "gen": {"n": 0, "d": 0}, "vision": {"n": 0, "d": 0},
               "n_groups_with_ties": 0}
        for k in groups:
            tg, tn = bg[k], bn[k]
            if len(tg) > 1 or len(tn) > 1:
                rec["n_groups_with_ties"] += 1
            agree = bool(tg & tn)          # tie-aware: intersecting optimum sets agree
            has_gen = any(m in GEN_MODELS for m in SETS[k[0]])
            bucket = rec["gen"] if has_gen else rec["vision"]
            bucket["n"] += 1
            if not agree:
                rec["n_disagree"] += 1
                bucket["d"] += 1
                rec["disagree_groups"].append({
                    "set": k[0], "rate": k[1], "has_gen": has_gen,
                    "gpu_best": sorted(map(str, tg)), "npu_best": sorted(map(str, tn))})
        c[str(beta)] = rec

    # Cross-check: the SAME rule over all 45 groups must reproduce the published
    # analysis/platform_divergence.md figures (beta=1.0 -> 29, beta=0.5 -> 30).
    c_all = {}
    for beta in BETAS:
        bg, bn = best_placements(rows["gpu"], beta), best_placements(rows["npu"], beta)
        keys = sorted(bg)
        nd = sum(1 for k in keys if not (bg[k] & bn[k]))
        c_all[str(beta)] = {"n_groups": len(keys), "n_disagree": nd}
    facts["C_all45_crosscheck"] = c_all
    facts["C_all45_matches_published"] = (c_all["1.0"]["n_disagree"] == 29
                                          and c_all["0.5"]["n_disagree"] == 30)
    facts["C"] = c
    facts["C_exhaustive_sets"] = exh_sids
    facts["C_tie_rule"] = (f"tie-aware argmax: measured-best = all placements within "
                           f"{TIE_EPS} of the group max; a group AGREES iff the two "
                           f"platforms' optimum sets intersect")

    # ---------- D: totals reconciliation ----------
    d = {}
    for pl in rows:
        d[pl] = {
            "sum_n_measured_placements": sum(per_set[s]["n_measured"][pl] for s in ORDER),
            "expected_placements": 180,
            "sum_windows": sum(per_set[s]["n_windows"][pl] for s in ORDER),
            "expected_windows": 540,
            "n_groups": len({(r["sid"], r["rate"]) for r in rows[pl]}),
            "expected_groups": 45,
            "unmapped_sid_rows": sum(1 for r in rows[pl] if r["sid"] is None),
        }
        d[pl]["placements_ok"] = d[pl]["sum_n_measured_placements"] == 180
        d[pl]["windows_ok"] = d[pl]["sum_windows"] == 540
        d[pl]["groups_ok"] = d[pl]["n_groups"] == 45
    facts["D"] = d
    facts["D_per_set"] = {s: {"n_measured": per_set[s]["n_measured"],
                              "n_windows": per_set[s]["n_windows"]} for s in ORDER}

    # duplicate check: a placement measured twice within one (set, rate)?
    dup = {}
    for pl in rows:
        cnt = defaultdict(int)
        for r in rows[pl]:
            cnt[(r["sid"], r["rate"], r["placement"])] += 1
        dup[pl] = {f"{k[0]}@{k[1]}": v for k, v in cnt.items() if v > 1}
    facts["D_duplicate_placements"] = dup

    # ---------- E: vision-only control sets ----------
    facts["E"] = {s: {"mode": per_set[s]["mode"], "n_feasible": per_set[s]["n_feasible"],
                      "n_measured": per_set[s]["n_measured"]["gpu"]}
                  for s in ("S9", "S10")}

    (OUT / "coverage_facts.json").write_text(
        json.dumps(facts, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in facts.items()
                      if not k.endswith("_detail") and k not in ("B3_detail", "A_detail")},
                     indent=2, ensure_ascii=False, default=str))
    return facts, csv_rows, per_set, by_rate_path


if __name__ == "__main__":
    main()
