"""Phase 2 campaign plan (v1.2 sections 2.1-2.3): schedule files, run list, execution order, run-count verification.

  python3 -B code/phase2_plan.py --write     -> schedules/<family>_x<m>.yaml, schedules/README.md, schedules/RUN_PLAN.md, schedules/run_plan.csv

Fixed rules
  family_index      1..22 in the order of gen_schedules.reconstructed_families() (N ascending, then name)
  instance order    sorted model names; combination id = bit mask (bit i = 1 -> instance i on GPU), combination_000 = all-CPU
  rate              0.1 * GPU isolated throughput (offline_profile/offline_profile.json) * multiplier, multiplier in {1, 2, 3, 4}
  rep0 block        family order  = permutation from SeedSequence([42, 0]);
                    inside family = permutation of all (multiplier, combination) from SeedSequence([42, family_index])
  rep r in {1, 2}   only 3x, N >= 3; after the whole rep0 block; family order from SeedSequence([42, 0, r]),
                    candidate order from SeedSequence([42, family_index, r])
  run_key           (family, multiplier, combination, rep);  result file results/<family>/x<m>/run_<combination>_rep<r>.json
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import yaml

import gen_schedules as G

ROOT = Path(__file__).resolve().parent.parent
SCHED = ROOT / "schedules"
RESULTS = ROOT / "results"
BETA = 0.1
MULTS = (1, 2, 3, 4)
REPEAT_MULT = 3
REPEAT_MIN_N = 3
EXTRA_REPS = (1, 2)


def families():
    """[(family_index, key, models)]"""
    return [(i + 1, k, v[0]) for i, (k, v) in enumerate(G.reconstructed_families().items())]


def perm(seed, n):
    return np.random.default_rng(np.random.SeedSequence(seed)).permutation(n)


def out_path(key, mult, comb, rep):
    return RESULTS / key / f"x{mult}" / f"run_{comb}_rep{rep}.json"


def jobs():
    """Ordered list of dicts; exec_order is the planned position (stable across resumes)."""
    fams = families()
    out = []
    order0 = perm([42, 0], len(fams))
    for fpos in order0:
        fi, key, models = fams[fpos]
        cand = [(m, mask) for m in MULTS for mask in range(2 ** len(models))]
        for j in perm([42, fi], len(cand)):
            m, mask = cand[j]
            out.append({"block": "rep0", "family_index": fi, "family": key, "N": len(models), "mult": m, "combination": f"combination_{mask:03d}", "rep": 0})
    rep_fams = [f for f in fams if len(f[2]) >= REPEAT_MIN_N]
    for r in EXTRA_REPS:
        for fpos in perm([42, 0, r], len(rep_fams)):
            fi, key, models = rep_fams[fpos]
            for mask in perm([42, fi, r], 2 ** len(models)):
                out.append({"block": f"rep{r}", "family_index": fi, "family": key, "N": len(models), "mult": REPEAT_MULT,
                            "combination": f"combination_{int(mask):03d}", "rep": r})
    for n, j in enumerate(out):
        j["exec_order"] = n
        j["schedule"] = str(SCHED / f"{j['family']}_x{j['mult']}.yaml")
        j["out"] = str(out_path(j["family"], j["mult"], j["combination"], j["rep"]))
    return out


def est_sec(n):
    return 15.7 + 0.47 * n


def phase1b_run_times():
    """N -> list of run seconds from the Phase 1b pilot logs (open-loop runs only)."""
    by = {}
    pat = re.compile(r"\] (\S+)__combination_\d+__rep\d+ rc=0 status=ok ([\d.]+)s")
    for log in ("phase1b_pilot.log", "repeat_5_4.log"):
        p = ROOT / "logs" / log
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            m = pat.search(line)
            if not m:
                continue
            stem = m.group(1)
            n = int(stem[1]) if re.match(r"n\d_", stem) else 1 if stem.startswith("single_") else None
            if n:
                by.setdefault(n, []).append(float(m.group(2)))
    return by


def write():
    SCHED.mkdir(exist_ok=True)
    fams = families()
    prof = G.gpu_throughput()
    for fi, key, models in fams:
        for m in MULTS:
            y = G.build(key, models, BETA, m, profile=prof)
            y["meta"]["family_index"] = fi
            y["meta"]["candidates"] = 2 ** len(models)
            (SCHED / f"{key}_x{m}.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
    J = jobs()
    with open(SCHED / "run_plan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exec_order", "block", "family_index", "family", "N", "mult", "combination", "rep"], extrasaction="ignore")
        w.writeheader()
        w.writerows(J)
    # ---- verification
    placements = sum(2 ** len(m) for _, _, m in fams)
    base = [j for j in J if j["block"] == "rep0"]
    rep = [j for j in J if j["block"] != "rep0"]
    rep_fams = [f for f in fams if len(f[2]) >= REPEAT_MIN_N]
    rep_cands = sum(2 ** len(m) for _, _, m in rep_fams)
    keys = {(j["family"], j["mult"], j["combination"], j["rep"]) for j in J}
    checks = {
        "families == 22": len(fams) == 22,
        "unique placements (sum of 2^N) == 504": placements == 504,
        "base runs == 504 x 4 == 2,016": len(base) == 2016 == placements * 4,
        "3x N>=3 workloads == 12": len(rep_fams) == 12,
        "candidates of the 12 repeated workloads == 480": rep_cands == 480,
        "repeat runs == 480 x 2 == 960": len(rep) == 960 == rep_cands * 2,
        "total == 2,976": len(J) == 2976,
        "run_key unique": len(keys) == len(J),
        "every base (family, mult) holds exactly 2^N combinations": all(
            len({j["combination"] for j in base if j["family"] == k and j["mult"] == m}) == 2 ** len(ms) for _, k, ms in fams for m in MULTS),
        "repeat blocks contain only 3x, N>=3": all(j["mult"] == 3 and j["N"] >= 3 for j in rep),
        "all schedule files exist": all(Path(j["schedule"]).exists() for j in J),
        "max N == 7": max(len(m) for _, _, m in fams) == 7,
    }
    t_base = sum(est_sec(j["N"]) for j in base)
    t_rep = sum(est_sec(j["N"]) for j in rep)
    p1b = phase1b_run_times()
    L = ["# Campaign run plan and run-count verification (Phase 2 v1.2 section 2.2)", "",
         "Generated by `code/phase2_plan.py --write`. Run list with planned order: `schedules/run_plan.csv`.", "",
         "## 1. Verification", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {'PASS' if v else '**FAIL**'} |" for k, v in checks.items()]
    L += ["", f"**Overall: {'PASS' if all(checks.values()) else 'FAIL'}**", "",
          "\"504 placements\" = the sum over the 22 families of the feasible placements 2^N of each family; it does not depend on the rate. "
          "Each placement is measured at 4 rates (2,016 runs); the 480 placements of the 12 N>=3 families get 2 more runs at 3x (960 runs).", "",
          "## 2. Families", "", "| idx | family | N | 2^N | runs rep0 (4 rates) | extra runs at 3x (rep1+rep2) |", "|---|---|---|---|---|---|"]
    for fi, key, models in fams:
        n = len(models)
        L.append(f"| {fi} | {key} | {n} | {2 ** n} | {2 ** n * 4} | {2 ** n * 2 if n >= 3 else 0} |")
    L.append(f"| | **total** | | **{placements}** | **{len(base)}** | **{len(rep)}** |")
    L += ["", "## 3. Runs per family x rate", "", "| family | x1 | x2 | x3 (rep0+rep1+rep2) | x4 |", "|---|---|---|---|---|"]
    for fi, key, models in fams:
        c = {m: sum(1 for j in J if j["family"] == key and j["mult"] == m) for m in MULTS}
        L.append(f"| {key} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |")
    L += ["", "## 4. Expected time", "", "Per run `15.7 + 0.47 N` s (Phase 1b measurement).", "",
          "| block | runs | hours |", "|---|---|---|",
          f"| rep0 (1,2,3,4x, all families) | {len(base)} | {t_base / 3600:.2f} |",
          f"| rep1 + rep2 (3x, N>=3) | {len(rep)} | {t_rep / 3600:.2f} |",
          f"| total | {len(J)} | {(t_base + t_rep) / 3600:.2f} |", "",
          "Phase 1b actual run time (wall time of one executor subprocess, from `logs/phase1b_pilot.log` and `logs/repeat_5_4.log`):", "",
          "| N | runs | p50 s | p95 s | formula s |", "|---|---|---|---|---|"]
    for n in sorted(p1b):
        v = np.array(p1b[n])
        L.append(f"| {n} | {len(v)} | {np.percentile(v, 50):.1f} | {np.percentile(v, 95):.1f} | {est_sec(n):.1f} |")
    L += ["", "## 5. Order", "",
          "Family order of rep0 (family_index): " + " ".join(str(families()[p][0]) for p in perm([42, 0], len(fams))), "",
          "Blocks: rep0 (all) -> rep1 (3x, N>=3) -> rep2 (3x, N>=3). Rules: see the docstring of `code/phase2_plan.py` and `schedules/README.md`.", ""]
    (SCHED / "RUN_PLAN.md").write_text("\n".join(L))
    (SCHED / "README.md").write_text("\n".join([
        "# schedules/ (Phase 2 campaign)", "",
        "- File: `<family>_x<m>.yaml`, m in 1..4. Family key = model names sorted and joined by `+`.",
        "- family_index: 1..22 in the order N ascending, then key (see RUN_PLAN.md section 2).",
        "- Instance order = sorted model names. Instance name = model name (one instance per model).",
        "- Combination id = bit mask, `combination_<mask:03d>`; bit i = 1 puts instance i on GPU. `combination_000` = all-CPU, the largest = all-GPU.",
        "  `placement` is the same mask as a string, instance 0 first (`G`/`C`).",
        "- Rate of an instance: `rate = infps = 0.1 * base_rate * multiplier`, `base_rate` = GPU isolated throughput of the model in",
        "  `offline_profile/offline_profile.json`. The rate does not depend on the placement.",
        "- Layout: `meta` + `combinations.<id>.instances.<name>.{model, execution, base_rate, beta, multiplier, rate, infps}` (executor layout).",
        "  The dataset schedule CSVs (`dataset/*_schedules_x3.csv`) carry the same numbers in the layout of the existing parser",
        "  (`combination_xxx -> instance -> model, execution, infps`); see `dataset/README.md`.",
        "- All 2^N placements of every family are listed; nothing is sampled.", ""]))
    print("\n".join(L[4:20]))
    print(f"total {len(J)} runs, estimated {(t_base + t_rep) / 3600:.2f} h")
    return all(checks.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        sys.exit(0 if write() else 1)
    J = jobs()
    print(len(J), J[0], J[-1])
