"""Phase 4b run plan: rep1 and rep2 of the Phase 4 candidates (12 families x {1.5, 2.5, 3.5} x 2^N = 1,440 per repeat).
New file; it imports the Phase 4 plan (intermediate/code_copy/intermediate_plan.py, unchanged) for the family list, schedule paths and
the run-time formula. Changes relative to the Phase 4 plan are listed in ../CHANGES_REPEATS.md.

  python3 -B revision_profiling/intermediate/code_copy_repeats/repeats_plan.py --write
      -> intermediate/RUN_PLAN_REPEATS.md, intermediate/logs/run_plan_repeats.csv, intermediate/logs/schedule_sha256.json

Fixed rules
  candidates        exactly the Phase 4 rep0 run list (intermediate/schedules/run_plan.csv), same schedule YAMLs (not regenerated)
  blocks            rep1 = one block with all 1,440 runs, then rep2 = a separate block with all 1,440 runs
  order             in block r: family order = permutation from SeedSequence([43, r]);
                    inside a family all (rate, combination) pairs in the permutation from SeedSequence([43, r, family_index]);
                    family_index = the fixed Phase 2 index of the Phase 4 RUN_PLAN.md family table (11..22)
  run_key           (family, rate, combination, rep);  result file intermediate/results_rep<r>/<family>/x<m>/run_<combination>_rep<r>.json
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
IROOT = HERE.parent
ROOT = IROOT.parent
sys.path.insert(0, str(IROOT / "code_copy"))
import intermediate_plan as P4  # noqa: E402  (Phase 4 plan, unchanged)

SEED = 43
REPS = (1, 2)
LOGS = IROOT / "logs"
SCHED = P4.SCHED                       # intermediate/schedules (Phase 4 files, read only)


def results_dir(rep):
    return IROOT / f"results_rep{rep}"


def out_path(key, mult, comb, rep):
    return results_dir(rep) / key / P4.tag(mult) / f"run_{comb}_rep{rep}.json"


def jobs():
    fams = P4.families()
    out = []
    for r in REPS:
        for fpos in P4.perm([SEED, r], len(fams)):
            fi, key, models = fams[fpos]
            cand = [(m, mask) for m in P4.MULTS for mask in range(2 ** len(models))]
            for j in P4.perm([SEED, r, fi], len(cand)):
                m, mask = cand[j]
                out.append({"block": f"rep{r}", "family_index": fi, "family": key, "N": len(models), "mult": m,
                            "combination": f"combination_{mask:03d}", "rep": r})
    for n, j in enumerate(out):
        j["exec_order"] = n
        j["schedule"] = str(SCHED / f"{j['family']}_{P4.tag(j['mult'])}.yaml")
        j["out"] = str(out_path(j["family"], j["mult"], j["combination"], j["rep"]))
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def schedule_files():
    return sorted(SCHED.glob("*.yaml"))


def plan_family_index():
    """family -> Phase 2 index from the table of the Phase 4 RUN_PLAN.md."""
    out = {}
    for line in (IROOT / "RUN_PLAN.md").read_text().splitlines():
        m = re.match(r"\| (\d+) \| (\S+) \| (\d) \| (\d+) \| (\d+) \|", line)
        if m:
            out[m.group(2)] = int(m.group(1))
    return out


def write():
    J = jobs()
    rep0 = pd.read_csv(SCHED / "run_plan.csv")
    sched = {str(p.relative_to(IROOT)): sha(p) for p in schedule_files()}
    used = {str(Path(j["schedule"]).relative_to(IROOT)) for j in J}
    p4_seconds = pd.read_csv(IROOT / "results" / "integrity" / "run_seconds_by_N.csv").set_index("N")
    keys = [(j["family"], float(j["mult"]), j["combination"], j["rep"]) for j in J]
    cand_rep0 = set(zip(rep0["family"], rep0["mult"].astype(float), rep0["combination"]))
    fi_plan = plan_family_index()
    fams = P4.families()
    checks = {
        "families = the 12 families of the Phase 4 RUN_PLAN.md (same Phase 2 index)": {k: fi for fi, k, _ in fams} == fi_plan and len(fi_plan) == 12,
        "family_index of every run = the RUN_PLAN.md index (not renumbered by the shuffled order)": all(fi_plan[j["family"]] == j["family_index"] for j in J),
        "runs == 2 x 1,440 == 2,880": len(J) == 2880,
        "run_key (family, rate, combination, rep) unique": len(set(keys)) == len(keys),
        "each block holds exactly the Phase 4 rep0 candidate set (1,440 = 36 workloads x 2^N)": all(
            {(f, m, c) for f, m, c, r in keys if r == rep} == cand_rep0 and sum(1 for k in keys if k[3] == rep) == 1440 for rep in REPS),
        "36 workloads (12 families x 3 rates)": len({(f, m) for f, m, _ in cand_rep0}) == 36,
        "blocks run in sequence: every rep1 run before every rep2 run": max(j["exec_order"] for j in J if j["rep"] == 1) < min(j["exec_order"] for j in J if j["rep"] == 2),
        "schedules = the Phase 4 YAML files (36, read only, not regenerated)": used == set(sched) and len(sched) == 36,
        "result paths are new (intermediate/results_rep1, results_rep2), no file exists yet": all(
            str(Path(j["out"]).relative_to(IROOT)).startswith(f"results_rep{j['rep']}/") and not Path(j["out"]).exists() for j in J),
        "order differs from rep0 and between rep1 and rep2": (
            [(j["family"], j["mult"], j["combination"]) for j in J if j["rep"] == 1]
            != [(j["family"], j["mult"], j["combination"]) for j in J if j["rep"] == 2]
            != list(zip(rep0["family"], rep0["mult"], rep0["combination"]))),
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "schedule_sha256.json").write_text(json.dumps({"taken_at": datetime.now().isoformat(sep=" ", timespec="seconds"), "sha256": sched}, indent=1))
    with open(LOGS / "run_plan_repeats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exec_order", "block", "family_index", "family", "N", "mult", "combination", "rep"], extrasaction="ignore")
        w.writeheader()
        w.writerows(J)
    # position of the three measurements of every candidate (rep0 from the Phase 4 plan, positions in hours of planned time)
    pos0 = {(f, float(m), c): i for i, (f, m, c) in enumerate(zip(rep0["family"], rep0["mult"], rep0["combination"]))}
    t = {r: sum(P4.est_sec(j["N"]) for j in J if j["rep"] == r) for r in REPS}
    t_p4 = {r: sum(float(p4_seconds.loc[j["N"], "p50"]) for j in J if j["rep"] == r) for r in REPS}
    L = ["# Phase 4b run plan — rep1, rep2 of the intermediate rates 1.5×, 2.5×, 3.5×", "",
         "Generated by `intermediate/code_copy_repeats/repeats_plan.py --write`. Run list: `logs/run_plan_repeats.csv`; schedule hashes: `logs/schedule_sha256.json`.", "",
         "## 1. Verification", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {'PASS' if v else '**FAIL**'} |" for k, v in checks.items()]
    L += ["", f"**Overall: {'PASS' if all(checks.values()) else 'FAIL'}**", "",
          "## 2. Families (Phase 4 RUN_PLAN.md, fixed index)", "", "| Phase 2 idx | family | N | 2^N | runs per repeat |", "|---|---|---|---|---|"]
    for fi, key, models in fams:
        L.append(f"| {fi} | {key} | {len(models)} | {2 ** len(models)} | {2 ** len(models) * 3} |")
    L += ["", "## 3. Order", "",
          "- rep1 block: family order from `SeedSequence([43, 1])`: " + " ".join(str(fams[p][0]) for p in P4.perm([SEED, 1], len(fams))) + ".",
          "- rep2 block: family order from `SeedSequence([43, 2])`: " + " ".join(str(fams[p][0]) for p in P4.perm([SEED, 2], len(fams))) + ".",
          "- Inside a family: all 3 × 2^N (rate, combination) pairs in the permutation from `SeedSequence([43, r, family_index])`.",
          "- rep0 (Phase 4) used `SeedSequence([43, 0])` and `SeedSequence([43, family_index])`; the three measurements of a candidate are in "
          "three different blocks (rep0 on 2026-10-19, rep1 and rep2 one after the other in this campaign).", "",
          "## 4. Expected time", "", "Per run `15.7 + 0.47 N` s (formula), and with the Phase 4 measured median per N (`results/integrity/run_seconds_by_N.csv`).", "",
          "| block | runs | hours (formula) | hours (Phase 4 p50) |", "|---|---|---|---|"]
    for r in REPS:
        L.append(f"| rep{r} | {sum(1 for j in J if j['rep'] == r)} | {t[r] / 3600:.2f} | {t_p4[r] / 3600:.2f} |")
    L += [f"| **total** | **{len(J)}** | **{sum(t.values()) / 3600:.2f}** | **{sum(t_p4.values()) / 3600:.2f}** |", "",
          "Phase 4 rep0 measured 7.43 h for 1,440 runs (inter-run overhead included).", "",
          "| N | formula s | Phase 4 p50 s |", "|---|---|---|"]
    for n in sorted(Counter(j["N"] for j in J)):
        L.append(f"| {n} | {P4.est_sec(n):.2f} | {float(p4_seconds.loc[n, 'p50']):.2f} |")
    L += ["", "## 5. Runtime", "",
          "- The original `revision_profiling/code/executor.py` and its modules (sha256 asserted equal to Phase 4 `logs/runtime_sha256.json` at the start of each block, "
          "recorded in `logs/runtime_sha256_rep{1,2}.json`), β = 0.1, Δ = 5 s warm-up + 10 s window, input queue 2 (drop-newest), no nice, "
          "offline profile of Phase 2 (sha256 compared).", ""]
    (IROOT / "RUN_PLAN_REPEATS.md").write_text("\n".join(L))
    print("\n".join(L))
    return all(checks.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        sys.exit(0 if write() else 1)
    J = jobs()
    print(len(J), J[0], J[-1])
