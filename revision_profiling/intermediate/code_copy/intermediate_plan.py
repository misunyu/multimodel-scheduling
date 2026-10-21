"""Phase 4 campaign plan (intermediate rates 1.5x, 2.5x, 3.5x). Copy of code/phase2_plan.py with the changes listed in ../CHANGES.md.

  python3 -B revision_profiling/intermediate/code_copy/intermediate_plan.py --write
      -> intermediate/schedules/<family>_x<m>.yaml, intermediate/schedules/run_plan.csv, intermediate/RUN_PLAN.md

Fixed rules
  families          the 12 families with N >= 3 of Phase 2 (family_index = Phase 2 index, from gen_schedules.reconstructed_families())
  instance order    sorted model names; combination id = bit mask (bit i = 1 -> instance i on GPU), combination_000 = all-CPU
  rate              0.1 * GPU isolated throughput (offline_profile/offline_profile.json, the Phase 2 file, not re-measured) * multiplier,
                    multiplier in {1.5, 2.5, 3.5}  (code/gen_schedules.build() unchanged; it already takes a float multiplier)
  order             family order = permutation from SeedSequence([43, 0]);
                    inside family = permutation of all (multiplier, combination) from SeedSequence([43, family_index])
  repeats           rep0 only
  run_key           (family, multiplier, combination, rep0);  result file intermediate/results/<family>/x<m>/run_<combination>_rep0.json
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
IROOT = HERE.parent                        # revision_profiling/intermediate
ROOT = IROOT.parent                        # revision_profiling
ORIG_CODE = ROOT / "code"
sys.path.insert(0, str(ORIG_CODE))
import gen_schedules as G  # noqa: E402  (original, unchanged)

assert Path(G.__file__).resolve().parent == ORIG_CODE
SCHED = IROOT / "schedules"
RESULTS = IROOT / "results"
BETA = 0.1
MULTS = (1.5, 2.5, 3.5)
MIN_N = 3
SEED = 43
EXPECTED_FAMILIES = {   # Phase 2 3x evaluation set (REPORT_PHASE2.md section 3, out_reprofiled/per_workload_x3.csv / fig3_data.csv)
    "gpt2+mnasnet+resnet50": 3, "gpt2+resnext50+vgg19": 3, "gpt2+resnext50+yolov4": 3, "mnasnet+resnet50+resnext50": 3,
    "mnasnet+resnext50+yolov4": 3, "resnet50+vgg19+yolov4": 3, "gpt2+mnasnet+resnet50+resnext50": 4,
    "gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12": 5, "gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12": 6,
    "mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19": 6,
    "gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19": 7,
    "mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4": 7}


def tag(m):
    return f"x{m:g}"


def families():
    """[(family_index (Phase 2), key, models)] for N >= 3."""
    allf = [(i + 1, k, v[0]) for i, (k, v) in enumerate(G.reconstructed_families().items())]
    assert len(allf) == 22
    return [f for f in allf if len(f[2]) >= MIN_N]


def perm(seed, n):
    return np.random.default_rng(np.random.SeedSequence(seed)).permutation(n)


def out_path(key, mult, comb, rep):
    return RESULTS / key / tag(mult) / f"run_{comb}_rep{rep}.json"


def jobs():
    fams = families()
    out = []
    for fpos in perm([SEED, 0], len(fams)):
        fi, key, models = fams[fpos]
        cand = [(m, mask) for m in MULTS for mask in range(2 ** len(models))]
        for j in perm([SEED, fi], len(cand)):
            m, mask = cand[j]
            out.append({"block": "rep0", "family_index": fi, "family": key, "N": len(models), "mult": m, "combination": f"combination_{mask:03d}", "rep": 0})
    for n, j in enumerate(out):
        j["exec_order"] = n
        j["schedule"] = str(SCHED / f"{j['family']}_{tag(j['mult'])}.yaml")
        j["out"] = str(out_path(j["family"], j["mult"], j["combination"], j["rep"]))
    return out


def est_sec(n):
    return 15.7 + 0.47 * n


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write():
    SCHED.mkdir(parents=True, exist_ok=True)
    fams = families()
    prof = G.gpu_throughput()
    rate_rows = []
    for fi, key, models in fams:
        x3 = yaml.safe_load((ROOT / "schedules" / f"{key}_x3.yaml").read_text())     # Phase 2 schedule (read only)
        for m in MULTS:
            y = G.build(key, models, BETA, m, profile=prof)
            y["meta"]["family_index"] = fi
            y["meta"]["candidates"] = 2 ** len(models)
            (SCHED / f"{key}_{tag(m)}.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
            # rate assert: every instance of every combination = 0.1 * isolated * m, and = Phase 2 x3 rate * m / 3
            y2 = yaml.safe_load((SCHED / f"{key}_{tag(m)}.yaml").read_text())
            assert len(y2["combinations"]) == 2 ** len(models)
            for comb, c in y2["combinations"].items():
                for name, inst in c["instances"].items():
                    exp = BETA * prof[inst["model"]] * m
                    r3 = x3["combinations"][comb]["instances"][name]
                    assert abs(inst["infps"] - exp) <= 1e-12 * exp and inst["infps"] == inst["rate"], (key, m, comb, name)
                    assert inst["base_rate"] == r3["base_rate"] == prof[inst["model"]], (key, comb, name)
                    assert abs(inst["infps"] - r3["infps"] * m / 3) <= 1e-9 * exp, (key, m, comb, name)
                    assert inst["execution"] == r3["execution"]
            for name in sorted(models):
                inst = y2["combinations"]["combination_000"]["instances"][name]
                rate_rows.append({"family": key, "rate": tag(m), "model": name, "isolated_gpu_fps": prof[name], "rate_req_s": inst["infps"],
                                  "expected_0.1_x_iso_x_m": BETA * prof[name] * m, "phase2_x3_rate": x3["combinations"]["combination_000"]["instances"][name]["infps"],
                                  "requests_in_10s_window": inst["infps"] * 10})
    J = jobs()
    with open(SCHED / "run_plan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exec_order", "block", "family_index", "family", "N", "mult", "combination", "rep"], extrasaction="ignore")
        w.writeheader()
        w.writerows(J)
    with open(SCHED / "rate_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rate_rows[0]))
        w.writeheader()
        w.writerows(rate_rows)
    placements = sum(2 ** len(m) for _, _, m in fams)
    keys = {(j["family"], j["mult"], j["combination"], j["rep"]) for j in J}
    checks = {
        "families: the 12 N>=3 families of the Phase 2 3x set (names and N)": {k: len(m) for _, k, m in fams} == EXPECTED_FAMILIES,
        "candidates (sum of 2^N) == 480 (= Phase 2 '3x N>=3 candidates')": placements == 480,
        "runs == 480 x 3 == 1,440": len(J) == 1440 == placements * 3,
        "run_key unique": len(keys) == len(J),
        "every (family, rate) holds exactly 2^N combinations": all(
            len({j["combination"] for j in J if j["family"] == k and j["mult"] == m}) == 2 ** len(ms) for _, k, ms in fams for m in MULTS),
        "rep0 only": all(j["rep"] == 0 for j in J),
        "rate = 0.1 x isolated GPU throughput x m for every instance of every combination (36 workloads)": True,   # asserted above
        "all schedule files exist": all(Path(j["schedule"]).exists() for j in J),
        "offline profile = the Phase 2 file (same path, unchanged vs git HEAD, base_rate = Phase 2 3x schedules)": (
            G.PROFILE == ROOT / "offline_profile" / "offline_profile.json"
            and __import__("subprocess").run(["git", "-C", str(ROOT.parent), "diff", "--quiet", "HEAD", "--", str(G.PROFILE)]).returncode == 0
            and __import__("subprocess").run(["git", "-C", str(ROOT.parent), "ls-files", "--error-unmatch", str(G.PROFILE)], capture_output=True).returncode == 0),
    }
    t = {m: sum(est_sec(j["N"]) for j in J if j["mult"] == m) for m in MULTS}
    L = ["# Phase 4 run plan — intermediate rates 1.5×, 2.5×, 3.5× (CPU–GPU)", "",
         "Generated by `intermediate/code_copy/intermediate_plan.py --write`. Run list with planned order: `schedules/run_plan.csv`; "
         "per-instance rates: `schedules/rate_check.csv`.", "",
         "## 1. Verification", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {'PASS' if v else '**FAIL**'} |" for k, v in checks.items()]
    L += ["", f"**Overall: {'PASS' if all(checks.values()) else 'FAIL'}**", "",
          "## 2. Families (Phase 2 3× N≥3 evaluation set)", "",
          "대조: `REPORT_PHASE2.md` §3 표(12 workloads), `revision_exp/out_reprofiled/per_workload_x3.csv`, `fig3_data.csv`(N≥3 12, ≥4 6, ≥5 5, ≥6 4, ≥7 2).", "",
          "| Phase 2 idx | family | N | 2^N | runs (3 rates) |", "|---|---|---|---|---|"]
    for fi, key, models in fams:
        L.append(f"| {fi} | {key} | {len(models)} | {2 ** len(models)} | {2 ** len(models) * 3} |")
    L.append(f"| | **total** | | **{placements}** | **{len(J)}** |")
    from collections import Counter
    cN = Counter(len(m) for _, _, m in fams)
    L += ["", f"N 분포: {dict(sorted(cN.items()))} → N≥3 {len(fams)}, ≥4 {sum(v for k, v in cN.items() if k >= 4)}, ≥5 {sum(v for k, v in cN.items() if k >= 5)}, "
          f"≥6 {sum(v for k, v in cN.items() if k >= 6)}, ≥7 {sum(v for k, v in cN.items() if k >= 7)} (fig3_data.csv와 같음).", "",
          "## 3. Rate", "",
          "`r_m = 0.1 × GPU isolated throughput_m × m`, m ∈ {1.5, 2.5, 3.5}. Isolated throughput은 Phase 2의 `offline_profile/offline_profile.json`(재측정 없음). "
          "`code/gen_schedules.py`의 `build()`가 이미 실수 배수를 받으므로 원본을 그대로 import해 썼다(사본 수정 없음). "
          "12 families × 3 rates의 모든 combination·instance에서 rate = 0.1 × isolated × m, = Phase 2 3× rate × m/3, base_rate = profile 값임을 assert했다.", "",
          "| model | isolated GPU fps | 1.5× req/s | 2.5× | 3× (Phase 2) | 3.5× |", "|---|---|---|---|---|---|"]
    seen = {}
    for r in rate_rows:
        seen.setdefault(r["model"], {"iso": r["isolated_gpu_fps"], "x3": r["phase2_x3_rate"]})[r["rate"]] = r["rate_req_s"]
    for mdl, v in sorted(seen.items()):
        L.append(f"| {mdl} | {v['iso']:.2f} | {v['x1.5']:.3f} | {v['x2.5']:.3f} | {v['x3']:.3f} | {v['x3.5']:.3f} |")
    L += ["", "## 4. Runs and expected time", "", "Per run `15.7 + 0.47 N` s (Phase 1b measurement, same formula as Phase 2).", "",
          "| rate | runs | hours |", "|---|---|---|"]
    for m in MULTS:
        L.append(f"| {tag(m)} | {sum(1 for j in J if j['mult'] == m)} | {t[m] / 3600:.2f} |")
    L += [f"| **total** | **{len(J)}** | **{sum(t.values()) / 3600:.2f}** |", "",
          "## 5. Order and runtime", "",
          "- Family order: permutation from `SeedSequence([43, 0])` over the 12 families; inside a family all (rate, combination) pairs in the "
          "permutation from `SeedSequence([43, family_index])` (Phase 2 rule with seed 42 → 43). Family order (Phase 2 idx): "
          + " ".join(str(fams[p][0]) for p in perm([SEED, 0], len(fams))) + ".",
          "- rep0 only. `run_key = (family, rate, combination, rep0)`; resumable (a valid result is skipped).",
          "- Runtime: the original `revision_profiling/code/executor.py` (unchanged; it aborts if a repository module is loaded from outside "
          "`revision_profiling/code/`), Δ = warm-up 5 s + window 10 s, input queue 2 with drop-newest, process feeder, thread count, no nice — "
          "all the executor defaults used in Phase 2.", ""]
    (IROOT / "RUN_PLAN.md").write_text("\n".join(L))
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
