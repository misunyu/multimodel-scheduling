"""cpu_placement.md 의 파생 주장 재검산 (읽기 전용)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
f = json.loads((HERE / "facts.json").read_text())
pc = pd.read_csv(HERE / "per_candidate.csv", keep_default_na=False)

problems = []
checks = {}

# per_candidate.csv 자체 정합
checks["csv_180_rows"] = len(pc) == 180
checks["csv_windows_540"] = int(pc["n_windows"].sum()) == 540
checks["csv_k_matches_cpu_models"] = all(
    r.k_cpu == (0 if r.cpu_models == "" else len(r.cpu_models.split("|")))
    for r in pc.itertuples())
checks["csv_views_split"] = all(
    r.n_views == (0 if r.cpu_models == "" else len(r.cpu_models.split("|")))
    + (0 if r.accel_models == "" else len(r.accel_models.split("|")))
    for r in pc.itertuples())
checks["csv_stratum_consistent"] = all(
    r.stratum_accel_free == r.n_free_instances - r.k_cpu for r in pc.itertuples())

# §1 파생값
kd = {r["k_cpu"]: r for r in f["k_distribution"]["gpu"]}
checks["k_ge4_is_46"] = sum(kd[k]["n_candidates"] for k in (4, 5, 6, 7)) == 46
checks["k0_one_per_set"] = (f["k0_candidates"]["n"] == 15 and
                            set(f["k0_candidates"]["by_set"].values()) == {1})
checks["window_is_3x_candidate"] = all(r["n_windows"] == 3 * r["n_candidates"]
                                       for r in f["k_distribution"]["gpu"])

# 가속기가 완전히 비는 배치 = qwen2_vl 을 포함하지 않는 세트의 all-CPU 배치
# (vision-only 4세트 + LLM 세트 S3 = 5개. "vision-only 뿐"이 아니다)
empty_accel = pc[pc["accel_models"] == ""]
checks["empty_accel_is_5"] = len(empty_accel) == 5
checks["empty_accel_sets"] = sorted(empty_accel["set_id"]) == ["S1", "S10", "S2", "S3", "S9"]
checks["empty_accel_sets_are_the_vlm_free_ones"] = sorted(empty_accel["set_id"]) == sorted(
    s for s, v in __import__("yaml").safe_load(
        (ROOT / "working_sets.yaml").read_text())["sets"].items()
    if "qwen2_vl" not in v)
checks["empty_accel_all_k_eq_f"] = all(r.k_cpu == r.n_free_instances
                                       for r in empty_accel.itertuples())

# §2 파생값
mt = {r["model"]: r for r in f["models"]["gpu"]}
inst = f["instance_totals"]["gpu"]
free_inst = inst["cand_view_instances"] - mt["qwen2_vl"]["cand_appearances"]
checks["free_instances_864"] = free_inst == 864
checks["free_cpu_ratio_0517"] = round(inst["cand_cpu_instances"] / free_inst, 3) == 0.517
checks["overall_ratio_0454"] = round(inst["cand_cpu_instances"]
                                     / inst["cand_view_instances"], 3) == 0.454
checks["vlm_zero_cpu"] = mt["qwen2_vl"]["cand_on_cpu"] == 0 and \
    mt["qwen2_vl"]["window_on_cpu"] == 0
checks["llm_65_195"] = (mt["llama1b"]["cand_on_cpu"] == 65 and
                        mt["llama1b"]["window_on_cpu"] == 195)
checks["only_resnet50_below_half"] = [m for m, r in mt.items()
                                      if r["cpu_capable"] and r["cpu_ratio_cand"] < 0.5] \
    == ["resnet50"]
# y3_valid 교차검산: VLM 포함 세트 window + S3 의 llama1b-on-accel window
sets = {r["set_id"]: r for r in f["sets"]["gpu"]}
import yaml  # noqa: E402
SETS = yaml.safe_load((ROOT / "working_sets.yaml").read_text())["sets"]
vlm_sets = [s for s, v in SETS.items() if "qwen2_vl" in v]
s3_llm_cpu, s3_total = f["llm_cpu_detail"]["llama1b"]["by_set"]["S3"]
y3_expected = (sum(sets[s]["n_candidates_measured"] for s in vlm_sets)
               + (s3_total - s3_llm_cpu)) * 3
p1 = json.loads((ROOT / "xgboost_model/full_collection_540/analysis/"
                 "groupkfold_gpu_metrics.json").read_text())
y3_stored = p1["targets"]["y3"]["n_rows_evaluated"]
checks["y3_valid_derivation_372"] = y3_expected == y3_stored == 372

# §3 파생값
ex = [r for r in f["sets"]["gpu"] if r["mode"] == "exhaustive"]
st = [r for r in f["sets"]["gpu"] if r["mode"] == "stratified"]
checks["exhaustive_frac_exactly_half"] = all(r["mean_cpu_fraction_of_free"] == 0.5
                                             for r in ex)
checks["stratified_frac_above_half"] = all(0.52 <= r["mean_cpu_fraction_of_free"] <= 0.53
                                           for r in st)
checks["each_set_exactly_one_k0"] = all(
    r["n_candidates_measured"] - r["n_with_cpu_ge1"] == 1 for r in f["sets"]["gpu"])
# 층 쿼터가 CPU-heavy 쪽에 1개 더 가는지 (낮은 a 층 >= 높은 대칭 층)
quota_bias = {}
for s, v in f["strata"].items():
    if v["mode"] != "stratified":
        continue
    fcnt = v["n_free"]
    bias = [(a, v["by_stratum"][str(a)]["measured"],
             v["by_stratum"][str(fcnt - a)]["measured"])
            for a in range(fcnt // 2 + 1)]
    quota_bias[s] = bias
checks["stratified_quota_cpu_heavy"] = all(
    lo >= hi for s in quota_bias for _, lo, hi in quota_bias[s])

# §5 파생값
ec = f["exec_cpu_feature"]
checks["exec_cpu_active_165"] = ec["candidates_active"] == 165
checks["exec_cpu_windows_495"] = ec["windows_active"] == 495
checks["exec_cpu_sum_values_0_7"] = ec["distinct_values_sum"] == list(range(8))
checks["exec_cpu_mean_23_values"] = len(ec["mean_feature_values"]) == 23
checks["free_model_ratios_in_band"] = all(
    0.483 <= r["cpu_ratio_cand"] <= 0.541 for r in f["models"]["gpu"] if r["cpu_capable"])

# 코드 근거 줄 대조
SUITE = (ROOT / "xgboost_model/deploy_selector_xgb_suite.py").read_text().splitlines()
GCS = (ROOT / "xgboost_model/gen_collection_schedules.py").read_text().splitlines()
REG = (ROOT / "model_registry.py").read_text().splitlines()
CITES = [(REG, 45, "DEVICE_CONSTRAINTS"),
         (SUITE, 587, "y3 is trained ONLY on windows"),
         (SUITE, 591, "vision-only set has no tokens")]
for lines, ln, needle in CITES:
    got = lines[ln - 1] if 0 < ln <= len(lines) else "<범위 밖>"
    if needle not in got:
        problems.append(f"줄 근거 불일치: {ln} 에 '{needle}' 없음 -> {got.strip()!r}")
if not any("itertools.product" in l for l in GCS):
    problems.append("gen_collection_schedules.py 에 itertools.product 없음")
if not any("def all_placements" in l for l in GCS):
    problems.append("gen_collection_schedules.py 에 all_placements 없음")

for k, v in checks.items():
    if not v:
        problems.append(f"검산 실패: {k}")

(HERE / "verification.json").write_text(json.dumps(
    {"checks": checks, "problems": problems, "quota_bias": quota_bias,
     "y3_expected": y3_expected, "y3_stored": y3_stored},
    indent=2, ensure_ascii=False))

for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
if problems:
    print("\n=== 문제 ===")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("\n검증 통과")
