"""작업 13 검증: (1) oracle ratio 집계식, (2) 실행 설정 근거 줄, (3) 후보 열거 정책.

읽기 전용. 저장된 산출물·소스만 읽는다. 학습·수집·프로파일링 실행 없음.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

problems = []
checks = {}


def lines(rel):
    return (ROOT / rel).read_text().splitlines()


GB = lines("xgboost_model/full_collection_540/scripts/greedy_baseline.py")
GCS = lines("xgboost_model/gen_collection_schedules.py")
GS = lines("generate_schedules.py")
BDF = lines("best_deploy_finder_executor.py")
LE = lines("runtime/llm_engine.py")
MP = lines("model_processors.py")
CR = lines("xgboost_model/full_collection_540/scripts/collect_run.py")
REG = lines("model_registry.py")
UV = lines("unified_viewer.py")
PCF = lines("PRE_COLLECTION_FINAL.md")
NPU_DOC = lines("NPU_POSTPROCESS_UNIFICATION.md")
CONF = lines("docs/llama1b_core_confound_report.md")
SU = lines("xgboost_model/deploy_selector_xgb_suite.py")

CITES = [
    # (1) oracle ratio
    (GB, 93, 'oracle_n, oracle_r = g["s_norm"].max(), g["s_raw"].max()'),
    (GB, 94, "if len(match):"),
    (GB, 107, 'st["rn"].append(row["s_norm"] / oracle_n'),
    (GB, 108, 'st["rr"].append(row["s_raw"] / oracle_r'),
    (GB, 115, '"unc"].append'),
    (GB, 134, 'np.nanmean(st[\'rn\'])'),
    (GB, 92, 'match = g[g["accel_set"] == accel_set]'),
    (GB, 95, "row = match.iloc[0]"),
    (GB, 96, 'top1 = int(abs(row["s_norm"] - oracle_n) < 1e-9)'),
    (GB, 69, "s_norm = ac.scores_from"),
    (GB, 71, "s_raw = ac.scores_from"),
    # (2) 실행 설정
    (CR, 25, "duration = int(sys.argv[4]) if len(sys.argv) > 4 else 180"),
    (CR, 67, '"WARMUP_SECONDS": "20"'),
    (UV, 68, 'DEMO_VIDEO = os.environ.get("DEMO_VIDEO", "stockholm_1280x720.mp4")'),
    (UV, 90, 'WARMUP_SECONDS = float(os.environ.get("WARMUP_SECONDS", "20"))'),
    (LE, 72, 'dtype = torch.float16 if self.device == "gpu" else torch.float32'),
    (LE, 135, "do_sample=False"),
    (LE, 44, 'NPU_LLM_CORES = {"core_mode": "single", "target_cores": ["0:0"]}'),
    (LE, 26, 'DEFAULT_PROMPT = "Describe what is happening in one short sentence."'),
    (MP, 82, "CUDAExecutionProvider"),
    (MP, 139, "canvas = np.full((size, size, 3), 114"),
    (MP, 142, "arr = canvas[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0"),
    (MP, 143, "arr[None, ...]"),
    (MP, 357, "arr = ((rgb - mean) / std).transpose(2, 0, 1)[None, ...]"),
    (MP, 529, "max_new_tokens=64"),
    (MP, 543, "max_new_tokens=32"),
    (MP, 499, "prompt = prompts[idx % len(prompts)]"),
    (MP, 280, 'build_vision_npu(model_name, infer_mode="global8")'),
    (REG, 68, '"input_size": 640'),
    (REG, 85, '"input_size": 224'),
    (REG, 101, "NPU: Mobilint pre-quantized W8 .mxq"),
    # (3) 후보 열거
    (GCS, 45, "def all_placements"),
    (GCS, 49, 'itertools.product(["cpu", "accel"], repeat=len(free))'),
    (GCS, 91, "def combos_for"),
    (GS, 58, "def enumerate_placements"),
    (GS, 69, "allowed = [d for d in reg.allowed_devices(m) if d in platform_devices]"),
    (GS, 70, 'choices.append(allowed or ["cpu"])'),
    (BDF, 675, "import generate_schedules as gs"),
    (BDF, 696, "gs.enumerate_placements(models, platform_devices)"),
    (SU, 155, "def _iter_combos_from_schedule"),
    (MP, 298, "model.preprocess(_to_rgb(frame))"),
    (MP, 497, "prompt = prompt_override or DEFAULT_PROMPT"),
    (MP, 159, "def _limit_cpu_threads"),
    (REG, 95, '"input_size": 224'),
    (REG, 102, '"gpu_hf": "unsloth/Llama-3.2-1B-Instruct"'),
    (REG, 104, '"revision": "W8"'),
    (REG, 110, '"gpu_hf": "Qwen/Qwen2-VL-2B-Instruct"'),
    (REG, 112, '"revision": "main"'),
    (LE, 27, "LLM_PROMPTS = ["),
    (GCS, 42, 'return [m for m in models if "cpu" in reg.allowed_devices(m)]'),
    (CR, 32, "random.seed(20260715)"),
    (CR, 100, "one_run(anchor"),
    (PCF, 115, "Intel Core Ultra 9 285K"),
    (NPU_DOC, 114, "int8"),
    (CONF, 70, "global8"),
]
for src, ln, needle in CITES:
    got = src[ln - 1] if 0 < ln <= len(src) else "<범위 밖>"
    if needle not in got:
        problems.append(f"줄 근거 불일치: {ln} 에 {needle!r} 없음 -> {got.strip()!r}")

# ---------------------------------------------------------------- (1) 재계산
CSV = ROOT / "xgboost_model/full_collection_540/analysis/greedy_results.csv"
d = pd.read_csv(CSV, comment="#")
recomputed = {}
for (pl, var), g in d.groupby(["platform", "variant"]):
    c = g[g["collected"] == 1]
    recomputed[f"{pl}/{var}"] = {
        "n_groups": int(len(g)),
        "n_collected": int(len(c)),
        "n_uncollected": int((g["collected"] == 0).sum()),
        # (a) 그룹별 비율의 산술평균, 수집 그룹만
        "mean_of_group_ratios_norm": round(float(np.nanmean(c["ratio_norm"])), 3),
        "mean_of_group_ratios_raw": round(float(np.nanmean(c["ratio_raw"])), 3),
        # (b) 반증 후보: 합의 비율 (원고 문장이 이쪽이면 값이 달라야 한다)
        "ratio_of_sums_norm": round(float(c["s_norm"].sum() / c["oracle_norm"].sum()), 3),
        "ratio_of_sums_raw": round(float(c["s_raw"].sum() / c["oracle_raw"].sum()), 3),
        # (c) 반증 후보: 전체 45그룹 기준 (미수집을 0으로)
        "mean_over_all_45_norm": round(float(np.nansum(c["ratio_norm"]) / len(g)), 3),
        "top1": round(float(c["top1"].mean()), 3),
        "n_nan_ratio_norm": int(c["ratio_norm"].isna().sum()),
        "ratio_norm_min": round(float(c["ratio_norm"].min()), 4),
        "ratio_norm_max": round(float(c["ratio_norm"].max()), 4),
    }

# 원고 Table 5 (App.D) 기재값
TABLE5 = {"gpu/load_factor": (0.341, 0.575, 0.669, 41, 4),
          "gpu/speedup":     (0.424, 0.734, 0.786, 33, 12),
          "npu/load_factor": (0.250, 0.590, 0.669, 36, 9),
          "npu/speedup":     (0.333, 0.793, 0.760, 33, 12)}
for k, (t1, rn, rr, ncol, nunc) in TABLE5.items():
    r = recomputed[k]
    checks[f"table5_{k}_mean_of_group_ratios_matches"] = (
        r["mean_of_group_ratios_norm"] == rn and r["mean_of_group_ratios_raw"] == rr)
    checks[f"table5_{k}_top1_and_denominators_match"] = (
        r["top1"] == t1 and r["n_collected"] == ncol and r["n_uncollected"] == nunc)
    # 대안 정의는 값이 달라야 한다 (= 원고 문장이 (a)임을 배타적으로 확정)
    checks[f"table5_{k}_ratio_of_sums_differs"] = r["ratio_of_sums_norm"] != rn
checks["no_nan_ratios_so_denominator_is_exactly_collected_groups"] = all(
    r["n_nan_ratio_norm"] == 0 for r in recomputed.values())
checks["group_max_is_the_same_oracle_as_top1"] = True  # GB:93,96 동일 oracle_n 사용
# 그룹 내 accel_set 중복이 없어 match.iloc[0] 이 유일해라는 확인
sys.path.insert(0, str(ROOT / "xgboost_model" / "full_collection_540" / "scripts"))
import analysis_common as ac  # noqa: E402
dup = 0
for pl in ac.PLATFORMS:
    _, _, M = ac.load_dataset(pl)
    for _, g in M.groupby([M["models"], M["rate_factor"]], sort=False):
        if g["accel_set"].duplicated().any():
            dup += 1
checks["accel_set_unique_within_group_so_iloc0_is_exact"] = dup == 0

# ---------------------------------------------------------------- (3) 열거 정책
import importlib  # noqa: E402
import model_registry as reg  # noqa: E402

gcs = importlib.import_module("xgboost_model.gen_collection_schedules")
gs = importlib.import_module("generate_schedules")

SETS = yaml.safe_load((ROOT / "working_sets.yaml").read_text())["sets"]
STRAT = set(yaml.safe_load((ROOT / "working_sets.yaml").read_text())["stratified_sets"])

enum_cmp = {}
for sid, models in SETS.items():
    for accel in ("gpu", "npu"):
        a = {tuple(sorted((m, accel if v == "accel" else "cpu") for m, v in p.items()))
             for p in gcs.all_placements(models)}
        b = {tuple(sorted(p.items()))
             for p in gs.enumerate_placements(models, ["cpu", accel])}
        enum_cmp[f"{sid}/{accel}"] = {"all_placements": len(a),
                                      "enumerate_placements": len(b),
                                      "identical": a == b}
checks["two_enumerators_agree_on_full_feasible_set"] = all(
    v["identical"] for v in enum_cmp.values())

# 수집 경로는 층화 세트에서 부분집합만 쓴다
sampling = {}
for sid, models in SETS.items():
    ps, tags = gcs.combos_for(sid, models)
    full = len(gcs.all_placements(models))
    sampling[sid] = {"full_feasible": full, "collection_emits": len(ps),
                     "stratified": sid in STRAT, "is_subset": len(ps) <= full}
checks["collection_samples_only_stratified_sets"] = all(
    (v["collection_emits"] < v["full_feasible"]) == v["stratified"]
    for v in sampling.values())
checks["deploy_path_has_no_sampling"] = not any(
    "stratif" in l or "sample" in l.lower()
    for l in GS[57:72] + BDF[662:726])

# 잠재적 발산: platform_devices 와 교집합이 빈 모델의 처리
fallback_case = gs.enumerate_placements(["qwen2_vl"], ["cpu"])
checks["enumerate_placements_falls_back_to_cpu_for_infeasible_model"] = (
    fallback_case == [{"qwen2_vl": "cpu"}])
all_pl_case = gcs.all_placements(["qwen2_vl"])
checks["all_placements_pins_vlm_to_accel"] = all_pl_case == [{"qwen2_vl": "accel"}]

from runtime.llm_engine import LLM_PROMPTS, DEFAULT_PROMPT  # noqa: E402
checks["llm_prompts_are_4_unique"] = len(LLM_PROMPTS) == 4 == len(set(LLM_PROMPTS))
checks["vlm_prompt_is_the_fixed_default"] = DEFAULT_PROMPT.startswith("Describe what is happening")
import subprocess  # noqa: E402
cpuinfo = subprocess.run(["grep", "-m1", "model name", "/proc/cpuinfo"],
                         capture_output=True, text=True).stdout
checks["live_host_cpu_is_core_ultra_9_285k"] = "Core(TM) Ultra 9 285K" in cpuinfo
checks["manuscript_i7_13700k_not_found_in_repo"] = not subprocess.run(
    ["grep", "-rl", "--exclude-dir=.venv", "--exclude-dir=.git",
     "--exclude-dir=runs", "--exclude-dir=manuscript", "13700K", "."],
    cwd=ROOT, capture_output=True, text=True).stdout.strip()

for k, v in checks.items():
    if not v:
        problems.append(f"검산 실패: {k}")

facts = {
    "q1_oracle_ratio": {
        "verdict": "(a) 그룹별 비율의 산술평균 — 수집된 그룹만",
        "formula": "ratio_col = mean_{g in G_collected} ( S(greedy_g) / max_{c in C_g} S(c) )",
        "denominator_per_group": "그룹의 best MEASURED score (Top-1 의 oracle 과 동일 값)",
        "aggregation_set": "greedy 산출 배치가 수집 조합에 존재하는 그룹만 "
                           "(gpu/load_factor 41, gpu/speedup 33, npu/load_factor 36, "
                           "npu/speedup 33 — 45 중)",
        "two_columns": "norm. = 그룹 정규화 점수 기준, raw = 원시 측정 총계 기준",
        "code": {"per_group_ratio": "greedy_baseline.py:100,102 (CSV) / :107-108 (집계)",
                 "oracle": "greedy_baseline.py:93",
                 "collected_only": "greedy_baseline.py:94 (len(match)) / :113-115 (else -> unc)",
                 "mean": "greedy_baseline.py:134 (np.nanmean)"},
        "recomputed": recomputed,
    },
    "q3_candidate_policy": {
        "enumerator_comparison": enum_cmp,
        "collection_sampling": sampling,
    },
    "checks": checks,
    "problems": problems,
    "input_sha256": {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                     for rel in ["xgboost_model/full_collection_540/scripts/greedy_baseline.py",
                                 "xgboost_model/full_collection_540/analysis/greedy_results.csv",
                                 "xgboost_model/gen_collection_schedules.py",
                                 "generate_schedules.py",
                                 "best_deploy_finder_executor.py",
                                 "runtime/llm_engine.py", "model_processors.py",
                                 "model_registry.py", "requirements.txt"]},
}
(HERE / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False))

print("== (1) oracle ratio 재계산 ==")
for k, r in sorted(recomputed.items()):
    print(f"  {k}: mean-of-ratios {r['mean_of_group_ratios_norm']:.3f}/"
          f"{r['mean_of_group_ratios_raw']:.3f}  | ratio-of-sums "
          f"{r['ratio_of_sums_norm']:.3f}  | all-45 {r['mean_over_all_45_norm']:.3f}"
          f"  | n={r['n_collected']}/{r['n_groups']}")
print("== (3) 열거 정책 ==")
print(f"  두 열거 함수 전 30조합 동일: "
      f"{all(v['identical'] for v in enum_cmp.values())}")
print(f"  수집이 부분집합인 세트: "
      f"{sorted(s for s, v in sampling.items() if v['collection_emits'] < v['full_feasible'])}")
print("== 검산 ==")
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
if problems:
    print("\n=== 문제 ===")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("\n검증 통과")
