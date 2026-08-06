"""작업 12 검증: report.md 의 줄 근거·수치를 재확인하고 facts.json 을 남긴다. 읽기 전용."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STATIC = ROOT / "xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json"
problems = []


def lines(rel):
    return (ROOT / rel).read_text().splitlines()


PM = lines("profile_models.py")
LE = lines("runtime/llm_engine.py")
SU = lines("xgboost_model/deploy_selector_xgb_suite.py")
GB = lines("xgboost_model/full_collection_540/scripts/greedy_baseline.py")
TD = lines("xgboost_model/full_collection_540/scripts/train_decomposed.py")
FS = lines("runs/20260804_180759_feature_spec/feature_spec.md")
MG = lines("xgboost_model/MODEL_GUIDE.md")

CITES = [
    # 생성 측
    (PM, 12, "cpu_load"), (PM, 12, "# ms"),
    (PM, 29, "--out xgboost_model/performance_data/sample_profiling_data"),
    (PM, 86, "def profile_vision"),
    (PM, 90, "t0 = time.time()"), (PM, 91, "build_vision_npu"),
    (PM, 92, "load_ms = (time.time() - t0) * 1000.0"),
    (PM, 114, "t0 = time.time()"), (PM, 115, "ort.InferenceSession"),
    (PM, 116, "load_ms = (time.time() - t0) * 1000.0"),
    (PM, 98, "for _ in range(warmup)"), (PM, 101, "for _ in range(iters)"),
    (PM, 132, "def profile_generative"), (PM, 139, 'p["load_ms"]'),
    (PM, 229, 'row[f"{dev}_load"] = round(load_ms, 3)'),
    (PM, 234, 'row[f"{dev}_load"] = round(load_ms, 3)'),
    (PM, 179, "prev = json.loads(outp.read_text())"),
    # LLMEngine
    (LE, 88, "t0 = time.time()"),
    (LE, 103, "self.load_ms = (time.time() - t0) * 1000.0"),
    (LE, 237, "def profile"), (LE, 246, '"load_ms": self.load_ms'),
    # 소비 측
    (SU, 101, "def load_static_profiles"), (SU, 118, '"load": _f("load")'),
    (SU, 126, "def _device_static"), (SU, 250, "def _view_features"),
    (SU, 253, "s_infer, s_load, s_tokens = _device_static"),
    (SU, 260, "capacity = (1000.0 / s_infer)"),
    (SU, 261, "load_factor = (infps / capacity)"),
    (SU, 273, '"view.static_load_sel": s_load'),
    # 이름만 같은 다른 load 들
    (GB, 37, 'cpu_infer, _, _ = _device_static'),
    (TD, 61, "s_infer, _, _ = _device_static"),
    (TD, 67, '"lf": infps / cap'),
    (TD, 75, '"co.same_dev_load_sum"'), (TD, 77, '"co.other_dev_load_sum"'),
    # 문서
    (FS, 44, "단독 모델 로드 시간"), (FS, 47, "view.infps / view.capacity_fps"),
    (FS, 56, "단독 모델 로드 시간"), (FS, 68, "단독 모델 로드 시간"),
    (MG, 392, "view.static_{infer,load,tokens}_sel"),
    (MG, 393, "단독 프로파일 값"),
]
for src, ln, needle in CITES:
    got = src[ln - 1] if 0 < ln <= len(src) else "<범위 밖>"
    if needle not in got:
        problems.append(f"줄 근거 불일치: {ln} 에 {needle!r} 없음 -> {got.strip()!r}")

# ---- _view_features 안에서 s_load 가 쓰이는 곳이 정확히 1곳인지 -----------------
body = SU[249:280]
n_sload = sum(l.count("s_load") for l in body)
checks = {}
checks["s_load_only_unpack_guard_and_one_use"] = n_sload == 5  # 언팩1 + 결측보정3 + 사용1
checks["s_load_used_in_exactly_one_feature_row"] = sum(
    1 for l in body if "s_load" in l and l.strip().startswith('"view.')) == 1
checks["capacity_uses_only_s_infer"] = all(
    "s_load" not in l for l in body if "capacity" in l or "load_factor" in l)

# ---- 값 확인 ------------------------------------------------------------------
blob = json.loads(STATIC.read_text())
rows = {r["model"]: r for r in blob["total_data"]}
pairs = {(m, d): rows[m].get(f"{d}_load") for m in rows for d in ("cpu", "gpu", "npu")}
checks["27_pairs_no_null"] = len(pairs) == 27 and all(v is not None for v in pairs.values())
checks["rep1_mobilenet_cpu"] = pairs[("mobilenet_v2", "cpu")] == 15.052
checks["rep2_yolo11x_cpu"] = pairs[("yolo11x", "cpu")] == 295.199
checks["rep3_qwen2vl_npu"] = pairs[("qwen2_vl", "npu")] == 1674.867
checks["qwen2vl_npu_is_global_max"] = max(pairs.values()) == pairs[("qwen2_vl", "npu")]
checks["mobilenet_gpu_is_global_min"] = min(pairs.values()) == pairs[("mobilenet_v2", "gpu")]

GEN = ["llama1b", "qwen2_vl"]
VIS = ["resnet50", "mobilenet_v2", "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
gen_loads = [pairs[(m, d)] for m in GEN for d in ("cpu", "gpu", "npu")]
vis_loads = [pairs[(m, d)] for m in VIS for d in ("cpu", "gpu", "npu")]
checks["all_generative_loads_exceed_all_vision"] = min(gen_loads) > max(vis_loads)
checks["vision_load_range"] = (round(min(vis_loads), 1), round(max(vis_loads), 1)) == (12.9, 295.2)
checks["gen_load_range"] = (round(min(gen_loads), 1), round(max(gen_loads), 1)) == (665.4, 1674.9)

from utils import resolve_onnx_path  # noqa: E402
size = {m: os.path.getsize(resolve_onnx_path(m)) / 1e6 for m in VIS}
rho = {}
for dev in ("cpu", "gpu", "npu"):
    L = [pairs[(m, dev)] for m in VIS]
    I = [rows[m][f"{dev}_infer"] for m in VIS]
    S = [size[m] for m in VIS]
    v2 = [m for m in VIS if m != "resnet50"]
    rho[dev] = {
        "size_vs_load": round(float(spearmanr(S, L).statistic), 3),
        "size_vs_load_excl_resnet50": round(float(spearmanr(
            [size[m] for m in v2], [pairs[(m, dev)] for m in v2]).statistic), 3),
        "load_vs_infer": round(float(spearmanr(L, I).statistic), 3),
    }
checks["size_corr_dominates_infer_corr"] = all(
    rho[d]["size_vs_load"] > rho[d]["load_vs_infer"] for d in rho)
checks["size_corr_at_least_0_92"] = all(rho[d]["size_vs_load"] >= 0.92 for d in rho)
checks["resnet50_first_in_recorded_run"] = \
    blob["profiling_runs"][0]["models"][0] == "resnet50"
checks["resnet50_inflated_on_gpu_npu_not_cpu"] = (
    pairs[("resnet50", "gpu")] > pairs[("yolo11l", "gpu")] * 3
    and pairs[("resnet50", "npu")] > pairs[("yolo11l", "npu")]
    and pairs[("resnet50", "cpu")] < pairs[("yolo11x", "cpu")])
checks["one_profiling_run_recorded"] = len(blob["profiling_runs"]) == 1
checks["generative_models_absent_from_run_record"] = not any(
    m in blob["profiling_runs"][0]["models"] for m in GEN)
checks["onnx_sizes"] = (round(size["resnet50"], 2), round(size["yolo11l"], 2),
                        round(size["yolo11x"], 2)) == (102.15, 101.74, 228.06)

# ---- 판정 반증 시도: 부하 조회 API 호출이 프로파일 경로에 있는가 -----------------
NEEDLES = ["nvidia-smi", "psutil", "utilization", "memory_used", "getUtilization",
           "device_status", "nvml"]
hits = [(rel, i + 1, l.strip()) for rel, src in
        (("profile_models.py", PM), ("runtime/llm_engine.py", LE))
        for i, l in enumerate(src) for n in NEEDLES if n in l]
checks["no_utilization_query_in_profiler"] = not hits

for k, v in checks.items():
    if not v:
        problems.append(f"검산 실패: {k}")

facts = {
    "verdict": {
        "field": "sample_profiling_data.json:total_data[].{cpu,gpu,npu}_load",
        "meaning": "모델을 해당 디바이스에 적재해 추론 가능 상태로 만들기까지의 벽시계 시간",
        "unit": "ms",
        "is_runtime_load_indicator": False,
        "producer": "profile_models.py:229 (vision) / :234 (generative)",
        "timed_region": {
            "vision_npu": "profile_models.py:90-92 (build_vision_npu)",
            "vision_cpu_gpu": "profile_models.py:114-116 (ort.InferenceSession)",
            "generative": "runtime/llm_engine.py:88-103 (from_pretrained + .to(cuda) + .eval())",
        },
        "averaging": "1회 측정, warmup 없음 (infer 는 warmup 3 + 15회 평균)",
    },
    "consumer": {
        "feature": ["views.{sum,mean,max}.view.static_load_sel"],
        "path": "load_static_profiles(suite:101,118) -> _device_static(suite:126) "
                "-> _view_features(suite:253,273)",
        "transform": "없음 (결측만 0.0 대체) — ms 그대로",
        "load_factor_uses_static_load": False,
        "load_factor_definition": "suite:260-261 — capacity=1000/s_infer, "
                                  "load_factor=infps/capacity",
        "other_load_named_features_unrelated": [
            "views.*.view.load_factor", "co.same_dev_load_sum", "co.other_dev_load_sum"],
    },
    "values": {f"{m}/{d}": v for (m, d), v in sorted(pairs.items())},
    "representative_pairs": {
        "mobilenet_v2/cpu": pairs[("mobilenet_v2", "cpu")],
        "yolo11x/cpu": pairs[("yolo11x", "cpu")],
        "qwen2_vl/npu": pairs[("qwen2_vl", "npu")],
    },
    "onnx_size_mb": {m: round(v, 2) for m, v in size.items()},
    "spearman": rho,
    "doc_consistency": {
        "feature_spec.md": "일치 (로드 시간/ms/단독 프로파일, load_factor 와 분리 기술)",
        "MODEL_GUIDE.md:390-393": "부분 불일치 — load_factor·capacity_fps 를 '단독 프로파일 값'에 "
                                  "묶어 서술 (실제로는 infps·s_infer 파생). 결론 무영향",
    },
    "unconfirmed": [
        "생성 모델 2행의 측정 실행 기록이 파일에 없음 (profiling_runs 는 vision 1건뿐)",
        "resnet50 의 GPU/NPU load 팽창이 최초 컨텍스트 초기화 때문이라는 것은 정황 추론",
        "build_vision_npu 가 적재 구간에서 수행하는 범위",
    ],
    "checks": checks,
    "problems": problems,
    "input_sha256": {
        rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        for rel in ["profile_models.py", "runtime/llm_engine.py",
                    "xgboost_model/performance_data/sample_profiling_data/"
                    "sample_profiling_data.json",
                    "xgboost_model/deploy_selector_xgb_suite.py"]
    },
}
(HERE / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False))

print("판정:", facts["verdict"]["meaning"], "| 단위", facts["verdict"]["unit"])
print("대표 3쌍:", facts["representative_pairs"])
print("spearman:", json.dumps(rho, ensure_ascii=False))
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
if problems:
    print("\n=== 문제 ===")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("\n검증 통과")
