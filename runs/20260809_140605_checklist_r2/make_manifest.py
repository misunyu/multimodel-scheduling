"""작업 14 run_manifest.json 생성."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


A = json.loads((HERE / "A_joint_paired/paired_metrics.json").read_text())
B = json.loads((HERE / "B_divergence_gap/gap_facts.json").read_text())
C = json.loads((HERE / "C_bootstrap/bootstrap_facts.json").read_text())
D = json.loads((HERE / "D_greedy_bound/bound_facts.json").read_text())
E = json.loads((HERE / "E_r3_definition/r3_facts.json").read_text())
F = json.loads((HERE / "F_regret/regret_facts.json").read_text())

INPUTS = [
    "xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json",
    "xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json",
    "xgboost_model/full_collection_540/analysis/groupkfold_gpu_oof.csv",
    "xgboost_model/full_collection_540/analysis/groupkfold_npu_oof.csv",
    "xgboost_model/full_collection_540/analysis/groupkfold_gpu_metrics.json",
    "xgboost_model/full_collection_540/analysis/groupkfold_npu_metrics.json",
    "xgboost_model/full_collection_540/analysis/unified_metrics.json",
    "xgboost_model/full_collection_540/analysis/decomposed_metrics.json",
    "xgboost_model/full_collection_540/analysis/decomposed_view_oof.csv",
    "xgboost_model/full_collection_540/analysis/greedy_results.csv",
    "xgboost_model/full_collection_540/scripts/analysis_common.py",
    "xgboost_model/full_collection_540/scripts/train_unified.py",
    "xgboost_model/full_collection_540/scripts/jsonl_to_windows.py",
    "xgboost_model/full_collection_540/scripts/collect_run.py",
    "xgboost_model/deploy_selector_xgb_suite.py",
    "unified_viewer.py", "view_handlers.py", "working_sets.yaml",
    "runs/20260730_190154_score_basis_audit/build_basis_audit.py",
]

outputs = {}
for p in sorted(HERE.rglob("*")):
    if p.is_file() and p.name != "run_manifest.json" and "__pycache__" not in str(p):
        outputs[str(p.relative_to(HERE))] = sha256(p)

manifest = {
    "run": HERE.name,
    "task": "작업 14 — 체크리스트 분석·재학습 대응 (A~F)",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
                  + " (pre-commit HEAD)",
    "scope": {
        "created": [f"runs/{HERE.name}/**"],
        "modified": [],
        "note": "기존 아티팩트·측정 데이터·원고 미수정. A 만 재학습했고 부스터·지표는 "
                "이 디렉터리에만 저장했다 (artifacts/, analysis/ 에 쓰지 않음).",
        "retraining": "A 만. 3 targets x 3 folds x 2 splits, seed 42, "
                      "analysis_common 헬퍼 그대로",
    },
    "inputs": {p: sha256(ROOT / p) for p in INPUTS if (ROOT / p).exists()},
    "outputs": outputs,
    "findings": {
        "A_joint_paired": {
            "pair_straddle_in_old_split": f"{A['pair_straddle']['n']}/{A['pair_straddle']['of']}",
            "old90": {pl: A["old90"]["per_platform"][pl]["ranking"] for pl in ("gpu", "npu")},
            "new45": {pl: A["new45"]["per_platform"][pl]["ranking"] for pl in ("gpu", "npu")},
            "delta": A["delta_new_minus_old"],
            "stored_reproduced": A["stored_unified_metrics_reproduced"],
            "verdict": "수치 유지 또는 개선 → 숫자 교체 가능 "
                       "(NPU Top-1 0.844→0.867, NPU 그룹-rho 0.974→0.984, GPU 사실상 불변)",
        },
        "B_divergence_gap": {
            "n_cells": B["gap_distribution_22_cells"]["n"],
            "gap_abs": B["gap_distribution_22_cells"],
            "gap_over_group_sd": B["gap_over_group_sd"],
            "tie_eps_comparison": B["tie_eps_comparison"],
        },
        "C_bootstrap": {k: v for k, v in C["results"].items()},
        "D_greedy_bound": {f"{r['platform']}/{r['variant']}":
                           {"reported": r["reported_fraction"],
                            "lower_bound": r["lower_bound_fraction"],
                            "lower_bound_top1": r["lower_bound_top1_all_groups"]}
                           for r in D["cells"]},
        "E_r3": {
            "verdict": E["verdict_measurement_time"]["definition"],
            "device_filter": E["verdict_measurement_time"]["device_filter"],
            "two_generative_models": E["verdict_measurement_time"]["two_generative_models"],
            "stored_per_view_is_reconstructed": True,
            "two_conventions_differ_in_windows":
                E["two_r3_conventions_in_repo"]["where_they_differ"]["n_windows"],
            "numeric_impact": "없음 — S3 그룹의 max(r3) 가 두 관례에서 동일",
            "y3_masking": E["y3_masking"],
        },
        "F_regret": {t["method"] + "/" + t["platform"] + "/" + t["denominator"]:
                     {"top1": t["top1"], "oracle_ratio_mean": t["oracle_ratio"]["mean"],
                      "regret_mean": t["regret"]["mean"],
                      "n_below_0.9": t["oracle_ratio"]["n_below_0.9"]}
                     for t in F["table"]},
    },
    "manuscript_actions": [
        "A: joint 숫자 교체 (Abstract NPU 84.4 -> 86.7, Fig.transfer joint NPU 0.974 -> 0.984)",
        "E: r3 정의문 추가 (모든 인스턴스 토큰율의 합, CPU 배치 생성 인스턴스 포함) + "
        "y3 마스킹 168창 구성(vision-only 156 + CPU-LLM 12) 명시",
        "D: Table 5 에 lower-bound 열 추가 시 'speedup 이 두 플랫폼 모두 우세' 문장을 "
        "NPU 로 한정해야 함",
        "C: GPU 에서 direct-decomposed 차이(+0.022)는 CI 하한이 0 — 무조건 서술 불가",
    ],
    "verification": {
        "A_stored_reproduced": A["stored_unified_metrics_reproduced"],
        "C_point_checks": C["checks"],
        "D_checks": D["checks"],
        "E_checks": E["checks"],
        "F_checks": F["checks"],
        "all_passed": (all(A["stored_unified_metrics_reproduced"].values())
                       and all(C["checks"].values()) and all(D["checks"].values())
                       and all(E["checks"].values()) and all(F["checks"].values())),
    },
}
(HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print("all_passed:", manifest["verification"]["all_passed"])
print("outputs:", len(outputs))
