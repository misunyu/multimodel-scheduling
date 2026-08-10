"""작업 19-r2: run_manifest.json + verification_report.md 생성."""
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
F = json.loads((HERE / "facts.json").read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


INPUTS = [
    "manuscript/mlforsys_main.tex",
    "xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json",
    "xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json",
    "xgboost_model/schedules/collection/collect_cpu_gpu.yaml",
    "xgboost_model/schedules/collection/collect_cpu_npu.yaml",
    "xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json",
    "working_sets.yaml", "model_registry.py",
    "xgboost_model/deploy_selector_xgb_suite.py",
    "xgboost_model/full_collection_540/scripts/analysis_common.py",
    "xgboost_model/full_collection_540/scripts/train_decomposed.py",
    "xgboost_model/full_collection_540/scripts/make_figures.py",
    "xgboost_model/full_collection_540/analysis/greedy_results.csv",
    "runs/20260730_190154_score_basis_audit/build_basis_audit.py",
    "runs/20260809_140605_checklist_r2/A_joint_paired/paired_metrics.json",
    "runs/20260809_140605_checklist_r2/C_bootstrap/bootstrap.py",
    "xgboost_model/full_collection_540/analysis/figures/fig_divergence.pdf",
    "xgboost_model/full_collection_540/analysis/figures/fig_transfer.pdf",
]

# 수집 데이터 무결성 (paper.lock 부재 → 이전 run 기록과 대조)
rec = json.loads((ROOT / "runs/20260806_103943_model_guide/run_manifest.json")
                 .read_text())["input_data_files"]
integ = {"source": "runs/20260806_103943_model_guide/run_manifest.json:input_data_files",
         "paper_lock_present": (ROOT / "paper.lock").exists(),
         "n_checked": 0, "n_match": 0, "mismatches": []}
for p, h in rec.items():
    if not h:
        continue
    integ["n_checked"] += 1
    cur = sha(ROOT / p)
    if cur == h:
        integ["n_match"] += 1
    else:
        integ["mismatches"].append({"path": p, "recorded": h, "current": cur})

outputs = {}
for p in sorted(HERE.rglob("*")):
    if p.is_file() and p.name != "run_manifest.json" and "__pycache__" not in str(p):
        outputs[str(p.relative_to(HERE))] = sha(p)

by_sec = Counter(r["section"] for r in F["comparison"])
manifest = {
    "run": HERE.name,
    "task": "작업 19-r2 — 논문 그림·표 재현 절차 문서화 + 재현성 검증",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
                  + " (pre-commit HEAD)",
    "scope": {"created": ["REPRODUCE.md", f"runs/{HERE.name}/**"], "modified": [],
              "note": "artifacts/·analysis/·figures/·manuscript/ 미수정. 재현 산출물은 "
                      "이 run 디렉터리에만. make_figures.py 는 FIGDIR 만 런타임 교체."},
    "manuscript": F["manuscript"],
    "anchors": F["anchors"],
    "gate": {"required_marker": ">= r19", "found_marker": "r18a",
             "r19_exists_on_system": False,
             "decision": "앵커 3건 전부 통과 + 시스템 최신본이 r18a 이므로 r18a 를 정본으로 진행",
             "copies_checked": ["manuscript/mlforsys_main.tex", "~/Downloads/mlforsys_main.tex",
                                "~/Downloads/_Neurips_MLForSys__Predictive_Placement/"
                                "mlforsys_main.tex"]},
    "inputs": {p: sha(ROOT / p) for p in INPUTS if (ROOT / p).exists()},
    "outputs": outputs,
    "verification": {
        "n_items": F["summary"]["n_items"],
        "n_agree": F["summary"]["n_agree"],
        "n_disagree": F["summary"]["n_disagree"],
        "n_conditional": F["summary"]["n_conditional"],
        "by_section": dict(sorted(by_sec.items())),
        "disagreements": F["summary"]["disagreements"],
        "all_agree": F["summary"]["n_disagree"] == 0,
        "negative_control": "reproduced.json 의 direct/gpu Top-1 을 0.8 로 바꾸면 "
                            "비교기가 2건을 불일치로 잡는다 (Table 1 · §4) — 검증됨",
    },
    "timings_sec": {**F["timings_sec"], "figures": 0.3},
    "determinism_spotcheck": F["determinism_spotcheck"],
    "bootstrap_seed": {"fixed": True, "value": 42,
                       "source": "runs/20260809_140605_checklist_r2/C_bootstrap/"
                                 "bootstrap.py:36,117",
                       "note": "재현 CI 가 App.H 표와 소수 셋째 자리까지 일치 (29항목)"},
    "data_integrity": integ,
    "environment": {"python": "3.10.12", "xgboost": "3.2.0", "numpy": "2.2.6",
                    "pandas": "2.3.3", "scipy": "1.15.3", "matplotlib": "3.10.9",
                    "pyyaml": "6.0.3", "poppler": "pdftotext/pdfinfo 사용 가능",
                    "latex": "미설치 (pdflatex/latexmk/tectonic 모두 없음)"},
}
(HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print("all_agree:", manifest["verification"]["all_agree"])
print("integrity:", integ["n_match"], "/", integ["n_checked"],
      "| paper.lock:", integ["paper_lock_present"])
print("by_section:", manifest["verification"]["by_section"])
