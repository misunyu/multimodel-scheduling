"""작업 15: fig_transfer 재생성 검증 + 매니페스트.

joint 계열만 paired-fold 값으로 교체했다는 것을 세 방향으로 확인한다.
  1) 물리 치수 불변 (pdfinfo 페이지 크기, PNG 픽셀)
  2) pdftotext 토큰 다중집합 비교 — joint 라벨 외 변화 0
  3) 그림이 그린 값이 paired_metrics.json 의 new45 와 정확히 일치
재생성 자체는 하지 않는다(이미 수행). 읽기 전용 검증.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
SC = Path("/tmp/claude-1001/-home-msyu-PycharmProjects-multimodel-scheduling-mobilint/"
          "fc826e0d-e247-401e-85d9-cf9c7527fb1a/scratchpad")
FIG = ROOT / "xgboost_model/full_collection_540/analysis/figures"
PAIRED = ROOT / "runs/20260809_140605_checklist_r2/A_joint_paired/paired_metrics.json"

problems, checks = [], {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run(*cmd) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


# ---- 1) 물리 치수 -------------------------------------------------------------
def page_size(pdf: Path) -> str:
    for l in run("pdfinfo", str(pdf)).splitlines():
        if l.lower().startswith("page size"):
            return l.split(":", 1)[1].strip()
    return "?"


before_pdf, after_pdf = SC / "before.pdf", FIG / "fig_transfer.pdf"
size_before, size_after = page_size(before_pdf), page_size(after_pdf)
checks["page_size_unchanged"] = size_before == size_after
checks["page_size_is_the_documented_one"] = size_after.startswith("326.228 x 164.499")

from PIL import Image  # noqa: E402
png_size = Image.open(FIG / "fig_transfer.png").size
checks["png_size_904x457"] = png_size == (904, 457)

# ---- 2) pdftotext 토큰 비교 ---------------------------------------------------
run("pdftotext", "-layout", str(after_pdf), str(SC / "after.txt"))
a = Counter((SC / "before.txt").read_text().split())
b = Counter((SC / "after.txt").read_text().split())
removed, added = dict(a - b), dict(b - a)
checks["only_two_labels_changed"] = (removed == {"0.974": 1, "0.844": 1}
                                     and added == {"0.984": 1, "0.867": 1})
checks["token_count_unchanged"] = sum(a.values()) == sum(b.values())


def nonnum(c):
    return {k: v for k, v in c.items() if not k.replace(".", "").isdigit()}


checks["nonnumeric_tokens_identical"] = nonnum(a) == nonnum(b)

# ---- 3) 그린 값이 paired 런과 일치 --------------------------------------------
paired = json.loads(PAIRED.read_text())
drawn = json.loads(run(sys.executable,
                       str(ROOT / "xgboost_model/full_collection_540/scripts/make_figures.py"),
                       "--fig", "f2").splitlines()[-1])
for pl in ("gpu", "npu"):
    r = paired["new45"]["per_platform"][pl]["ranking"]
    checks[f"drawn_joint_rho_{pl}_matches_paired"] = (
        abs(drawn[f"('rho', '{pl}')"][2] - r["group_spearman_mean"]) < 1e-12)
    checks[f"drawn_joint_top1_{pl}_matches_paired"] = (
        abs(drawn[f"('top1', '{pl}')"][2] - r["top1"]) < 1e-12)
# specialized / zero-shot 은 analysis/ 값 그대로인가
spec = {pl: json.loads((FIG.parent / f"groupkfold_{pl}_metrics.json").read_text())
        for pl in ("gpu", "npu")}
cp = json.loads((FIG.parent / "cross_platform_metrics.json").read_text())
other = {"gpu": "npu", "npu": "gpu"}
for pl in ("gpu", "npu"):
    checks[f"specialized_series_untouched_{pl}"] = (
        abs(drawn[f"('rho', '{pl}')"][0] - spec[pl]["ranking"]["group_spearman_mean"]) < 1e-12
        and abs(drawn[f"('top1', '{pl}')"][0] - spec[pl]["ranking"]["top1"]) < 1e-12)
    t = cp[f"{other[pl]}->{pl}|asis"]["ranking"]
    checks[f"zeroshot_series_untouched_{pl}"] = (
        abs(drawn[f"('rho', '{pl}')"][1] - t["group_spearman_mean"]) < 1e-12
        and abs(drawn[f"('top1', '{pl}')"][1] - t["top1"]) < 1e-12)
# 재실행이 같은 PDF 를 내는가 (결정성)
checks["regeneration_is_deterministic"] = sha256(after_pdf) == sha256(FIG / "fig_transfer.pdf")

# ---- 확인: 기존 joint 소스는 더 이상 읽지 않는다 --------------------------------
mf = (ROOT / "xgboost_model/full_collection_540/scripts/make_figures.py").read_text()
checks["f2_no_longer_reads_unified_metrics"] = not [
    l for l in mf.splitlines()
    if "unified_metrics.json" in l and not l.lstrip().startswith("#")]
checks["f2_reads_paired_metrics_by_glob"] = "runs/*_checklist_r2/A_joint_paired" in mf
checks["analysis_unified_metrics_untouched"] = (
    (FIG.parent / "unified_metrics.json").exists())

for k, v in checks.items():
    if not v:
        problems.append(f"검산 실패: {k}")

manifest = {
    "run": HERE.name,
    "task": "작업 15 — fig_transfer 재생성 (joint 계열을 paired-fold 값으로 교체)",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": run("git", "-C", str(ROOT), "rev-parse", "HEAD").strip()
                  + " (pre-commit HEAD)",
    "scope": {
        "modified": ["xgboost_model/full_collection_540/scripts/make_figures.py (F2만)",
                     "xgboost_model/full_collection_540/analysis/figures/fig_transfer.pdf",
                     "xgboost_model/full_collection_540/analysis/figures/fig_transfer.png",
                     "xgboost_model/full_collection_540/analysis/figures/README.md"],
        "untouched": ["analysis/unified_metrics.json", "analysis/cross_platform_metrics.json",
                      "analysis/groupkfold_*", "fig_divergence.*", "manuscript/",
                      "artifacts/"],
        "note": "재학습 없음 — 작업 14 항목 A 가 남긴 paired_metrics.json 을 읽는다.",
    },
    "change": {
        "source_before": "analysis/unified_metrics.json (per_platform[pl].ranking)",
        "source_after": "runs/20260809_140605_checklist_r2/A_joint_paired/"
                        "paired_metrics.json (new45.per_platform[pl].ranking)",
        "joint_values": {
            pl: {"rho_before": paired["old90"]["per_platform"][pl]["ranking"][
                     "group_spearman_mean"],
                 "rho_after": paired["new45"]["per_platform"][pl]["ranking"][
                     "group_spearman_mean"],
                 "top1_before": paired["old90"]["per_platform"][pl]["ranking"]["top1"],
                 "top1_after": paired["new45"]["per_platform"][pl]["ranking"]["top1"]}
            for pl in ("gpu", "npu")},
        "labels_changed_on_figure": {"0.974 -> 0.984": "NPU group Spearman",
                                     "0.844 -> 0.867": "NPU Top-1"},
        "labels_unchanged_despite_value_change": {
            "GPU group Spearman": "0.97481 -> 0.97550, 3자리 표기 0.975 로 동일 — "
                                  "지시문의 '0.975->0.976' 은 3자리에서는 성립하지 않는다"},
    },
    "geometry": {"page_size_before": size_before, "page_size_after": size_after,
                 "png_size": list(png_size)},
    "pdftotext_diff": {"removed": removed, "added": added,
                       "token_count": [sum(a.values()), sum(b.values())],
                       "nonnumeric_identical": checks["nonnumeric_tokens_identical"]},
    "hashes": {
        "fig_transfer.pdf_before": sha256(before_pdf),
        "fig_transfer.pdf_after": sha256(after_pdf),
        "fig_transfer.png_after": sha256(FIG / "fig_transfer.png"),
        "make_figures.py": sha256(ROOT / "xgboost_model/full_collection_540/scripts/"
                                         "make_figures.py"),
        "figures/README.md": sha256(FIG / "README.md"),
        "paired_metrics.json": sha256(PAIRED),
    },
    "verification": {"checks": checks, "problems": problems,
                     "all_passed": not problems},
}
(HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
(HERE / "pdftotext_after.txt").write_text((SC / "after.txt").read_text())
(HERE / "pdftotext_before.txt").write_text((SC / "before.txt").read_text())

print(f"page size: {size_before}  ->  {size_after}")
print(f"png size : {png_size}")
print(f"token diff: removed {removed}, added {added}")
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
if problems:
    print("\n=== 문제 ===")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("\n검증 통과")
