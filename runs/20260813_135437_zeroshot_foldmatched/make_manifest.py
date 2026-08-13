"""작업 20 검증 + run_manifest.json."""
import hashlib, json, subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
SC = Path("/tmp/claude-1001/-home-msyu-PycharmProjects-multimodel-scheduling-mobilint/"
          "fc826e0d-e247-401e-85d9-cf9c7527fb1a/scratchpad")
AN = ROOT / "xgboost_model/full_collection_540/analysis"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run(*c, cwd=ROOT):
    return subprocess.run(c, cwd=cwd, capture_output=True, text=True).stdout


cp = json.loads((AN / "cross_platform_metrics.json").read_text())
EXP = {"npu->gpu": (0.9274, 0.9333), "gpu->npu": (0.9073, 0.8000)}
checks, cells = {}, {}
for cell, (er, et) in EXP.items():
    r = cp[f"{cell}|fold_matched"]["ranking"]
    cells[cell] = {"group_spearman_mean": r["group_spearman_mean"], "top1": r["top1"],
                   "top5": r["top5"], "n_groups": r["n_groups"],
                   "asis_group_spearman_mean":
                       cp[f"{cell}|asis"]["ranking"]["group_spearman_mean"]}
    checks[f"{cell}_rho_0.4dp"] = round(r["group_spearman_mean"], 4) == er
    checks[f"{cell}_top1_0.4dp"] = round(r["top1"], 4) == et
    checks[f"{cell}_n_groups_45"] = r["n_groups"] == 45
checks["diagonal_fold_matched_equals_P1_OOF"] = all(
    cp[f"{p}->{p}|fold_matched"]["ranking"] == cp[f"{p}->{p}|asis"]["ranking"]
    for p in ("gpu", "npu"))
checks["asis_and_swap_preserved"] = all(
    f"{c}|{v}" in cp for c in EXP for v in ("asis", "swap"))

# 그림 대조
run("pdftotext", "-layout", str(AN / "figures/fig_transfer.pdf"), str(SC / "f2_after.txt"))
a = Counter((SC / "f2_before.txt").read_text().split())
b = Counter((SC / "f2_after.txt").read_text().split())
removed, added = dict(a - b), dict(b - a)
checks["figure_only_0902_to_0907"] = removed == {"0.902": 1} and added == {"0.907": 1}
checks["figure_token_count_same"] = sum(a.values()) == sum(b.values())
labels = sorted(t for t in b if t.replace(".", "").isdigit() and t.startswith("0."))
size = next((l.split(":", 1)[1].strip() for l in
             run("pdfinfo", str(AN / "figures/fig_transfer.pdf")).splitlines()
             if l.lower().startswith("page size")), "?")
checks["page_size_unchanged"] = size.startswith("326.228 x 164.499")
from PIL import Image
checks["png_904x457"] = Image.open(AN / "figures/fig_transfer.png").size == (904, 457)
checks["manuscript_copy_matches"] = sha(ROOT / "manuscript/fig_transfer.pdf") == \
    sha(AN / "figures/fig_transfer.pdf")
checks["fig_divergence_unchanged"] = sha(AN / "figures/fig_divergence.pdf") == \
    "3c13b1dfe82814c0df6a7a4b7b715b35cd08912c316eaa215e0a575c098f1fd5"
tex = (ROOT / "manuscript/mlforsys_main.tex").read_text()
checks["manuscript_says_0.907"] = "$0.972$ to $0.907$" in tex
checks["manuscript_no_0.902"] = "0.902" not in tex

latex = {t: (run("bash", "-c", f"command -v {t} || true").strip() or None)
         for t in ("pdflatex", "latexmk", "tectonic", "bibtex")}

m = {
    "run": HERE.name,
    "task": "작업 20 — zero-shot fold-matched 전환 및 fig_transfer.pdf 재생성",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": run("git", "rev-parse", "HEAD").strip() + " (pre-commit HEAD)",
    "no_retraining": "저장된 gkf_cpu_{pl}_{tag}_fold{f}.json 추론만 사용",
    "implementation": {
        "fold_matched_fn": "xgboost_model/full_collection_540/scripts/"
                           "cross_platform_transfer.py:78-102 (transfer_predict_fold_matched)",
        "variant_loop": "cross_platform_transfer.py:142",
        "fold_consistency_assert": "cross_platform_transfer.py:121-127 "
                                   "(fold_map['gpu'] == fold_map['npu'])",
        "figure_reader": "make_figures.py ZEROSHOT_VARIANT / EXPECTED_ZEROSHOT guard",
        "old_path_kept": "transfer_predict (mean-of-three) 그대로 유지, asis/swap 변형 보존",
    },
    "results": cells,
    "figure": {"labels": labels, "page_size": size,
               "pdftotext_removed": removed, "pdftotext_added": added},
    "backup": {p.name: sha(p) for p in sorted((HERE / "backup").iterdir())},
    "regenerated": {
        "analysis/cross_platform_metrics.json": sha(AN / "cross_platform_metrics.json"),
        "analysis/cross_platform.md": sha(AN / "cross_platform.md"),
        "analysis/figures/fig_transfer.pdf": sha(AN / "figures/fig_transfer.pdf"),
        "analysis/figures/fig_transfer.png": sha(AN / "figures/fig_transfer.png"),
        "manuscript/fig_transfer.pdf": sha(ROOT / "manuscript/fig_transfer.pdf"),
    },
    "boosters_used": {f"gkf_cpu_{pl}_{t}_fold{f}.json":
                      sha(ROOT / f"xgboost_model/artifacts/gkf_cpu_{pl}_{t}_fold{f}.json")
                      for pl in ("gpu", "npu") for t in ("y1", "y2", "y3")
                      for f in range(3)},
    "commands": [
        "python xgboost_model/full_collection_540/scripts/cross_platform_transfer.py",
        "python xgboost_model/full_collection_540/scripts/make_figures.py --fig f2",
        "cp xgboost_model/full_collection_540/analysis/figures/fig_transfer.pdf "
        "manuscript/fig_transfer.pdf",
    ],
    "latex_toolchain": latex,
    "compile_attempted": False,
    "compile_blocked_reason": "pdflatex/latexmk/tectonic/bibtex 모두 미설치 — 작업 4 수행 불가",
    "verification": {"checks": checks, "all_passed": all(checks.values())},
}
(HERE / "run_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
print("\nall_passed:", m["verification"]["all_passed"])
print("figure labels:", labels)
