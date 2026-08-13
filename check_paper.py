#!/usr/bin/env python3
"""Where does each number in the paper come from, and is it still that value?

One command answers both:

    python check_paper.py            # provenance table + verification
    python check_paper.py --json     # machine-readable

For every cell of Figure 2 and Table 1 it prints the producing script, the artifact
field the figure/table reads, the current value, and whether that value still matches
what the manuscript prints. It also checks the measurement inputs against paper.lock
and reports the SHA-256 of every artifact and booster involved.

Read-only: nothing is written, nothing is retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AN = ROOT / "xgboost_model/full_collection_540/analysis"
ART = ROOT / "xgboost_model/artifacts"
TEX = ROOT / "manuscript/mlforsys_main.tex"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(p: Path):
    return json.loads(p.read_text())


# ---------------------------------------------------------------- provenance map
# (row, platform, metric) -> how to get it
SPEC = load(AN / "groupkfold_gpu_metrics.json"), load(AN / "groupkfold_npu_metrics.json")
CP = load(AN / "cross_platform_metrics.json")
JOINT = load(AN / "unified_metrics.json")
DEC = load(AN / "decomposed_metrics.json")

_spec = {"gpu": SPEC[0], "npu": SPEC[1]}
_other = {"gpu": "npu", "npu": "gpu"}

ROWS = []


def add(figure, series, pl, metric, value, script, artifact, field):
    ROWS.append({"figure": figure, "series": series, "platform": pl, "metric": metric,
                 "value": value, "printed": round(value, 3), "script": script,
                 "artifact": artifact, "field": field})


for pl in ("gpu", "npu"):
    r = _spec[pl]["ranking"]
    for metric, key in (("Spearman", "group_spearman_mean"), ("Top-1", "top1")):
        add("Fig.2 + Table 1", "platform-specific", pl, metric, r[key],
            "scripts/train_groupkfold.py",
            f"analysis/groupkfold_{pl}_metrics.json", f"ranking.{key}")

    z = CP[f"{_other[pl]}->{pl}|fold_matched"]["ranking"]
    for metric, key in (("Spearman", "group_spearman_mean"), ("Top-1", "top1")):
        add("Fig.2", "zero-shot", pl, metric, z[key],
            "scripts/cross_platform_transfer.py",
            "analysis/cross_platform_metrics.json",
            f"{_other[pl]}->{pl}|fold_matched.ranking.{key}")

    j = JOINT["per_platform"][pl]["ranking"]
    for metric, key in (("Spearman", "group_spearman_mean"), ("Top-1", "top1")):
        add("Fig.2", "joint", pl, metric, j[key],
            "scripts/train_unified.py", "analysis/unified_metrics.json",
            f"per_platform.{pl}.ranking.{key}")

    d = DEC[pl]["ranking"]
    for metric, key in (("Spearman", "group_spearman_mean"), ("Top-1", "top1")):
        add("Table 1", "decomposed", pl, metric, d[key],
            "scripts/train_decomposed.py", "analysis/decomposed_metrics.json",
            f"{pl}.ranking.{key}")

# greedy: Top-1 only, over measured outputs (speedup ordering is the reported one)
import csv  # noqa: E402

gr = list(csv.DictReader(
    (l for l in (AN / "greedy_results.csv").read_text().splitlines()
     if not l.startswith("#"))))
for pl in ("gpu", "npu"):
    sel = [r for r in gr if r["platform"] == pl and r["variant"] == "speedup"
           and r["collected"] == "1"]
    add("Table 1", "capacity greedy", pl, "Top-1",
        sum(int(r["top1"]) for r in sel) / len(sel),
        "scripts/greedy_baseline.py", "analysis/greedy_results.csv",
        "top1 over collected==1 rows, variant=speedup")


# ------------------------------------------------------------------ manuscript
def printed_in_tex(tex: str, value: float) -> bool:
    return f"{value:.3f}" in tex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tex = TEX.read_text() if TEX.exists() else ""
    problems = []

    for r in ROWS:
        r["in_manuscript"] = printed_in_tex(tex, r["value"]) if tex else None
        if tex and not r["in_manuscript"]:
            problems.append(f"{r['series']}/{r['platform']} {r['metric']} "
                            f"= {r['printed']} not found in the manuscript")

    lock = subprocess.run(["sha256sum", "-c", "paper.lock"], cwd=ROOT,
                          capture_output=True, text=True)
    n_ok = sum(1 for l in lock.stdout.splitlines() if l.strip().endswith(": OK"))
    if lock.returncode != 0:
        problems.append("paper.lock verification failed")

    artifacts = {p: sha(ROOT / p) for p in [
        "xgboost_model/full_collection_540/analysis/groupkfold_gpu_metrics.json",
        "xgboost_model/full_collection_540/analysis/groupkfold_npu_metrics.json",
        "xgboost_model/full_collection_540/analysis/cross_platform_metrics.json",
        "xgboost_model/full_collection_540/analysis/unified_metrics.json",
        "xgboost_model/full_collection_540/analysis/decomposed_metrics.json",
        "xgboost_model/full_collection_540/analysis/greedy_results.csv",
        "xgboost_model/full_collection_540/analysis/figures/fig_divergence.pdf",
        "xgboost_model/full_collection_540/analysis/figures/fig_transfer.pdf",
    ] if (ROOT / p).exists()}
    boosters = {f"gkf_cpu_{pl}_{t}_fold{f}.json":
                sha(ART / f"gkf_cpu_{pl}_{t}_fold{f}.json")
                for pl in ("gpu", "npu") for t in ("y1", "y2", "y3") for f in range(3)}

    # the joint split must be the paired one, and zero-shot the fold-matched variant
    if JOINT.get("n_groups") != 45:
        problems.append(f"unified_metrics.json n_groups={JOINT.get('n_groups')}, "
                        "expected 45 (paired). Re-run scripts/train_unified.py")
    if not any(k.endswith("|fold_matched") for k in CP):
        problems.append("cross_platform_metrics.json has no fold_matched variant. "
                        "Re-run scripts/cross_platform_transfer.py")

    out = {
        "manuscript": {"path": str(TEX.relative_to(ROOT)), "sha256": sha(TEX)}
        if TEX.exists() else None,
        "rows": ROWS,
        "paper_lock": {"checked": n_ok, "exit": lock.returncode},
        "artifacts": artifacts,
        "boosters": boosters,
        "regenerate": [
            "python xgboost_model/full_collection_540/scripts/train_groupkfold.py",
            "python xgboost_model/full_collection_540/scripts/cross_platform_transfer.py",
            "python xgboost_model/full_collection_540/scripts/train_unified.py",
            "python xgboost_model/full_collection_540/scripts/train_decomposed.py",
            "python xgboost_model/full_collection_540/scripts/greedy_baseline.py",
            "python xgboost_model/full_collection_540/scripts/make_figures.py --fig f1",
            "python xgboost_model/full_collection_540/scripts/make_figures.py --fig f2",
        ],
        "problems": problems,
        "ok": not problems,
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if out["ok"] else 1

    print(f"manuscript  {TEX.name}  sha256 {sha(TEX)[:16]}…\n" if TEX.exists() else "")
    print(f"{'figure/table':16} {'series':18} {'plat':5} {'metric':9} "
          f"{'value':>8} {'in tex':7} script")
    print("-" * 100)
    for r in ROWS:
        mark = "yes" if r["in_manuscript"] else ("—" if r["in_manuscript"] is None
                                                 else "NO")
        print(f"{r['figure']:16} {r['series']:18} {r['platform']:5} {r['metric']:9} "
              f"{r['printed']:8.3f} {mark:7} {r['script']}")
        print(f"{'':16} {'':18} {'':5} {'':9} {'':8} {'':7} "
              f"  └ {r['artifact']} : {r['field']}")
    print(f"\npaper.lock: {n_ok} files OK (exit {lock.returncode})")
    print(f"artifacts hashed: {len(artifacts)}, boosters hashed: {len(boosters)}")
    print("\nregenerate everything:")
    for c in out["regenerate"]:
        print(f"  {c}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("\nOK — every Figure 2 / Table 1 value is present in the manuscript, "
              "the joint split is paired (45 groups), zero-shot is fold-matched, "
              "and paper.lock verifies.")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
