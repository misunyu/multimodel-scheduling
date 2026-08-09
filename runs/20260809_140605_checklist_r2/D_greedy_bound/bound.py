"""D (#11): greedy 의 worst-case 하한 Top-1.

현행 보고는 '측정 산출이 있는 그룹'만 분모로 쓴다(예: gpu/speedup 14/33 = 0.424).
미측정 산출(greedy 가 낸 배치가 수집 조합에 없음)을 전부 실패로 간주하면 분모가 45가
되어 하한이 나온다. 두 값을 병기 가능한 형태로 계산한다.

읽기 전용 — analysis/greedy_results.csv 만 읽는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
CSV = ROOT / "xgboost_model/full_collection_540/analysis/greedy_results.csv"

d = pd.read_csv(CSV, comment="#")
rows = []
for (pl, var), g in d.groupby(["platform", "variant"]):
    c = g[g["collected"] == 1]
    hits = int(c["top1"].sum())
    n_all, n_col = int(len(g)), int(len(c))
    n_unc = n_all - n_col
    rows.append({
        "platform": pl, "variant": var,
        "n_groups": n_all, "n_measured": n_col, "n_unmeasured": n_unc,
        "top1_hits": hits,
        "reported_top1_measured_only": round(hits / n_col, 3),
        "reported_fraction": f"{hits}/{n_col}",
        "lower_bound_top1_all_groups": round(hits / n_all, 3),
        "lower_bound_fraction": f"{hits}/{n_all}",
        # 상한: 미측정을 전부 성공으로 가정 (참고용 — 실제 주장에 쓰지 않음)
        "upper_bound_top1_all_groups": round((hits + n_unc) / n_all, 3),
        "upper_bound_fraction": f"{hits + n_unc}/{n_all}",
        "spread_upper_minus_lower": round(n_unc / n_all, 3),
    })
rows.sort(key=lambda r: (r["platform"], r["variant"]))

# 원고 Table 5 기재값과 대조
TABLE5 = {("gpu", "load_factor"): (0.341, "14/41"), ("gpu", "speedup"): (0.424, "14/33"),
          ("npu", "load_factor"): (0.250, "9/36"), ("npu", "speedup"): (0.333, "11/33")}
checks = {}
for r in rows:
    k = (r["platform"], r["variant"])
    checks[f"reported_matches_table5_{k[0]}_{k[1]}"] = (
        r["reported_top1_measured_only"] == TABLE5[k][0]
        and r["reported_fraction"] == TABLE5[k][1])
checks["hits_are_unchanged_across_denominators"] = all(
    r["top1_hits"] == int(r["lower_bound_fraction"].split("/")[0]) for r in rows)

out = {
    "definition": {
        "reported": "Top-1 = (정확히 맞힌 그룹 수) / (greedy 산출이 측정된 그룹 수)",
        "lower_bound": "Top-1_lb = (정확히 맞힌 그룹 수) / 45 — 미측정 산출을 전부 실패로 간주",
        "note": "분자(맞힌 그룹 수)는 두 정의에서 동일하다. 분모만 바뀐다.",
        "source": "analysis/greedy_results.csv (P5, greedy_baseline.py)",
    },
    "cells": rows,
    "checks": checks,
}
(HERE / "bound_facts.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

md = ["| 플랫폼 | 정렬 | 보고치 (측정분) | 하한 (45그룹) | 미측정 | 상한(참고) |",
      "|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['platform'].upper()} | {r['variant']} "
              f"| {r['reported_top1_measured_only']:.3f} ({r['reported_fraction']}) "
              f"| **{r['lower_bound_top1_all_groups']:.3f}** ({r['lower_bound_fraction']}) "
              f"| {r['n_unmeasured']}/45 "
              f"| {r['upper_bound_top1_all_groups']:.3f} ({r['upper_bound_fraction']}) |")
(HERE / "bound_table.md").write_text("\n".join(md) + "\n")

print("\n".join(md))
print()
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
