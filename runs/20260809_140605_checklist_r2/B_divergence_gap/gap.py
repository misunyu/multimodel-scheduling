"""B (#9 1단계): 불일치 11그룹의 score gap 정량화.

각 불일치 그룹에서, 플랫폼 P 기준으로
    own_best   = P 에서의 최고 측정 점수
    cross      = 상대 플랫폼의 최적 배치가 P 에서 받은 점수
    gap        = own_best - cross            (>= 0)
을 계산하고, tie 폭(1e-9) 및 그룹 내 후보 점수의 표준편차·범위와 비교한다.

점수는 원고 선언 기저(그룹 정규화, alpha=0.3, beta=1.0)를 쓴다 — 점수 정의는
runs/20260730_190154_score_basis_audit/build_basis_audit.py 를 import 해서 공유한다
(재구현 금지).

판정 문장은 쓰지 않는다. 수치만.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

AUDIT = ROOT / "runs/20260730_190154_score_basis_audit/build_basis_audit.py"
spec = importlib.util.spec_from_file_location("basis_audit", AUDIT)
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)

ALPHA, BETA, BASIS = 0.3, 1.0, "normalized"
TIE = ba.TIE_EPS

rows = {pl: ba.normalize(ba.load_rows(pl)) for pl in ("gpu", "npu")}
by_group = {pl: defaultdict(list) for pl in rows}
for pl, rs in rows.items():
    for r in rs:
        by_group[pl][(r["sid"], r["rate"])].append(r)

ties = {pl: ba.tie_sets(rows[pl], ALPHA, BETA, BASIS) for pl in rows}
keys = sorted(ties["gpu"], key=lambda k: (k[0], k[1]))
disagree = [k for k in keys if not (ties["gpu"][k] & ties["npu"][k])]

# F1 README 가 기록한 기대 11그룹과 대조
EXPECTED = {("base3", 1.0), ("base3", 3.0), ("base4", 1.0), ("base4", 3.0),
            ("base5", 1.0), ("base5", 2.0), ("base5", 3.0), ("S5", 3.0),
            ("S10", 2.0), ("S7", 2.0), ("S8", 2.0)}
assert set(disagree) == EXPECTED, (sorted(set(disagree) ^ EXPECTED))

table = []
for k in disagree:
    entry = {"set": k[0], "rate": k[1],
             "has_gen": bool(ba.HAS_GEN[k[0]]),
             "n_candidates": len(by_group["gpu"][k])}
    for pl, other in (("gpu", "npu"), ("npu", "gpu")):
        cand = by_group[pl][k]
        s = {r["placement"]: ba.score(r, ALPHA, BETA, BASIS) for r in cand}
        own_best = max(s.values())
        # 상대 플랫폼의 최적 배치(타이 집합)가 이 플랫폼에서 받는 점수 — 최댓값을 취한다
        cross_pl = [s[p] for p in ties[other][k] if p in s]
        assert cross_pl, (pl, k)
        cross = max(cross_pl)
        vals = np.array(list(s.values()), float)
        gap = own_best - cross
        rng = float(vals.max() - vals.min())
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        # 2등과의 간격 (자기 플랫폼 내부 경쟁의 촘촘함)
        srt = np.sort(vals)[::-1]
        top_gap = float(srt[0] - srt[1]) if len(srt) > 1 else float("nan")
        entry[pl] = {
            "own_best": round(own_best, 6),
            "cross_score": round(cross, 6),
            "gap_abs": round(gap, 6),
            "gap_over_sd": round(gap / sd, 4) if sd > 1e-12 else None,
            "gap_over_range": round(gap / rng, 4) if rng > 1e-12 else None,
            "gap_over_own_best": round(gap / own_best, 4) if abs(own_best) > 1e-12 else None,
            "group_score_sd": round(sd, 6),
            "group_score_range": round(rng, 6),
            "own_top1_to_top2_gap": round(top_gap, 6),
            "gap_over_tie_eps": round(gap / TIE, 1),
            "own_best_label": sorted(ba.cpu_label(p) for p in ties[pl][k]),
            "cross_best_label": sorted(ba.cpu_label(p) for p in ties[other][k]),
        }
    entry["min_gap_abs"] = min(entry["gpu"]["gap_abs"], entry["npu"]["gap_abs"])
    entry["max_gap_abs"] = max(entry["gpu"]["gap_abs"], entry["npu"]["gap_abs"])
    table.append(entry)

table.sort(key=lambda e: e["min_gap_abs"])

gaps_all = [e[pl]["gap_abs"] for e in table for pl in ("gpu", "npu")]
ratios_sd = [e[pl]["gap_over_sd"] for e in table for pl in ("gpu", "npu")
             if e[pl]["gap_over_sd"] is not None]
ratios_rng = [e[pl]["gap_over_range"] for e in table for pl in ("gpu", "npu")
              if e[pl]["gap_over_range"] is not None]


def summ(a, nd=6):
    a = np.asarray(a, float)
    return {"n": int(a.size), "min": round(float(a.min()), nd),
            "p25": round(float(np.percentile(a, 25)), nd),
            "median": round(float(np.median(a)), nd),
            "p75": round(float(np.percentile(a, 75)), nd),
            "max": round(float(a.max()), nd), "mean": round(float(a.mean()), nd)}


# 참고 척도: 45그룹 전체의 그룹 내 점수 표준편차 분포
all_sd = []
for k in keys:
    for pl in ("gpu", "npu"):
        v = np.array([ba.score(r, ALPHA, BETA, BASIS) for r in by_group[pl][k]], float)
        if len(v) > 1:
            all_sd.append(float(v.std(ddof=1)))

# ------------------------------------------------- 0.01 band 재판정 (2026-08-17 추가)
# App H 는 "under a coarser score-equivalence band of 0.01 ... the divergence count
# would be 9 of 45" 라고 쓰는데, 그 9 는 계산된 적이 없었다 (11 − 아래 밴드 미만 셀 2개
# 라는 산술 역산). 규칙이 두 가지로 갈리므로 둘 다 계산해 둔다:
#
#   any  — 두 margin 중 하나라도 밴드 안이면 그 그룹을 agreeing 으로 되돌린다
#   both — 양쪽 margin 이 모두 밴드 안이어야 agreeing
#
# any 는 관대하다: base3@3.0 은 NPU margin 0.0068 로 밴드에 들어오지만 GPU margin 은
# 0.8648 이다. 한쪽에서 거의 무손실이라는 이유로 다른 쪽의 0.86 손실을 덮는 셈이라,
# 두 수를 함께 보고할 수 있도록 both 도 남긴다.
BANDS = (0.005, 0.01, 0.02, 0.05)
N_GROUPS_TOTAL = len(keys)


def band_recount(band):
    """밴드 폭 `band` 에서 규칙별로 남는 divergence 수와 되돌려진 그룹."""
    res = {}
    for rule, agg in (("any", min), ("both", max)):
        # any:  min(gap) < band  -> 하나라도 안에 들어옴
        # both: max(gap) < band  -> 둘 다 안에 들어옴
        reverted = [f"{e['set']}@{e['rate']}" for e in table
                    if agg(e["gpu"]["gap_abs"], e["npu"]["gap_abs"]) < band]
        res[rule] = {"n_reverted": len(reverted), "reverted_groups": sorted(reverted),
                     "n_disagree": len(table) - len(reverted),
                     "n_groups_total": N_GROUPS_TOTAL}
    return res


band_table = {f"{b:g}": band_recount(b) for b in BANDS}
# 원고가 인용하는 값 — 여기서 어긋나면 App H 문장을 고쳐야 한다
_a01 = band_table["0.01"]["any"]["n_disagree"]
_b01 = band_table["0.01"]["both"]["n_disagree"]
assert (_a01, _b01) == (9, 11), f"0.01 band: any={_a01} both={_b01}, expected (9, 11)"

out = {
    "basis": {"name": "declared_normalized (원고 식 (1))", "alpha": ALPHA, "beta": BETA,
              "tie_eps": TIE,
              "score_defs_imported_from": str(AUDIT.relative_to(ROOT))},
    "n_disagree_groups": len(disagree),
    "score_equivalence_band": {
        "_rules": {
            "any": "그룹의 두 margin 중 하나라도 band 미만이면 agreeing 으로 되돌린다",
            "both": "두 margin 이 모두 band 미만이어야 agreeing 으로 되돌린다"},
        "_manuscript_cites": {"band": 0.01, "rule": "any", "n_disagree": 9,
                              "n_groups_total": N_GROUPS_TOTAL},
        "by_band": band_table,
    },
    "groups": table,
    "gap_distribution_22_cells": summ(gaps_all),
    "gap_over_group_sd": summ(ratios_sd, 4),
    "gap_over_group_range": summ(ratios_rng, 4),
    "reference_group_score_sd_all_90_cells": summ(all_sd),
    "tie_eps_comparison": {
        "tie_eps": TIE,
        "smallest_gap_abs": round(min(gaps_all), 6),
        "smallest_gap_in_tie_eps_units": round(min(gaps_all) / TIE, 1),
        "n_cells_with_gap_below_1e-3": int(sum(1 for g in gaps_all if g < 1e-3)),
        "n_cells_with_gap_below_1e-2": int(sum(1 for g in gaps_all if g < 1e-2)),
    },
}
(HERE / "gap_facts.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

# 사람이 읽는 표
md = ["| 세트@rate | gen | 후보 | GPU own_best | GPU cross | GPU gap | gap/sd | "
      "NPU own_best | NPU cross | NPU gap | gap/sd |", "|" + "---|" * 11]
for e in table:
    g, n = e["gpu"], e["npu"]
    md.append(f"| {e['set']}@{e['rate']} | {'Y' if e['has_gen'] else '-'} "
              f"| {e['n_candidates']} "
              f"| {g['own_best']:.4f} | {g['cross_score']:.4f} | **{g['gap_abs']:.4f}** "
              f"| {g['gap_over_sd']} "
              f"| {n['own_best']:.4f} | {n['cross_score']:.4f} | **{n['gap_abs']:.4f}** "
              f"| {n['gap_over_sd']} |")
(HERE / "gap_table.md").write_text("\n".join(md) + "\n")

print("== score-equivalence band 재판정 ==")
print(f"  {'band':>7} {'any: n_div':>11} {'되돌린 그룹':>22} {'both: n_div':>12} {'되돌린 그룹':>12}")
for b, r in band_table.items():
    print(f"  {b:>7} {r['any']['n_disagree']:>11} "
          f"{','.join(r['any']['reverted_groups']) or '-':>22} "
          f"{r['both']['n_disagree']:>12} "
          f"{','.join(r['both']['reverted_groups']) or '-':>12}")
print()
print(f"불일치 {len(disagree)}그룹 × 2플랫폼 = {len(gaps_all)} 셀")
print("gap 절대값:", out["gap_distribution_22_cells"])
print("gap / 그룹내 sd:", out["gap_over_group_sd"])
print("tie 폭 대비 최소 gap:", out["tie_eps_comparison"])
print("\n" + "\n".join(md[:6]))
