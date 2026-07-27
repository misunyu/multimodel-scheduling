#!/usr/bin/env python3
"""
그림 사이드카(*.values.json)를 confirmed_values.json과 대조한다.

이전 버전은 값만 비교했다. 그래서 "확정표에 실재하지만 다른 기준점에서 잰 값"이
그대로 통과했다(bounded_recovery_analysis가 Q3에 24.6(burst 기준)을 t0 축에 그린 건
값 검사만으로는 잡히지 않는다). 이 버전은 값과 함께 quantity / reference /
window_s / censored 를 모두 대조하고, 필드가 없으면 통과가 아니라 실패로 처리한다.

usage:  python3 check_figures.py [--figdir fig] [--tol 0.05]
exit 0 = 불일치 없음, exit 1 = 불일치 있음
"""

import argparse
import json
import pathlib
import sys

# 사이드카 항목에 반드시 있어야 하는 필드. 없으면 통과가 아니라 실패다.
REQUIRED = ("id", "value", "quantity", "reference", "unit", "n", "censored", "recovered")

# censored 항목은 창 길이가 없으면 검증 불가능하므로 추가로 요구한다.
REQUIRED_IF_CENSORED = ("window_s",)

# censored 와 recovered 는 독립이다.
#   censored = 값이 관측 창에 잘렸는가 (해칭 + 화살표)
#   recovered = 그 런이 회복했는가 (색/테두리)
# Q4 의 search=24s 처럼 "회복하지 못했지만 완결된 측정값"이 존재하므로 둘을 겹쳐 쓰면 안 된다.

# 값 외에 정확히 일치해야 하는 필드 (오차 허용 없음).
EXACT_MATCH = ("quantity", "reference", "unit", "censored", "recovered", "window_s")

VALID_QUANTITY = {"search", "drain", "persist", "cumV", "lastV", "bound"}
VALID_REFERENCE = {"t0", "burst", "detection_onset"}


def load_confirmed(path):
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    table = {}
    for entry in doc["values"]:
        if entry["id"] in table:
            raise SystemExit(f"confirmed_values.json: duplicate id {entry['id']}")
        table[entry["id"]] = entry
    return table


def check_entry(item, confirmed, sidecar_name, tol):
    """한 사이드카 항목을 검사해 문제 문자열 리스트를 돌려준다."""
    problems = []
    where = f"{sidecar_name}"

    missing = [f for f in REQUIRED if f not in item]
    if missing:
        ident = item.get("id", "<no id>")
        return [f"{where}[{ident}]: 필수 필드 누락 {missing}"]

    ident = item["id"]
    where = f"{sidecar_name}[{ident}]"

    if item["censored"]:
        missing = [f for f in REQUIRED_IF_CENSORED if item.get(f) is None]
        if missing:
            problems.append(f"{where}: censored 항목인데 {missing} 없음")

    if item["censored"] and item["recovered"]:
        problems.append(f"{where}: censored=true 인데 recovered=true — 회복했다면 값이 창에 잘릴 수 없다")

    if item["quantity"] not in VALID_QUANTITY:
        problems.append(f"{where}: 알 수 없는 quantity {item['quantity']!r}")
    if item["reference"] not in VALID_REFERENCE:
        problems.append(f"{where}: 알 수 없는 reference {item['reference']!r}")

    ref = confirmed.get(ident)
    if ref is None:
        problems.append(f"{where}: confirmed_values.json에 없는 id — 확정표에 없는 값을 그렸다")
        return problems

    got, want = item["value"], ref["value"]
    if want in (0, 0.0):
        ok = abs(got - want) <= tol
    else:
        ok = abs(got - want) <= abs(want) * tol
    if not ok:
        problems.append(f"{where}: value {got} != confirmed {want}")

    for field in EXACT_MATCH:
        if field not in item:
            continue
        if item[field] != ref.get(field):
            problems.append(
                f"{where}: {field} {item[field]!r} != confirmed {ref.get(field)!r}"
            )

    if "sd" in ref and "sd" in item and ref["sd"] != item["sd"]:
        problems.append(f"{where}: sd {item['sd']} != confirmed {ref['sd']}")
    if "n" in ref and ref["n"] is not None and item["n"] != ref["n"]:
        problems.append(f"{where}: n {item['n']} != confirmed {ref['n']}")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="fig")
    ap.add_argument("--confirmed", default=None)
    ap.add_argument("--tol", type=float, default=0.05, help="상대 허용오차 (기본 5%%)")
    args = ap.parse_args()

    figdir = pathlib.Path(args.figdir)
    confirmed_path = args.confirmed or (figdir / "confirmed_values.json")
    confirmed = load_confirmed(confirmed_path)

    sidecars = sorted(figdir.glob("*.values.json"))
    if not sidecars:
        print(f"사이드카를 찾지 못했다: {figdir}/*.values.json")
        return 1

    problems, checked = [], 0
    seen_ids = set()

    for path in sidecars:
        doc = json.loads(path.read_text(encoding="utf-8"))
        items = doc["values"] if isinstance(doc, dict) else doc
        for item in items:
            checked += 1
            if isinstance(item, dict) and "id" in item:
                seen_ids.add(item["id"])
            problems += check_entry(item, confirmed, path.name, args.tol)

    # 확정표에 있는데 어떤 그림도 쓰지 않은 값: 그림 갱신 누락의 신호일 수 있다.
    unused = [
        i for i, e in confirmed.items()
        if e.get("figure") and i not in seen_ids
    ]

    print(f"checked {checked} values across {len(sidecars)} sidecars")
    if unused:
        print(f"\n[warn] 확정표에 figure가 지정됐으나 사이드카에 없는 값 {len(unused)}건:")
        for i in unused:
            print(f"  - {i} (figure: {confirmed[i]['figure']})")

    if problems:
        print(f"\n[FAIL] {len(problems)} mismatches:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\n[OK] ALL FIGURE VALUES MATCH CONFIRMED TABLE (0 mismatches).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
