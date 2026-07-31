#!/usr/bin/env python3
"""Classify the numerals `check_paper.py` cannot check, by why it cannot.

A single count of "untracked numerals" is not actionable: a monitoring window of 3 s and a
service rate whose only provenance is a deleted run are both untracked, and only one of them
is a problem. This splits the count four ways, and the third class is the residual risk --
values the paper prints that cannot be traced to a committed run.

The classification is by LINE, hand-assigned, and lives here rather than in prose so that it
is recomputed against the current main.tex instead of being a tally that silently rots.
Lines not listed fall into `unclassified`, which should be empty; if it is not, the paper
changed and the map needs a line.

usage: python3 scripts/classify_untracked.py [--paper <dir>]
"""
import argparse, collections, json, os, subprocess, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# line -> (class, why)
CLASSES = {
    "constant": "정의상의 상수·설정 — 측정이 아니므로 낡을 수 없다",
    "table_i": "Table~I 모델 지연 — 별도 출처(docs/benchmark_model_table.csv), 런 디렉터리 아님",
    "hardware": "장비 사양 — 측정이 아니다",
    "table_cell": "표 셀 — 갱신 시 통째로 교체되므로 산문보다 위험이 낮다(우선순위 4)",
    "unresolvable": "**되짚을 수 없음** — 커밋된 런으로 추적 불가. 실질적 잔여 위험",
    "derived": "파생값 — 확정표 두 항목의 차/비이며 항목 자체가 아니다",
    "not_yet": "달 수 있으나 아직 달지 않음",
    "worklist": "주석 목록(docs/paper_cv_annotations.json)에 이미 올라 있음 — 정본 반영 대기",
}

LINE_CLASS = {
    421: "hardware", 425: "hardware", 434: "hardware",
    438: "constant", 459: "constant", 503: "constant", 516: "constant", 519: "constant",
    520: "constant", 521: "constant", 552: "constant", 554: "constant", 560: "constant",
    583: "constant", 598: "constant", 600: "constant", 602: "constant", 624: "constant",
    625: "constant", 627: "constant", 635: "constant", 654: "constant", 658: "constant",
    680: "constant", 690: "constant", 756: "constant", 784: "constant", 809: "constant",
    811: "constant", 821: "constant",
    441: "table_i", 442: "table_i", 443: "table_i", 444: "table_i", 445: "table_i",
    446: "table_i", 447: "table_i", 448: "table_i", 449: "table_i",
    664: "table_cell", 665: "table_cell", 666: "table_cell", 667: "table_cell",
    669: "table_cell", 670: "table_cell", 671: "table_cell", 672: "table_cell",
    791: "table_cell", 792: "table_cell", 793: "table_cell", 794: "table_cell",
    796: "table_cell",
    462: "unresolvable", 585: "unresolvable", 712: "unresolvable", 714: "unresolvable",
    739: "unresolvable", 743: "unresolvable", 745: "unresolvable", 747: "unresolvable",
    752: "unresolvable", 773: "unresolvable", 775: "unresolvable",
    581: "derived", 815: "derived",
    644: "not_yet", 646: "not_yet", 648: "not_yet", 684: "not_yet", 686: "not_yet",
    688: "not_yet",
}

NOTE = {
    462: "vision-4 구성 비교 V(t) 5.2/3.4 — **어느 리포트에도 없다**",
    585: "전환 시 서비스 공백 0.15–0.21 / 1.0 / 1.5–2.2 s — switching_lossless_report.md(07-24), 런 없음",
    712: "baseline 위반 수준 3.7 — 셀 평균 최저는 3.8(q3_sr_maxV). 4.1은 등재됨",
    714: "δ≈2.9 s(생성 모델 이전) — hotswap_resource_report.md, 런 없음. v33 로그의 hot-swap delta는 vision 뷰 0.08–1.08 s뿐",
    739: "YOLO11x 단독 ≈72 fps — **어느 리포트에도 없다**",
    743: "237 / 186 fps, peak 146, baseline 75–78 — best_so_far_report.md(**증발한 q4_bsf_out**). v33 재실측은 231.0 / 176.1 / 480.3 / 298–323 으로 등재됨",
    745: "2-gen 사례 51–59 — 런 없음",
    747: "2-gen 사례 25 / 54–60 / 26 / 40 — 런 없음",
    752: "봉투 24–26 / 23 — analytic, 런 아님",
    773: "NPU 경합 7 / 34–49 / 4.4× — q5_experiment_report.md §2.2, **λ=80 사전측정**(본 실험은 λ=45), 런 없음",
    775: "δ≈2.9 s — L714와 동일",
    581: "ΔCPU 0.18 / 0.12 pp — q11_*_cpu 두 항목의 차. 원값 9건은 L579에서 추적 중",
    815: "3.3 — q6_b_lastV 3.2 와 q6_c_lastV 3.4 의 중간값. 어느 셀에도 없다",
    644: "축적 오차 0.2% — c3_revalidation_report.md 에 있으나 확정표 미등재",
    646: "앵커 오차 −6.8% / +12.5% — 같음",
    648: "λ/μ*_eff = 1.02 — 같음",
    686: "배율 3.3–4.7× — 확정표 두 항목의 비",
    688: "26% tail 감소, 구성비 8:92 — 같음",
    684: "1.5 는 β_B 설정, 0.4 는 tail 평탄성 백분율",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="paper")
    a = ap.parse_args()
    out = subprocess.check_output(
        [sys.executable, os.path.join(PROJECT, "scripts", "check_paper.py"),
         "--paper", a.paper, "--list-untracked"], text=True, cwd=PROJECT)
    lines, mode = out.splitlines(), None
    per_line = collections.OrderedDict()
    for ln in lines:
        if ln.startswith("[untracked lines]"):
            mode = "plain"; continue
        if ln.startswith("[annotated lines"):
            mode = "annot"; continue
        if not ln.startswith("  main.tex:"):
            continue
        loc, nums = ln.strip().split("  ", 1)
        first = int(loc.split(":")[1].split("-")[0])
        per_line.setdefault(first, {"n": 0, "mode": mode, "nums": []})
        per_line[first]["n"] += len(nums.split(","))
        per_line[first]["nums"] += [x.strip() for x in nums.split(",")]

    # 아직 정본에 반영되지 않은 배포본에 대고 돌리면, 곧 주석이 붙을 줄들이 미분류로 나온다.
    # 목록에서 읽어 자동으로 표시한다 — 손으로 적으면 목록과 갈라진다.
    wl = json.load(open(os.path.join(PROJECT, "docs", "paper_cv_annotations.json"),
                        encoding="utf-8"))
    pending = {it["line"] for it in wl["annotations"]}

    tally = collections.Counter()
    unclassified = []
    for line, info in per_line.items():
        cls = LINE_CLASS.get(line) or ("worklist" if line in pending else None)
        if cls is None:
            unclassified.append((line, info["nums"]))
            cls = "unclassified"
        tally[cls] += info["n"]

    total = sum(tally.values())
    print(f"{total} untracked numerals across {len(per_line)} lines\n")
    for cls, why in CLASSES.items():
        if tally[cls]:
            print(f"{tally[cls]:4d}  {cls:14s} {why}")
    if tally["unclassified"]:
        print(f"{tally['unclassified']:4d}  unclassified   ← 이 스크립트의 지도가 낡았다")
        for line, nums in unclassified:
            print(f"        main.tex:{line}  {', '.join(nums)}")

    print("\n-- 되짚을 수 없는 값 (실질적 잔여 위험) --")
    for line, info in per_line.items():
        if LINE_CLASS.get(line) == "unresolvable":
            print(f"  main.tex:{line:4d}  {info['n']:2d}개  {NOTE.get(line, '')}")
    return 1 if tally["unclassified"] else 0


if __name__ == "__main__":
    sys.exit(main())
