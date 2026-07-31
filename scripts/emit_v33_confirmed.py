#!/usr/bin/env python3
"""Register the v33 campaign values the paper cites but the confirmed table never carried.

`check_paper.py` found roughly forty numerals in section IV with no confirmed entry. They
were not missing measurements -- `docs/main_rerun_report.md` tabulates most of them -- they
were simply never promoted into `fig/confirmed_values.json`. The cost of that has already
been paid once: Q4's baseline comparison sentence had to be DELETED rather than corrected,
because the re-run values it needed were not registerable.

Two rules decide what goes in:

  1. `source` must name a run directory (runs/README.md). A value whose only provenance is a
     report is a value whose audit chain is already broken -- the report is a derivative.
     Anything that traces only to a lost run stays OUT and is listed in the report instead.
  2. Values are TRANSCRIBED from the report, not recomputed. Where the report has no entry
     for a quantity the paper cites (peak violation, per-candidate service rates), the value
     is recomputed from the committed runs and the entry says so in its `note`, so the two
     provenances are never confused.

`scripts/aggregate_v33_campaign.py` reproduces the report's tables from the runs; it agrees
to within 2 V*s on cumulative violation (an integration-endpoint convention) and exactly
elsewhere. That agreement is what makes `source` more than a label.

usage: python3 scripts/emit_v33_confirmed.py [--dry-run]
"""
import argparse, json, os, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIRMED = os.path.join(PROJECT, "fig", "confirmed_values.json")

REPORT = "docs/main_rerun_report.md §1 표에서 전사"
RECOMP = ("리포트 표에 없는 양 — scripts/aggregate_v33_campaign.py 가 커밋된 런에서 재계산 "
          "(고정 94 s 창)")

SRC = {  # scenario -> (source string, example run)
    "q4": "runs/ (v33 q4, 25 runs) e.g. 20260729T070951Z_q4_BoundGuard_r0",
    "q6": "runs/ (v33 q6, 25 runs) e.g. 20260729T045546Z_q6_BoundGuard_r0",
    "q4b": "runs/ (v33 q4b, 25 runs) e.g. 20260729T061851Z_q4b_BoundGuard_r0",
    "q3": "runs/ (v33 q3, 20 runs) e.g. 20260729T013710Z_q3_BoundGuard_r0",
    "q5": "runs/ (v33 q5, 20 runs) e.g. 20260729T060516Z_q5_BoundGuard_r0",
}


def e(ident, value, unit, quantity, scenario, *, sd=None, n=5, censored=False,
      recovered=None, window_s=94, kind="hardware", note=REPORT, source=None):
    return {"id": ident, "value": value, "sd": sd, "unit": unit, "quantity": quantity,
            "reference": "t0", "n": n, "censored": censored, "recovered": recovered,
            "window_s": window_s, "source": source or SRC[scenario],
            "source_kind": kind, "note": note}


def build():
    out = []

    # ---- Q4: no admissible placement. Every variant ends in violation, so lastV/maxV are
    # censored and recovered is false for all five. The paper's deleted sentence compared
    # BoundGuard against the baselines; these are the entries that make it writable again.
    q4 = {  # variant: (lastV, sd, cumV, sd, hotswaps, maxV, sd)
        "bg":       (362.5, 112.2, 17059, 250, 9, 480.3, 46.4),
        "a":        (352.1, 130.5, 15410, 2411, 9, 670.8, 148.4),
        "b":        (294.2, 2.8, 22367, 226, 0, 298.2, 3.0),
        "c":        (292.6, 3.7, 17724, 184, 0, 298.7, 3.2),
        "adaptive": (315.1, 10.7, 23746, 837, 0, 322.8, 10.9),
    }
    for v, (lv, lsd, cv, csd, hs, mx, msd) in q4.items():
        if v != "bg":  # q4_bg_lastV is already registered
            out.append(e(f"q4_{v}_lastV", lv, "V", "lastV", "q4", sd=lsd,
                         censored=True, recovered=False))
        out.append(e(f"q4_{v}_cumV", cv, "V*s", "cumV", "q4", sd=csd,
                     censored=True, recovered=False))
        out.append(e(f"q4_{v}_maxV", mx, "V", "maxV", "q4", sd=msd,
                     censored=False, recovered=False, note=RECOMP))
        out.append(e(f"q4_{v}_hotswaps", hs, "swaps", "count", "q4", sd=0.0,
                     recovered=False))
    # Rejection accounting. The paper attributes 45 rejections to BoundGuard's five runs;
    # 45 is the total over BoundGuard AND variant C (10 runs). Both are registered so the
    # distinction is stateable rather than arguable.
    out.append(e("q4_bg_saturated_advances", 20, "events", "count", "q4", sd=0.0,
                 recovered=False,
                 note="BoundGuard 5런의 QoS-validate 20건이 전부 saturation guard 경유. "
                      "executor.log 집계 (main_rerun_report.md §3 는 변형 합산 45를 싣는다)."))
    out.append(e("q4_saturated_advances_all", 45, "events", "count", "q4", sd=0.0, n=10,
                 recovered=False,
                 note="BoundGuard 20 + 변형 C 25 = 45, 10런 합산. main_rerun_report.md §3."))
    # Per-candidate service rates measured during validation. The published 237.3/186.2 come
    # from `q4_bsf_out`, one of the evaporated runs; these are the v33 re-run's own.
    out.append(e("q4_cand_service_rate_best", 231.0, "fps", "throughput", "q4", sd=2.5,
                 recovered=False,
                 note="cand_2 (가장 무거운 모델을 가속기에서 뺀 후보). executor.log 의 "
                      "QoS-validate service_rate 5런 평균. 게재본 237.3 은 증발한 q4_bsf_out 유래."))
    out.append(e("q4_cand_service_rate_worst", 176.1, "fps", "throughput", "q4", sd=4.0,
                 recovered=False,
                 note="cand_5 (예측기가 마지막으로 순위매긴 후보). 게재본 186.2 는 q4_bsf_out 유래."))

    # ---- Q6: the 2x2 ablation. lastV/hotswaps/recoveries are already registered; the
    # cumulative violations the paper quotes (2226 vs 264) were not.
    q6 = {"bg": (303, 59, 46.5, 7.6, True), "a": (2226, 34, 52.6, 0.5, False),
          "b": (265, 2, 3.5, 0.1, False), "c": (216, 6, 3.5, 0.1, False),
          "adaptive": (264, 4, 3.5, 0.1, False)}
    for v, (cv, csd, mx, msd, rec) in q6.items():
        out.append(e(f"q6_{v}_cumV", cv, "V*s", "cumV", "q6", sd=csd,
                     censored=not rec, recovered=rec))
        out.append(e(f"q6_{v}_maxV", mx, "V", "maxV", "q6", sd=msd,
                     censored=False, recovered=rec, note=RECOMP))

    # ---- section 4b (prediction-match control)
    q4b = {"bg": (146, 6, 6), "a": (374, 183, 15), "b": (147, 5, 6),
           "c": (153, 10, 6), "adaptive": (146, 6, 6)}
    for v, (cv, csd, hs) in q4b.items():
        # Variant A recovers in 2 runs of 5, so `recovered` is not a property of the cell --
        # it is null rather than false, and its integral is censored because three of the
        # five never return below epsilon inside the window.
        rec = None if v == "a" else True
        out.append(e(f"q4b_{v}_cumV", cv, "V*s", "cumV", "q4b", sd=csd,
                     censored=(v == "a"), recovered=rec))
        out.append(e(f"q4b_{v}_hotswaps", hs, "swaps", "count", "q4b", sd=0.0,
                     recovered=rec))

    # ---- Q3: baselines never recover, so their terminal V is censored. maxV is a different
    # quantity from lastV and the paper cites both in adjacent sentences (~3.7-4.1 while in
    # violation, ~3.5 at the end); registering only one would let the other drift.
    q3 = {"bg": (0.0, 0.0, 280, 33, 46.8, 6.7, True),
          "adaptive": (3.5, 0.1, 292, 4, 3.9, 0.1, False),
          "sr": (3.4, 0.1, 289, 8, 3.8, 0.1, False),
          "static": (3.5, 0.1, 336, 6, 4.1, 0.1, False)}
    for v, (lv, lsd, cv, csd, mx, msd, rec) in q3.items():
        if v != "bg":
            out.append(e(f"q3_{v}_lastV", lv, "V", "lastV", "q3", sd=lsd,
                         censored=True, recovered=False))
        out.append(e(f"q3_{v}_cumV", cv, "V*s", "cumV", "q3", sd=csd,
                     censored=not rec, recovered=rec))
        out.append(e(f"q3_{v}_maxV", mx, "V", "maxV", "q3", sd=msd,
                     censored=False, recovered=rec, note=RECOMP))
    out.append(e("q3_bg_hotswaps", 6, "swaps", "count", "q3", sd=0.0, recovered=True))

    # ---- Q5
    q5 = {"bg": (0.0, 0.0, 382, 8, True), "adaptive": (26.7, 0.6, 2390, 30, False),
          "sr": (26.9, 0.6, 2395, 25, False), "static": (27.6, 1.3, 2544, 45, False)}
    for v, (lv, lsd, cv, csd, rec) in q5.items():
        if v != "bg":
            out.append(e(f"q5_{v}_lastV", lv, "V", "lastV", "q5", sd=lsd,
                         censored=True, recovered=False))
        out.append(e(f"q5_{v}_cumV", cv, "V*s", "cumV", "q5", sd=csd,
                     censored=not rec, recovered=rec))
    # The part of Q5's lead that precedes the controller being enabled: the stable phase is
    # already violating there, so no method -- ours or a baseline -- can act during it.
    out.append(e("q5_lead_pre_activation", 9.6, "s", "lead", "q5", sd=0.5, recovered=True,
                 source="runs/20260729T060516Z_q5_BoundGuard_r0",
                 note="q5_lead 22.4 s 중 컨트롤러 활성 이전 구간. "
                      "scripts/compute_bound.py 의 lead_pre_activation_s, 5런 [9,9,10,10,10]. "
                      "main_rerun_report.md §4."))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    doc = json.load(open(CONFIRMED, encoding="utf-8"))
    existing = {x["id"] for x in doc["values"]}
    new = build()
    ids = [x["id"] for x in new]
    dupe_in_batch = {i for i in ids if ids.count(i) > 1}
    if dupe_in_batch:
        print(f"[FAIL] duplicate ids within the batch: {sorted(dupe_in_batch)}")
        return 1
    clash = sorted(set(ids) & existing)
    if clash:
        print(f"[FAIL] these ids already exist -- v37 registers, it does not overwrite: {clash}")
        return 1

    print(f"{len(new)} new entries "
          f"({sum(1 for x in new if x['note'] == RECOMP)} recomputed, "
          f"{sum(1 for x in new if x['note'] == REPORT)} transcribed, "
          f"{sum(1 for x in new if x['note'] not in (RECOMP, REPORT))} with their own note)")
    if a.dry_run:
        for x in new:
            print(f"  {x['id']:32s} {str(x['value']):>9} {x['unit']:<6} "
                  f"cens={str(x['censored']):5s} rec={str(x['recovered'])}")
        return 0
    doc["values"] += new
    json.dump(doc, open(CONFIRMED, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {CONFIRMED} ({len(doc['values'])} entries total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
