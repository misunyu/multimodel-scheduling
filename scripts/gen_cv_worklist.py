#!/usr/bin/env python3
"""Emit the `%!cv` annotation worklist for the canonical main.tex.

The annotations cannot be applied here: paper/main.tex is a distribution copy and editing it
breaks its hash by design. So the repo produces a worklist instead, and the edit happens on
the canonical copy outside. This script writes both forms of it:

  docs/paper_cv_annotations.json   machine-readable; scripts/apply_cv_annotations.py consumes it
  docs/paper_cv_annotation_worklist.md   the same thing to read

Each item carries the sha256 of the line it targets. Line numbers shift the moment anyone
edits the paper, and applying an annotation to the wrong line would attach a value to a
sentence that never claimed it -- a silent, plausible-looking corruption. The hash makes that
a hard failure instead.

The list is curated by hand, not generated: scripts/propose_cv_annotations.py matches by
value alone and cheerfully pairs MobileNet-v2's 14 ms with a 14.0 s recovery time. What is
curated is which numerals are worth tracking; the values themselves are read from the
confirmed table here so this file cannot carry a stale copy of them.

usage: python3 scripts/gen_cv_worklist.py
"""
import hashlib, json, os, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(PROJECT, "paper", "main.tex")
CONFIRMED = os.path.join(PROJECT, "fig", "confirmed_values.json")
OUT_JSON = os.path.join(PROJECT, "docs", "paper_cv_annotations.json")
OUT_MD = os.path.join(PROJECT, "docs", "paper_cv_annotation_worklist.md")

# (line, what the sentence claims, ids)   -- section IV prose first, per the priority order:
# prose ahead of captions ahead of cross-section citations ahead of table cells, because a
# table is replaced wholesale when it is updated and a sentence is not.
WORKLIST = [
    # (anchor, what the sentence claims, ids)
    #
    # Entries are keyed by a distinctive SUBSTRING of the target sentence, not by line
    # number. A revision that inserts one paragraph shifts every later line, and the sha256
    # guard cannot catch that: gen_cv_worklist re-reads the current main.tex, so it would
    # cheerfully hash whatever now sits at the old number. That happened -- a paragraph added
    # to Section III moved eighteen targets down by one line, and every hash still "matched".
    # The anchor must resolve to exactly one line or generation fails.
    ("We measured what it costs",
     "§III — what a hot-swap costs in memory (v38/v39)",
     ["hs_vision_vram_incr_max", "hs_gen_rss_incr_max", "hs_gen_rss_headroom"]),
    ("Recovery completes in",
     "Q1 — recovery totals, lead, and the part of the lead that precedes activation",
     ["q3_bg_persist", "q5_bg_persist", "q3_lead", "q5_lead", "q5_lead_pre_activation"]),
    ("Throughput is identical to the arrival rate",
     "Q1.1 — steady-state throughput, inference latency, CPU occupancy",
     ["q11_gpu_boundguard_throughput", "q11_gpu_boundguard_infer_ms",
      "q11_gpu_static_infer_ms", "q11_npu_adaptive_infer_ms",
      "q11_npu_boundguard_infer_ms", "q11_gpu_boundguard_cpu", "q11_gpu_static_cpu",
      "q11_npu_static_cpu", "q11_npu_adaptive_cpu"]),
    ("No setting produces a false detection",
     "Q1.2 — detection lag and recovery latency across the window sweep",
     ["q12_detect_lat_T1_eps1.0", "q12_detect_lat_T3_eps1.0", "q12_detect_lat_T5_eps1.0",
      "q12_recover_lat_T1_eps1.0", "q12_recover_lat_T5_eps1.0"]),
    ("among the admissible windows",
     "Q1.2 — the T=1 vs T=3 comparison the section's conclusion turns on",
     ["q12_detect_lat_T1_eps1.0", "q12_detect_lat_T3_eps1.0",
      "q12_recover_lat_T1_eps1.0", "q12_recover_lat_T3_eps1.0"]),
    ("Static is the outlier",
     "Q1.3 — Static terminal V, cumulative violation, persistence range, ratios",
     ["q13_static_lastV", "q13_gpu_static_cumV", "q13_npu_static_cumV",
      "q13_gpu_bg_persist", "q13_gpu_sr_persist", "q13_gpu_bg_cumV", "q13_gpu_sr_cumV",
      "q13_gpu_cumV_ratio", "q13_npu_cumV_ratio"]),
    ("We do not read the small differences",
     "Q1.3 — the search that is lead, and the no-dwell variant's counts",
     ["q13_gpu_bg_search", "q4b_a_recoveries", "q4b_a_hotswaps", "q4b_bg_hotswaps"]),
    ("The sweep separates the model's two halves",
     "Q1.4 — sweep regression slopes against the predicted 1.0",
     ["c3_gpu_dslope_dlambda", "c3_npu_dslope_dlambda"]),
    ("Inverting the measurements for the service rate",
     "Q1.4 — the three service rates bracketing the effective drain rate",
     ["c3_gpu_mu_star_sat", "c3_npu_mu_star_sat", "c3_gpu_mu_star_raw",
      "c3_npu_mu_star_raw", "c3_gpu_mu_star_eff", "c3_npu_mu_star_eff"]),
    ("enlarging the buffer does not improve deadline attainment",
     "Q1.5 — buffer sweep: drops fall, miss rate barely moves",
     ["q15_gpu_static_b0.5_drop", "q15_gpu_static_b1.5_drop",
      "q15_gpu_static_b0.5_miss", "q15_gpu_static_b1.5_miss"]),
    ("both metrics separate the methods",
     "Q1.5 — the reconfigure-vs-stay-put division",
     ["q15_gpu_static_b0.5_miss", "q15_gpu_static_b0.5_p99",
      "q15_npu_adaptive_b0.5_p99", "q15_gpu_stoprestart_b0.5_p99"]),
    ("Stop-and-restart reaches",
     "Q1.5 — stop-restart vs in-place tail",
     ["q15_gpu_stoprestart_b0.5_p99", "q15_gpu_boundguard_b0.5_p99",
      "q15_gpu_adaptive_b0.5_p99"]),
    ("This scenario and the prediction-match control",
     "Q2 / prediction-match — the recovery times that are indistinguishable",
     ["q4b_adaptive_persist", "q4b_c_persist"]),
    ("stay in violation for the remainder of the observation window",
     "Q3 — the level the baselines sit at while in violation (peak, not terminal)",
     ["q3_static_maxV"]),
    ("BoundGuard instead treats each ranked placement",
     "Q3 — persist, search, drain, lead, envelope, baseline terminal V",
     ["q3_bg_persist", "q3_bg_search", "q3_bg_drain", "q3_lead", "q3_bound",
      "q3_baseline_lastV"]),
    ("Figure~\\ref{fig:bounded} shows the outcome",
     "Q4 — terminal and peak violation across the five variants "
     "(CORRECTS the annotation now in the distribution copy, which names q4_b_lastV: "
     "294.2 lies inside the printed range 293--315 and so appears nowhere on the line)",
     ["q4_bg_lastV", "q4_c_lastV", "q4_adaptive_lastV", "q4_bg_maxV", "q4_b_maxV",
      "q4_adaptive_maxV"]),
    ("Figure~\\ref{fig:npu_violation} shows the outcome",
     "Q5 — persist, lead, the pre-activation part of the lead, baseline terminal V",
     ["q5_bg_persist", "q5_lead", "q5_lead_pre_activation", "q5_baseline_lastV",
      "q5_adaptive_lastV", "q5_sr_lastV"]),
    ("Removing dwell fails differently",
     "Q6 — variant A's terminal violation, integrated violation, and transition counts",
     ["q6_a_lastV", "q6_a_cumV", "q6_adaptive_cumV", "q6_a_hotswaps", "q6_bg_hotswaps",
      "q4b_a_recoveries", "q4b_a_hotswaps"]),
]


def main():
    lines = open(TEX, encoding="utf-8").read().splitlines()
    table = {e["id"]: e for e in json.load(open(CONFIRMED, encoding="utf-8"))["values"]}
    items, missing, unresolved = [], [], []
    for anchor, what, ids in WORKLIST:
        hits = [i + 1 for i, l in enumerate(lines) if anchor in l]
        if len(hits) != 1:
            unresolved.append((anchor, hits))
            continue
        ln = hits[0]
        text = lines[ln - 1]
        for i in ids:
            if i not in table:
                missing.append(i)
        items.append({
            "line": ln,
            "anchor": anchor,
            "what": what,
            "ids": ids,
            "annotation": "%!cv " + ",".join(ids),
            "line_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "line_excerpt": (text[:90] + "…") if len(text) > 90 else text,
            "values": {i: {"value": table[i]["value"], "unit": table[i].get("unit")}
                       for i in ids if i in table},
        })
    if unresolved:
        print("[FAIL] anchors that did not resolve to exactly one line:")
        for anchor, hits in unresolved:
            print(f"    {len(hits)} hits: {anchor!r}")
        return 1
    if missing:
        print(f"[FAIL] worklist names ids that are not in the confirmed table: {sorted(set(missing))}")
        return 1

    doc = {"_meta": {
        "note": "논문 정본(저장소 밖)에 붙일 %!cv 주석 목록. 배포본 paper/main.tex 은 편집하지 않는다.",
        "target_main_tex_sha256": hashlib.sha256(open(TEX, "rb").read()).hexdigest(),
        "apply_with": "python3 scripts/apply_cv_annotations.py --tex <정본 경로>",
        "verify_with": "주석을 붙인 새 main.tex 와 새 paper.lock 을 받은 뒤 `make check`",
        "line_sha256": "대상 줄의 sha256. 줄 번호는 편집으로 밀리므로 해시로 고정한다.",
    }, "annotations": items}
    json.dump(doc, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n_ids = sum(len(i["ids"]) for i in items)
    n_uniq = len({i for it in items for i in it["ids"]})
    md = [
        "# `%!cv` 주석 부착 목록 (v36 작업 190 · v37 작업 196 확장)",
        "",
        "**이 목록은 저장소 밖 정본에 적용한다.** `paper/main.tex` 는 배포본이며 편집하면",
        "`scripts/check_paper.py` 의 해시 검사가 실패한다 — 그것이 의도된 동작이다.",
        "",
        f"주석 {len(items)}개 · 추적 값 {n_ids}건(고유 {n_uniq}건).",
        f"대상 `main.tex` sha256 `{doc['_meta']['target_main_tex_sha256'][:16]}…`",
        "",
        "적용:",
        "",
        "```bash",
        "python3 scripts/apply_cv_annotations.py --tex /경로/정본/main.tex",
        "```",
        "",
        "각 항목의 `line_sha256` 가 대상 줄과 일치할 때만 붙는다. 줄 번호는 편집으로 밀리고,",
        "엉뚱한 줄에 붙은 주석은 **그 문장이 하지도 않은 주장에 값을 매다는** 조용한 오염이 된다.",
        "",
        "---",
        "",
    ]
    for it in items:
        md += [f"### L{it['line']} — {it['what']}", "",
               "```", it["line_excerpt"], "```", "",
               "줄 끝에 붙일 것:", "", "```latex", it["annotation"], "```", "",
               "| id | 확정표 값 |", "|---|---|"]
        for i in it["ids"]:
            v = it["values"][i]
            md += [f"| `{i}` | {v['value']}{' ' + v['unit'] if v['unit'] else ''} |"]
        md += [""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print(f"wrote {OUT_JSON} and {OUT_MD}: {len(items)} annotations, "
          f"{n_ids} tracked values ({n_uniq} unique ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
