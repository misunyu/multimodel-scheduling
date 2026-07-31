#!/usr/bin/env python3
"""Register the v39 measurements: NPU transition peaks, YOLO11x sustained rate, churn at n=8.

Values come from the analysis modules rather than being typed here, so the report, the
confirmed table and the console tables cannot drift apart. `source` names a run directory.

Two entries are UPDATED rather than added: hs_{bg,a}_final_rss were registered at n=3 in v38
and this round takes them to n=8. That is the same quantity measured better, not a new one --
keeping both would leave two numbers for one fact, which is the failure mode v37 spent a
round cleaning up. The superseded values are recorded in each entry's note.

usage: python3 scripts/emit_v39_confirmed.py [--dry-run]
"""
import argparse, csv, collections, datetime, glob, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_hotswap_mem as A

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIRMED = os.path.join(PROJECT, "fig", "confirmed_values.json")
NPU_SRC = ("runs/ (v39 CPU--NPU hot-swap memory campaign, 6 runs) e.g. "
           "20260731T053447Z_npumemBG_r0")
PROBE_SRC = "runs/20260731T055603Z_probe_yolo11x_r0"
CHURN_SRC = ("runs/ (v38+v39 GPU hot-swap memory campaign, 16 runs) e.g. "
             "20260731T044450Z_memBG_r0")


def e(ident, value, unit, quantity, *, sd=None, n=1, note="", kind="hardware",
      source=NPU_SRC, reference="commit"):
    return {"id": ident, "value": value, "sd": sd, "unit": unit, "quantity": quantity,
            "reference": reference, "n": n, "censored": False, "recovered": None,
            "window_s": None, "source": source, "source_kind": kind, "note": note}


def npu_entries():
    res = [A.analyse(d) for d in sorted(glob.glob(os.path.join(PROJECT, "runs",
                                                               "*_npumem*_r[0-2]")))]
    if len(res) != 6:
        raise SystemExit(f"expected 6 NPU runs, found {len(res)}")
    s = A.summarise(res)
    host, dram = A._host_ram_mib(), A._npu_capacity_mib()
    out = []
    for kind, tag, human in (("generative", "gen", "생성 모델 이전 (kill-then-start)"),
                             ("vision", "vision", "vision in-place hot-swap")):
        k = s["kinds"][kind]
        for f, short, cap, dev in (("rss_mib", "rss", host, "호스트 RAM"),
                                   ("npu_ours_mib", "dram", dram, "NPU DRAM")):
            if f not in k:
                continue
            v = k[f]
            thin = ("  **n=2 로 얇다** — Q5 는 cand_2 에서 회복해 순수 vision 전환이 드물다."
                    if v["n"] < 3 else "")
            out.append(e(f"hs_npu_{tag}_{short}_incr", v["incr_mean"], "MiB", short,
                         sd=v["incr_sd"], n=v["n"],
                         note=f"CPU–NPU, {human}, {dev}: 전환 창의 최대 − 인접 정상 상태의 큰 쪽. "
                              f"범위 {v['incr_min']}~{v['incr_max']} MiB.{thin}"))
            out.append(e(f"hs_npu_{tag}_{short}_incr_max", v["incr_max"], "MiB", short,
                         sd=0.0, n=v["n"],
                         note=f"CPU–NPU, {human}, {dev}: 최악 전환 {v['worst_transition']} "
                              f"({v['worst_run']})"))
            out.append(e(f"hs_npu_{tag}_{short}_hold", v["hold_mean"], "s", "hold",
                         n=v["n"],
                         note=f"CPU–NPU, {human}, {dev}: 피크의 95 % 이상이 유지된 시간. "
                              f"최대 {v['hold_max']} s"))
            if v["incr_max"] > 0:
                head = (cap - (v["worst_base"] or 0)) / v["incr_max"]
                out.append(e(f"hs_npu_{tag}_{short}_headroom", round(head, 1), "x", "headroom",
                             sd=0.0, n=v["n"],
                             note=f"CPU–NPU, {human}, {dev}: (가용량 {cap} − 최악 전환의 정상 "
                                  f"상태 {v['worst_base']}) / 증분 {v['incr_max']}"))
    p = s["peaks"]
    out.append(e("hs_npu_peak_rss", p["rss_mib"]["max"], "MiB", "rss",
                 sd=p["rss_mib"]["sd"], n=6,
                 note="CPU–NPU 6런 중 전 구간 최대 host RSS(프로세스 트리 합, 공유 페이지 중복 포함)"))
    out.append(e("hs_npu_peak_dram", p["npu_ours_mib"]["max"], "MiB", "vram",
                 sd=p["npu_ours_mib"]["sd"], n=6,
                 note="CPU–NPU 6런 중 전 구간 최대 NPU DRAM(우리 PID 귀속분). "
                      "정상 상태는 vision 3종만 52 MiB, LLaMA-1B 상주 시 1918 MiB"))
    out.append(e("hs_npu_dram_total", dram, "MiB", "vram", sd=0.0, n=1, reference="window",
                 note="Aries(aries0) 총 DRAM. mblt-status 가 보고하는 값"))
    return out


def probe_entry():
    vals = []
    for d in sorted(glob.glob(os.path.join(PROJECT, "runs", "*_probe_yolo11x_r*"))):
        met = [datetime.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
               for r in csv.DictReader(open(os.path.join(d, "metrics.csv")))]
        lo, hi = min(met), max(met)
        ts = sorted(float(r["ts"]) for r in
                    csv.DictReader(open(os.path.join(d, "perframe.csv")))
                    if r["view"] == "view1")
        c = collections.Counter(int(t - lo) for t in ts if lo <= t <= hi)
        # drop 5 s of warmup; the rest is the saturated steady state
        vals.append(st.mean([c[x] for x in range(5, int(hi - lo))]))
    if len(vals) < 3:
        raise SystemExit(f"expected 3 probe runs, found {len(vals)}")
    return [e("q4_yolo11x_sustained", round(st.mean(vals), 1), "fps", "throughput",
              sd=round(st.pstdev(vals), 1), n=len(vals), source=PROBE_SRC, reference="window",
              note="YOLO11x 단독, GPU, 파이프라인 완료 프레임 수로 측정(λ=400 으로 포화, "
                   f"워밍업 5 s 제외). 런별 {[round(v, 1) for v in sorted(vals)]}. "
                   "Table~I 의 13.07 ms(=76.5 fps)는 격리 추론 지연이며 파이프라인이 지탱하는 "
                   "값이 아니다. Q4 의 λ=90 은 이 값의 1.46배")]


def churn_updates():
    fin = {"bg": [], "a": []}
    for d in sorted(glob.glob(os.path.join(PROJECT, "runs", "*_memBG_r[0-7]")) +
                    glob.glob(os.path.join(PROJECT, "runs", "*_memA_r[0-7]"))):
        if not os.path.isfile(os.path.join(d, "mem.csv")):
            continue
        r = A.analyse(d)
        fin["a" if "_memA_" in d else "bg"].append(r["rss_floor_sequence"][-1])
    a, b = fin["a"], fin["bg"]
    sp = ((st.pstdev(a) ** 2 + st.pstdev(b) ** 2) / 2) ** 0.5
    diff = st.mean(a) - st.mean(b)
    overlap = not (max(a) < min(b) or max(b) < min(a))
    sep = (not overlap) and abs(diff) > 2 * sp
    verdict = ("분리됨" if sep else "**분리되지 않음** — 범위가 겹치고 평균차가 2·s_p 미만")
    old = {"bg": "7701.0 ± 265.5 (n=3, v38)", "a": "8073.4 ± 96.8 (n=3, v38)"}
    out = []
    for tag, label, vals in (("bg", "BoundGuard, 전환 6회", b), ("a", "변형 A(no-dwell), 전환 7회", a)):
        out.append(e(f"hs_{tag}_final_rss", round(st.mean(vals), 1), "MiB", "rss",
                     sd=round(st.pstdev(vals), 1), n=len(vals), source=CHURN_SRC,
                     reference="window",
                     note=f"{label}: 마지막 전환 이후의 정상 상태 RSS. v38 의 n=3 값 {old[tag]} 을 "
                          f"**대체**한다(같은 양, 표본 보강). 사전 고정 판정(docs/v39_conditions.md "
                          f"작업 208): {verdict}. 평균차 {diff:.1f} MiB, 2·s_p {2 * sp:.1f} MiB"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    doc = json.load(open(CONFIRMED, encoding="utf-8"))
    by_id = {x["id"]: i for i, x in enumerate(doc["values"])}
    new = npu_entries() + probe_entry()
    upd = churn_updates()

    clash = sorted(x["id"] for x in new if x["id"] in by_id)
    if clash:
        print(f"[FAIL] already registered: {clash}")
        return 1
    missing = [x["id"] for x in upd if x["id"] not in by_id]
    if missing:
        print(f"[FAIL] meant to update but not present: {missing}")
        return 1

    print(f"{len(new)} new entries, {len(upd)} updated in place")
    if a.dry_run:
        for x in new + upd:
            mark = "UPD" if x["id"] in by_id else "new"
            print(f"  [{mark}] {x['id']:30s} {str(x['value']):>9} {x['unit']:<4} "
                  f"n={x['n']:<2} sd={x['sd']}")
        return 0
    for x in upd:
        doc["values"][by_id[x["id"]]] = x
    doc["values"] += new
    json.dump(doc, open(CONFIRMED, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {CONFIRMED} ({len(doc['values'])} entries total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
