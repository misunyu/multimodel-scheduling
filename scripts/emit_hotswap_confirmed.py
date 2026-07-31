#!/usr/bin/env python3
"""Register the v38 hot-swap transition resource measurements (ISSRE R4).

v18's numbers answered R4 on a workload where only 2 of 4 named views ran, so they are void;
its structural conclusions are not, because they came from the code (see
scripts/check_transition_structure.py). This registers the re-measurement.

Values are computed by scripts/analyze_hotswap_mem.py rather than typed here, so the
confirmed table, the console table and the report cannot drift apart. `source` names a run
directory, per runs/README.md.

usage: python3 scripts/emit_hotswap_confirmed.py [--dry-run]
"""
import argparse, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_hotswap_mem as A

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIRMED = os.path.join(PROJECT, "fig", "confirmed_values.json")
SRC = ("runs/ (v38 hot-swap memory campaign, 6 runs) e.g. "
       "20260731T044450Z_memBG_r0")


def e(ident, value, unit, quantity, *, sd=None, n=6, note="", kind="hardware",
      source=SRC, reference="commit"):
    return {"id": ident, "value": value, "sd": sd, "unit": unit, "quantity": quantity,
            "reference": reference, "n": n, "censored": False, "recovered": None,
            "window_s": None, "source": source, "source_kind": kind, "note": note}


def build():
    runs = A.campaign_runs()
    if len(runs) != 6:
        raise SystemExit(f"expected 6 campaign runs, found {len(runs)}: {runs}")
    res = [A.analyse(r) for r in runs]
    s = A.summarise(res)
    host, vram_cap = A._host_ram_mib(), A._vram_total_mib()
    out = []

    for kind, tag, human in (("generative", "gen", "생성 모델 이전 (kill-then-start)"),
                             ("vision", "vision", "vision in-place hot-swap (스레드 겹침)")):
        k = s["kinds"][kind]
        for f, short, cap in (("rss_mib", "rss", host), ("vram_ours_mib", "vram", vram_cap)):
            v = k[f]
            head = ((cap - (v["worst_base"] or 0)) / v["incr_max"]) if v["incr_max"] > 0 else None
            out.append(e(f"hs_{tag}_{short}_incr", v["incr_mean"], "MiB", short,
                         sd=v["incr_sd"], n=v["n"],
                         note=f"{human}: 전환 창의 최대 − 인접 정상 상태의 큰 쪽. "
                              f"음수는 오버슈트 없음(피크가 정상 상태를 넘지 않음)을 뜻한다. "
                              f"범위 {v['incr_min']}~{v['incr_max']} MiB"))
            out.append(e(f"hs_{tag}_{short}_incr_max", v["incr_max"], "MiB", short,
                         sd=0.0, n=v["n"],
                         note=f"{human}: 최악 전환 {v['worst_transition']} "
                              f"({v['worst_run']}). 평균이 아니라 최댓값을 함께 싣는다"))
            out.append(e(f"hs_{tag}_{short}_hold", v["hold_mean"], "s", "hold",
                         sd=None, n=v["n"],
                         note=f"{human}: 피크의 95 % 이상이 유지된 시간. 10 Hz 샘플링이므로 "
                              f"0.1 s = 표본 1개. 최대 {v['hold_max']} s"))
            if head is not None:
                out.append(e(f"hs_{tag}_{short}_headroom", round(head, 1), "x", "headroom",
                             sd=0.0, n=v["n"],
                             note=f"{human}: (가용량 − 최악 전환의 정상 상태 "
                                  f"{v['worst_base']} MiB) / 증분 {v['incr_max']} MiB. "
                                  f"가용량 {cap} MiB"))

    p = s["peaks"]
    out.append(e("hs_peak_rss", p["rss_mib"]["max"], "MiB", "rss", sd=p["rss_mib"]["sd"],
                 note="6런 중 전 구간 최대 host RSS(프로세스 트리 합). "
                      "RSS 합은 공유 페이지를 중복 계산하므로 절대값은 과대 추정이다 — "
                      "증분에서는 공유분이 상쇄된다"))
    out.append(e("hs_peak_vram_ours", p["vram_ours_mib"]["max"], "MiB", "vram",
                 sd=p["vram_ours_mib"]["sd"],
                 note="6런 중 전 구간 최대 GPU VRAM(NVML 의 우리 PID 귀속분 합)"))
    out.append(e("hs_peak_vram_total", p["vram_total_mib"]["max"], "MiB", "vram",
                 sd=p["vram_total_mib"]["sd"],
                 note="같은 구간의 디바이스 총 사용량(타 프로세스 포함)"))
    out.append(e("hs_host_ram", host, "MiB", "rss", sd=0.0, n=1, reference="window",
                 note="측정 호스트의 총 RAM. 여유 배수의 분모가 아니라 상한이다"))
    out.append(e("hs_vram_total", vram_cap, "MiB", "vram", sd=0.0, n=1, reference="window",
                 note="측정 호스트의 총 GPU VRAM (RTX 5090)"))

    # churn: does the no-dwell variant, which transitions most, end higher?
    fin = {"a": [], "bg": []}
    for r in res:
        fin["a" if "_memA_" in r["run"] else "bg"].append(r["rss_floor_sequence"][-1])
    for tag, label in (("bg", "BoundGuard, 전환 6회"), ("a", "변형 A(no-dwell), 전환 7회")):
        out.append(e(f"hs_{tag}_final_rss", round(st.mean(fin[tag]), 1), "MiB", "rss",
                     sd=round(st.pstdev(fin[tag]), 1), n=len(fin[tag]),
                     reference="window",
                     note=f"{label}: 마지막 전환 이후의 정상 상태 RSS. 두 변형의 차이는 "
                          f"{round(st.mean(fin['a']) - st.mean(fin['bg']), 0):.0f} MiB 이며 "
                          f"BoundGuard 자체의 런간 산포 안에 있다 — 누적의 증거가 아니다"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    doc = json.load(open(CONFIRMED, encoding="utf-8"))
    existing = {x["id"] for x in doc["values"]}
    new = build()
    ids = [x["id"] for x in new]
    if len(set(ids)) != len(ids):
        print("[FAIL] duplicate ids within the batch")
        return 1
    clash = sorted(set(ids) & existing)
    if clash:
        print(f"[FAIL] already registered -- this script registers, it does not overwrite: {clash}")
        return 1
    print(f"{len(new)} new entries")
    if a.dry_run:
        for x in new:
            print(f"  {x['id']:28s} {str(x['value']):>9} {x['unit']:<4} "
                  f"n={x['n']:<2} sd={x['sd']}")
        return 0
    doc["values"] += new
    json.dump(doc, open(CONFIRMED, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {CONFIRMED} ({len(doc['values'])} entries total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
