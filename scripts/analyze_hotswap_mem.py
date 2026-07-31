#!/usr/bin/env python3
"""Extract transition-time resource peaks from the 10 Hz mem.csv of a v38 run.

R4 asks about the transient at a transition, so everything here is defined relative to the
phase boundaries, and the boundaries come from `metrics.csv`'s `combination` column -- never
from the resource trace itself, which would make the answer circular.

The transition window is the GAP in metrics.csv. That is not a convenience: the executor's
1 Hz sampler stalls while a placement is applied, so its series has a hole of several seconds
exactly across a generative relocation. The hole is the transition, and the 10 Hz observer is
what sees inside it -- which is the whole reason a 1 Hz measurement could not answer R4.

  steady_pre    median over the last 2 s of samples before the window
  peak          max inside the window
  steady_post   median over the first 2 s of samples after the window
  increment     peak - max(steady_pre, steady_post)
                the larger neighbour, so a transition that merely climbs to a higher new
                steady state reports ~0 rather than crediting the climb as overlap cost
  peak_hold_s   time the trace stays within 5 % of the peak

Transition KIND comes from the executor log segment between the two `Starting schedule`
markers: a `[bg-llm] start` line means a generative relocation (kill-then-start, so the cost
is a load transient rather than co-residence), `hot-swap complete` means a vision in-place
swap (thread-level overlap: the new worker is started and only then is the old one stopped).

usage: python3 scripts/analyze_hotswap_mem.py runs/<dir> [--json out.json]
       python3 scripts/analyze_hotswap_mem.py --all [--json out.json]
"""
import argparse, collections, csv, datetime, glob, json, os, re, statistics as st, sys


def _host_ram_mib():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1048576)
    except Exception:
        return None


def _npu_capacity_mib():
    """From the sampler's own column when a run carried it; the device is 16384 MiB."""
    try:
        import subprocess, re as _re
        out = subprocess.run(["mblt-status", "-q", "-s", "MEMORY"],
                             capture_output=True, text=True, timeout=5).stdout
        m = _re.search(r"Total\s*:\s*([0-9.]+)\s*MB", out)
        return round(float(m.group(1))) if m else None
    except Exception:
        return None


def _vram_total_mib():
    try:
        import pynvml
        pynvml.nvmlInit()
        return round(sum(pynvml.nvmlDeviceGetMemoryInfo(
            pynvml.nvmlDeviceGetHandleByIndex(i)).total
            for i in range(pynvml.nvmlDeviceGetCount())) / 1048576)
    except Exception:
        return None

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = "%Y-%m-%d %H:%M:%S"
METRICS = ("rss_mib", "vram_ours_mib", "vram_total_mib")
# v39: the CPU--NPU runs carry three more columns. Older mem.csv files do not, and the
# NPU ones leave them blank when mblt-status could not be read -- both are treated as absent
# rather than as zero, because "no NPU memory" and "could not ask" are different answers.
NPU_METRICS = ("npu_ours_mib", "npu_total_used_mib")
MARGIN_PRE, MARGIN_POST = 1.0, 2.0
STEADY_WIN = 2.0


def load(run):
    mem = list(csv.DictReader(open(os.path.join(run, "mem.csv"))))
    fields = list(METRICS)
    if mem and mem[0].get("npu_ours_mib") not in (None, ""):
        fields += list(NPU_METRICS)
    for r in mem:
        for k in ("epoch",) + tuple(fields):
            r[k] = float(r[k])
    for r in mem:
        r["_fields"] = fields
    met = []
    for r in csv.DictReader(open(os.path.join(run, "metrics.csv"))):
        try:
            met.append((datetime.datetime.strptime(r["timestamp"], TS).timestamp(),
                        r["combination"]))
        except Exception:
            continue
    return mem, met


def phase_spans(met):
    """[(combination, first_epoch, last_epoch)] in order."""
    out = []
    for ep, combo in met:
        if out and out[-1][0] == combo:
            out[-1][2] = ep
        else:
            out.append([combo, ep, ep])
    return [tuple(x) for x in out]


def log_segments(run):
    """combination -> the log text applied while starting it."""
    p = os.path.join(run, "executor.log")
    if not os.path.isfile(p):
        return {}
    segs, cur = {}, None
    for line in open(p, errors="replace"):
        m = re.search(r"\[Executor\] Starting schedule: (\S+)", line)
        if m:
            cur = m.group(1)
            segs[cur] = []
            continue
        if cur:
            segs[cur].append(line)
    return {k: "".join(v) for k, v in segs.items()}


def kind_of(seg):
    k = []
    if re.search(r"\[bg-llm\] (start|stop)", seg or ""):
        k.append("generative")
    if "hot-swap complete" in (seg or ""):
        k.append("vision")
    return "+".join(k) if k else "none"


def window_stats(mem, lo, hi, field):
    vals = [r[field] for r in mem if lo <= r["epoch"] <= hi]
    return vals


def analyse(run):
    mem, met = load(run), None
    mem, met = mem[0], mem[1]
    spans = phase_spans(met)
    segs = log_segments(run)
    t0 = mem[0]["epoch"]
    out = []
    last_w_hi = mem[0]["epoch"] - 1e9
    for i in range(1, len(spans)):
        prev_c, prev_lo, prev_hi = spans[i - 1]
        next_c, next_lo, next_hi = spans[i]
        # Windows must not overlap. Under the no-dwell variant the candidate phases are one
        # or two seconds long, so an unclipped window swallows the previous transition and
        # reports its peak again -- which looked like two transitions each costing 2 GB when
        # there was one. Clip to the end of the previous window.
        w_lo, w_hi = max(prev_hi - MARGIN_PRE, last_w_hi), next_lo + MARGIN_POST
        last_w_hi = w_hi
        rec = {"run": os.path.basename(run), "from": prev_c, "to": next_c,
               "kind": kind_of(segs.get(next_c)),
               "gap_s": round(next_lo - prev_hi, 1),
               "t_rel_s": round(prev_hi - t0, 1),
               "n_samples_in_window": len(window_stats(mem, w_lo, w_hi, "rss_mib"))}
        # A baseline needs enough samples to be a baseline. With 1 s candidate phases there
        # often are not any, and a median over two points is not a steady state -- those
        # cells report null rather than a number that looks measured.
        MIN_STEADY = 10          # 1 s at 10 Hz
        for f in mem[0]["_fields"]:
            pre = window_stats(mem, w_lo - STEADY_WIN, w_lo, f)
            peak_v = window_stats(mem, w_lo, w_hi, f)
            post = window_stats(mem, w_hi, w_hi + STEADY_WIN, f)
            if len(peak_v) < 5:
                continue
            peak = max(peak_v)
            s_pre = round(st.median(pre), 1) if len(pre) >= MIN_STEADY else None
            s_post = round(st.median(post), 1) if len(post) >= MIN_STEADY else None
            base = max([x for x in (s_pre, s_post) if x is not None], default=None)
            hold = sum(1 for v in peak_v if v >= peak * 0.95) / 10.0
            rec[f] = {"steady_pre": s_pre, "peak": round(peak, 1), "steady_post": s_post,
                      "increment": (round(peak - base, 1) if base is not None else None),
                      "peak_hold_s": round(hold, 1),
                      "n_pre": len(pre), "n_post": len(post)}
        out.append(rec)
    # churn: does the floor ratchet up across successive transitions?
    floors = [r["rss_mib"]["steady_post"] for r in out if r.get("rss_mib")
              and r["rss_mib"]["steady_post"] is not None]
    return {"run": os.path.basename(run), "transitions": out,
            "rss_floor_sequence": floors,
            "global_peak": {f: round(max(r[f] for r in mem), 1)
                            for f in mem[0]["_fields"]}}


def campaign_runs(platform="gpu"):
    """The v38 campaign proper: rep 0-2 of each variant.

    r99 is the pilot and r90/r91 the sampler control; including them would mix a
    configuration check and an observer-cost check into the measurement.
    """
    out = []
    pats = ["*_mem*_r[0-2]"] if platform == "gpu" else ["*_npumem*_r[0-2]"]
    for d in sorted(sum([glob.glob(os.path.join(PROJECT, "runs", p)) for p in pats], [])):
        if os.path.isfile(os.path.join(d, "mem.csv")):
            out.append(d)
    return out


def summarise(results):
    """Bucket transitions by kind and reduce. One implementation, so the report, the
    confirmed table and the console table cannot disagree with each other."""
    buckets = collections.defaultdict(list)
    peaks = collections.defaultdict(list)
    for r in results:
        for f, v in r["global_peak"].items():
            peaks[f].append(v)
        for t in r["transitions"]:
            if t["kind"] == "none":
                continue
            buckets["generative" if "generative" in t["kind"] else "vision"].append(t)
    out = {"peaks": {f: {"max": max(v), "mean": round(st.mean(v), 1),
                         "sd": round(st.pstdev(v), 1), "n": len(v)}
                     for f, v in peaks.items()},
           "kinds": {}}
    for kind, ts in buckets.items():
        k = {}
        for f in ("rss_mib", "vram_ours_mib", "npu_ours_mib"):
            usable = [t for t in ts if t.get(f) and t[f]["increment"] is not None]
            if not usable:
                continue
            inc = [t[f]["increment"] for t in usable]
            hold = [t[f]["peak_hold_s"] for t in usable]
            worst = max(usable, key=lambda t: t[f]["increment"])
            k[f] = {"n": len(usable),
                    "incr_mean": round(st.mean(inc), 1), "incr_sd": round(st.pstdev(inc), 1),
                    "incr_max": round(max(inc), 1), "incr_min": round(min(inc), 1),
                    "hold_mean": round(st.mean(hold), 2), "hold_max": round(max(hold), 1),
                    "peak_max": round(max(t[f]["peak"] for t in usable), 1),
                    "worst_base": worst[f]["steady_post"] or worst[f]["steady_pre"],
                    "worst_run": worst["run"], "worst_transition": f"{worst['from']}->{worst['to']}"}
        out["kinds"][kind] = k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs = a.runs
    if a.all or not runs:
        runs = sorted(glob.glob(os.path.join(PROJECT, "runs", "*_mem*_r*")) +
                      glob.glob(os.path.join(PROJECT, "runs", "*_npumem*_r*")))
    runs = [r for r in runs if os.path.isfile(os.path.join(r, "mem.csv"))]
    res = [analyse(r) for r in runs]

    print(f"{'run':34s} {'from->to':22s} {'kind':17s} {'gap':>5s} "
          f"{'rss pre/peak/post':>26s} {'incr':>7s} {'hold':>5s} {'vram incr':>9s}")
    print("-" * 132)
    for r in res:
        for t in r["transitions"]:
            m, v = t.get("rss_mib"), t.get("vram_ours_mib")
            if not m:
                continue
            print(f"{r['run'][:34]:34s} {t['from'] + '->' + t['to']:22s} {t['kind']:17s} "
                  f"{t['gap_s']:5.1f} "
                  f"{str(m['steady_pre']):>8s}/{m['peak']:8.1f}/{str(m['steady_post']):>8s} "
                  f"{str(m['increment']):>7s} {m['peak_hold_s']:5.1f} "
                  f"{str(v['increment'] if v else '-'):>9s}")
    print()
    for r in res:
        print(f"{r['run'][:40]:40s} global peak "
              f"rss {r['global_peak']['rss_mib']:8.1f}  "
              f"vram_ours {r['global_peak']['vram_ours_mib']:7.1f}  "
              f"vram_total {r['global_peak']['vram_total_mib']:7.1f}  "
              f"| rss floors {r['rss_floor_sequence']}")
    # ---- summary by transition kind ------------------------------------------------
    # Capacities are the measurement host's, read once and quoted with the headroom so the
    # multiple can never be reused on a different machine by accident.
    host_mib = _host_ram_mib()
    vram_mib = _vram_total_mib()
    print(f"\ncapacity: host RAM {host_mib} MiB, GPU VRAM {vram_mib} MiB")
    summ = summarise([r for r in res if "_r9" not in r["run"]])
    print(f"\n{'kind':12s} {'n':>3s} {'metric':>14s} {'incr mean':>10s} {'incr max':>9s} "
          f"{'hold mean':>10s} {'peak max':>9s} {'headroom at max':>16s}")
    print("-" * 92)
    for kind, k in sorted(summ["kinds"].items()):
        for f, cap in (("rss_mib", host_mib), ("vram_ours_mib", vram_mib),
                       ("npu_ours_mib", _npu_capacity_mib())):
            if f not in k or cap is None:
                continue
            v = k[f]
            # Headroom at the WORST transition, not on averages, and against ITS OWN steady
            # state: what was left above the level the system actually sat at, divided by
            # what that transition briefly added on top.
            head = ((cap - (v["worst_base"] or 0)) / v["incr_max"]) if v["incr_max"] > 0 \
                else float("inf")
            v["headroom"] = round(head, 1) if head != float("inf") else None
            print(f"{kind:12s} {v['n']:3d} {f:>14s} {v['incr_mean']:10.1f} {v['incr_max']:9.1f} "
                  f"{v['hold_mean']:10.2f} {v['peak_max']:9.1f} "
                  f"{(f'{head:.0f}x' if head != float('inf') else 'no overshoot'):>16s}")
    if a.json:
        json.dump({"capacity_mib": {"host_ram": host_mib, "vram_total": vram_mib},
                   "runs": res}, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
