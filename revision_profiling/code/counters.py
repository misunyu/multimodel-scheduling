"""Event logs and window aggregation (spec v2.2 section 2.4).

Every thread appends timestamps (time.monotonic_ns) to its own lists and never waits for anyone.
Aggregation happens after the run: an event belongs to the window iff t0 <= ts < t1.
"""
import sys

sys.dont_write_bytecode = True

import numpy as np


class InstanceLog:
    """Per-instance event buffers. Each list is appended to by exactly one thread."""

    def __init__(self, name):
        self.name = name
        # feeder
        self.gen_ts = []        # actual tick time (ns)
        self.gen_ok = []        # 1 = enqueued, 0 = dropped_full
        self.gen_delay = []     # actual - grid (ns)
        self.gen_id = []        # request id (tick index)
        self.missed_grid = []   # grid time (ns) of skipped ticks
        # worker
        self.deq_ts = []
        self.deq_id = []
        self.done_id = []
        self.fail_id = []
        self.done_ts = []       # completed: output queue put succeeded
        self.done_enq = []      # enqueue ts of the completed request
        self.done_deq = []
        self.done_pre = []      # preprocess duration (ns)
        self.fail_ts = []
        self.fail_kind = []     # "exception:<repr>" | "output_put_failed"
        # consumer
        self.consumer_fail_ts = []


def _count(ts, t0, t1):
    a = np.asarray(ts, dtype=np.int64)
    return int(((a >= t0) & (a < t1)).sum())


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def scheduled_ticks(start_ns, period_ns, t0, t1):
    """Number of grid points start + int(k*period) inside [t0, t1)."""
    k_lo = max(0, int((t0 - start_ns) / period_ns) - 2)
    k_hi = int((t1 - start_ns) / period_ns) + 2
    k = np.arange(k_lo, k_hi + 1, dtype=np.float64)
    grid = start_ns + (k * period_ns).astype(np.int64)
    return int(((grid >= t0) & (grid < t1)).sum())


def window_stats(log, t0, t1, feeder_start_ns, period_ns, snap_start, snap_end, leftover_ids, closed_loop=False):
    """Phase 1b section 3.
    Primary: everything is derived from event timestamps in [t0, t1); pending at a boundary is reconstructed per request
    (enqueue ts < boundary and not yet completed/failed at the boundary). Conservation must be exactly 0.
    Cross-check: snap_start / snap_end = reader-process snapshots {"ts", "qsize", "inflight"}; the difference to the
    timestamp-based pending is recorded as snapshot_diff_* (measurement error, never a tolerance).
    trace_check: every request (by id) that was pending at t0 or generated in the window takes exactly one of
    dropped_full / completed / failed / pending_end."""
    gen_ts = np.asarray(log.gen_ts, dtype=np.int64)
    gen_ok = np.asarray(log.gen_ok, dtype=np.int64)
    gen_id = np.asarray(log.gen_id, dtype=np.int64)
    gen_delay = np.asarray(log.gen_delay, dtype=np.int64)
    done_ts = np.asarray(log.done_ts, dtype=np.int64)
    done_id = np.asarray(log.done_id, dtype=np.int64)
    fail_ts = np.asarray(log.fail_ts, dtype=np.int64)
    fail_id = np.asarray(log.fail_id, dtype=np.int64)
    deq_id = np.asarray(log.deq_id, dtype=np.int64)

    gw = (gen_ts >= t0) & (gen_ts < t1)
    generated = int(gw.sum())
    enqueued = int(gen_ok[gw].sum())
    dropped_full = generated - enqueued
    dw = (done_ts >= t0) & (done_ts < t1)
    fw = (fail_ts >= t0) & (fail_ts < t1)
    completed = int(dw.sum())
    failed = int(fw.sum())
    fail_out = int(sum(1 for ts, k in zip(log.fail_ts, log.fail_kind) if t0 <= ts < t1 and k == "output_put_failed"))
    consumer_failed = _count(log.consumer_fail_ts, t0, t1)

    # per-request state, joined by request id
    problems = []
    for nm, arr in (("gen_id", gen_id), ("deq_id", deq_id), ("done_id", done_id), ("fail_id", fail_id)):
        if len(np.unique(arr)) != len(arr):
            problems.append(f"duplicate_{nm}")
    if len(np.intersect1d(done_id, fail_id)):
        problems.append("completed_and_failed")
    enq_mask = gen_ok == 1
    enq_id, enq_ts = gen_id[enq_mask], gen_ts[enq_mask]
    fin_of = {int(i): int(t) for i, t in zip(done_id, done_ts)}
    fin_of.update({int(i): int(t) for i, t in zip(fail_id, fail_ts)})
    enq_set = set(int(i) for i in enq_id)
    if not set(fin_of) <= enq_set:
        problems.append("finished_but_never_enqueued")
    if not set(int(i) for i in deq_id) <= enq_set:
        problems.append("dequeued_but_never_enqueued")
    never = enq_set - set(fin_of)                       # enqueued, never finished: must be in the queue or in flight at stop
    unfinished_deq = set(int(i) for i in deq_id) - set(fin_of)
    if never != set(leftover_ids) | unfinished_deq or len(unfinished_deq) > 1:
        problems.append("unfinished_requests_do_not_match_queue_at_stop")
    big = np.iinfo(np.int64).max
    fin_ts = np.array([fin_of.get(int(i), big) for i in enq_id], dtype=np.int64)
    pend0_mask = (enq_ts < t0) & (fin_ts >= t0)
    pend1_mask = (enq_ts < t1) & (fin_ts >= t1)
    p0, p1 = int(pend0_mask.sum()), int(pend1_mask.sum())

    # trace: population = pending at t0 + generated in the window; each id must have exactly one outcome
    pop = set(int(i) for i in enq_id[pend0_mask]) | set(int(i) for i in gen_id[gw])
    out_drop = set(int(i) for i in gen_id[gw & (gen_ok == 0)])
    out_done = set(int(i) for i in done_id[dw])
    out_fail = set(int(i) for i in fail_id[fw])
    out_pend = set(int(i) for i in enq_id[pend1_mask])
    n_paths = {i: 0 for i in pop}
    orphans = 0
    for sset in (out_drop, out_done, out_fail, out_pend):
        for i in sset:
            if i in n_paths:
                n_paths[i] += 1
            else:
                orphans += 1
    missing = sum(1 for v in n_paths.values() if v == 0)
    multi = sum(1 for v in n_paths.values() if v > 1)
    if missing or multi or orphans:
        problems.append(f"trace:missing={missing},multiple={multi},orphan_events={orphans}")
    trace_check = "ok" if not problems else ";".join(problems)

    lat_ms = (done_ts[dw] - np.asarray(log.done_enq, dtype=np.int64)[dw]) / 1e6
    svc_ms = (done_ts[dw] - np.asarray(log.done_deq, dtype=np.int64)[dw]) / 1e6
    pre_ms = np.asarray(log.done_pre, dtype=np.int64)[dw] / 1e6
    if closed_loop:
        sched, missed = None, 0
    else:
        sched = scheduled_ticks(feeder_start_ns, period_ns, t0, t1)
        missed = _count(log.missed_grid, t0, t1)
    s0 = snap_start["qsize"] + snap_start["inflight"]
    s1 = snap_end["qsize"] + snap_end["inflight"]
    d0, d1 = s0 - p0, s1 - p1
    return {
        "generated": generated, "enqueued": enqueued, "dropped_full": dropped_full, "completed": completed, "failed": failed,
        "failed_output_put": fail_out, "consumer_failed": consumer_failed,
        "scheduled_ticks": sched, "missed_ticks": missed, "realization": (generated / sched) if sched else None,
        "tick_delay_ms_max": float(gen_delay[gw].max() / 1e6) if generated else None,
        "tick_delay_ms_p95": _pct(gen_delay[gw] / 1e6, 95),
        "pending_start": p0, "pending_end": p1,                       # timestamp-based (primary)
        "conservation_gen": generated - enqueued - dropped_full,
        "conservation_flow": enqueued - completed - failed - (p1 - p0),
        "trace_check": trace_check, "trace_population": len(pop),
        "snapshot_start": snap_start, "snapshot_end": snap_end,
        "snapshot_pending_start": s0, "snapshot_pending_end": s1,
        "snapshot_diff_start": d0, "snapshot_diff_end": d1,
        "snapshot_read_lag_ms_start": (snap_start["ts"] - t0) / 1e6, "snapshot_read_lag_ms_end": (snap_end["ts"] - t1) / 1e6,
        "snapshot_flag": bool(abs(d0) > 1 or abs(d1) > 1),
        "inflight_start": snap_start["inflight"], "inflight_end": snap_end["inflight"],
        "latency_ms_p50": _pct(lat_ms, 50), "latency_ms_p95": _pct(lat_ms, 95),
        "service_ms_p50": _pct(svc_ms, 50), "service_ms_p95": _pct(svc_ms, 95),
        "service_ms_mean": float(svc_ms.mean()) if completed else None,
        "preprocess_ms_mean": float(pre_ms.mean()) if completed else None,
        "q_i": (dropped_full / generated) if generated else None,
    }


def invalid_reasons(instances, module_violations):
    """zero_generated, in-flight > 1, module path, conservation != 0 (timestamp based, exact), trace_check, consumer_failed."""
    r = []
    if module_violations:
        r.append("module_path")
    for inst in instances:
        w = inst["window"]
        n = inst["name"]
        if w["generated"] == 0:
            r.append(f"zero_generated:{n}")
        if w["inflight_start"] not in (0, 1) or w["inflight_end"] not in (0, 1) or not inst["inflight_sequential_ok"]:
            r.append(f"inflight_gt_1:{n}")
        if w["conservation_gen"] != 0 or w["conservation_flow"] != 0:
            r.append(f"conservation:{n}")
        if w["trace_check"] != "ok":
            r.append(f"trace_check:{n}")
        if w["consumer_failed"] > 0 or inst["consumer_failed_total"] > 0:
            r.append(f"consumer_failed:{n}")
        if not inst["total_identity_ok"]:
            r.append(f"total_identity:{n}")
    return r


def system_stats(instances, window_sec):
    comp = sum(i["window"]["completed"] for i in instances)
    drop = sum(i["window"]["dropped_full"] for i in instances)
    gen = sum(i["window"]["generated"] for i in instances)
    q = [i["window"]["q_i"] for i in instances]
    ok = all(v is not None for v in q)
    legacy = [1000.0 / i["window"]["service_ms_mean"] for i in instances if i["window"]["service_ms_mean"]]
    return {
        "completed": comp,
        "dropped_full": drop,
        "generated": gen,
        "failed": sum(i["window"]["failed"] for i in instances),
        "window_sec_actual": window_sec,
        "throughput_fps": comp / window_sec,
        "drop_rate_fps": drop / window_sec,
        "generated_rate_fps": gen / window_sec,
        "drop_macro": float(np.mean(q)) if ok else None,
        "drop_max": float(np.max(q)) if ok else None,
        "service_rate_sum_legacy": float(sum(legacy)) if len(legacy) == len(instances) else None,
    }
