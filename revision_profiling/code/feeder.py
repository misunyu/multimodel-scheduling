"""Per-instance request generator (spec v2.2 section 2.3).

Open loop: absolute-time grid start + k/r. A tick that is late beyond the next grid point is skipped (missed_ticks)
and the feeder realigns to the next grid point after "now". No catch-up bursts.
Closed loop (offline profile only): the feeder waits for a free slot, so the worker is never idle and nothing is dropped.
The feeder only puts an index into the pre-decoded input list.

The same loop runs either in a thread of the executor process (attempt 1: realization far below 0.95 once any CPU
instance is present, see pilot/arrival/attempt1_thread_feeder_dev) or in its own process (default; spec 2.3 fallback).
"""
import sys

sys.dont_write_bytecode = True

import time


def feeder_loop(q, rate, n_inputs, start_ns, should_stop, closed_loop=False):
    """Returns the event buffers: gen_ts, gen_ok, gen_delay, missed_grid (all ns) and gen_id (request id = tick index)."""
    gen_ts, gen_ok, gen_delay, missed, gen_id = [], [], [], [], []
    mono = time.monotonic_ns
    k = 0
    if closed_loop:
        while mono() < start_ns and not should_stop():
            time.sleep(0.001)
        while not should_stop():
            ok, ts = q.put_nowait(k)
            if not ok:
                time.sleep(0.00005)
                continue
            gen_ts.append(ts)
            gen_ok.append(1)
            gen_delay.append(0)
            gen_id.append(k)
            k += 1
        return gen_ts, gen_ok, gen_delay, missed, gen_id
    period = 1e9 / float(rate)
    while not should_stop():
        g = start_ns + int(k * period)
        now = mono()
        if now < g:
            time.sleep(min((g - now) / 1e9, 0.05))
            continue
        if now >= start_ns + int((k + 1) * period):
            # late beyond the next grid point: skip every grid point that is already due
            k_now = int((now - start_ns) / period)
            while start_ns + int(k_now * period) > now:
                k_now -= 1
            for j in range(k, k_now + 1):
                missed.append(start_ns + int(j * period))
            k = k_now + 1
            continue
        ok, ts = q.put_nowait(k)
        gen_ts.append(ts)
        gen_ok.append(ok)
        gen_delay.append(ts - g)
        gen_id.append(k)
        k += 1
    return gen_ts, gen_ok, gen_delay, missed, gen_id


def feeder_process_main(name, q, rate, n_inputs, start_val, stop_val, closed_loop, conn, code_dir, repo):
    """Entry point of a feeder process. Reports its event buffers, its module paths and clock info through conn."""
    import os
    from pathlib import Path
    import numpy as np
    conn.send({"hello": name, "pid": os.getpid(), "mono_ns": time.monotonic_ns(),
               "clock": time.get_clock_info("monotonic").implementation})
    while start_val.value == 0 and not stop_val.value:
        time.sleep(0.001)
    bufs = feeder_loop(q, rate, n_inputs, start_val.value, lambda: stop_val.value, closed_loop)
    mods, bad = {}, []
    for mname, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(repo)
        except ValueError:
            continue
        if rel.parts[0] == ".venv":
            continue
        mods[mname] = str(rel)
        if p.parent != Path(code_dir):
            bad.append(f"{mname}:{rel}")
    conn.send({"name": name, "bufs": [np.asarray(b, dtype=np.int64) for b in bufs], "modules": mods, "violations": bad,
               "cpu_sec_total": time.process_time(), "dont_write_bytecode": sys.dont_write_bytecode})
    conn.close()


def boundary_reader_main(names, queues, start_val, stop_val, warmup_ns, window_ns, conn):
    """Separate process (no GIL shared with the workers) that reads every queue's state at t0 and t1.
    It only reads shared memory; nobody waits for it."""
    while start_val.value == 0 and not stop_val.value:
        time.sleep(0.001)
    t0 = start_val.value + warmup_ns
    out = []
    for t in (t0, t0 + window_ns):
        while True:
            d = t - time.monotonic_ns()
            if d <= 0:
                break
            time.sleep(min(d / 1e9, 0.2))
        out.append({n: q.snapshot() for n, q in zip(names, queues)})
    conn.send(out)
    conn.close()
