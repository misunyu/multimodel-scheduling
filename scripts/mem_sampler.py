#!/usr/bin/env python3
"""Sample host RSS and GPU VRAM of a process tree at 10 Hz, from outside that tree.

ISSRE R4 asks whether a hot-swap begun during a contention-driven failure can exhaust
resources by running the outgoing and incoming instances at once. Answering it needs the
transition transient, not the steady state, and the transient is short: the generative
relocation takes about 2.9 s and a vision swap under a second. A 1 Hz sampler -- which is
what run_scenario.py's --proc-cpu uses -- can miss such a peak entirely, so this runs at
10 Hz.

It samples from the OBSERVER side (the parent that spawned the executor) rather than from
inside the executor, so the measurement cannot become part of what is measured. The cost of
that choice is recorded rather than assumed: scripts/run_hotswap_mem_campaign.py runs the
same scenario with and without the sampler.

Two caveats that belong next to the numbers, not in a footnote:

  * `rss_mib` sums RSS over the process tree. RSS double-counts pages shared between parent
    and children (the CUDA runtime, the ORT libraries), so the tree sum OVERSTATES real
    consumption. For a transition INCREMENT -- peak minus adjacent steady state -- the
    shared part cancels, which is why the increment is the number worth quoting and the
    absolute is not. PSS would be exact but /proc/*/smaps_rollup at 10 Hz across a dozen
    processes is too slow to sample the transient it is meant to catch.
  * `vram_ours_mib` is NVML's per-process attribution summed over our tree; `vram_total_mib`
    is the device total including other users. Both are reported because "how much did we
    add" and "how close to the limit did the machine get" are different questions.

NPU DRAM (v39) comes from `mblt-status -q -s MEMORY,PROCESS_INFO`, the vendor's nvidia-smi
equivalent, which reports usage against a 16384 MiB total and a per-PID table. It is a
subprocess call rather than a library query, so it is only made when --npu is passed: on the
CPU--GPU scenarios it would add latency for a device that is not in use. The per-PID table is
parsed the same way as NVML's, so `npu_ours_mib` and `npu_total_used_mib` mean what
`vram_ours_mib` and `vram_total_mib` mean.

usage:
  python3 scripts/mem_sampler.py --pid <pid> --out mem.csv [--hz 10]
  (or import sample_into() and call it around a subprocess)
"""
import argparse, os, sys, time

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

_NVML = None
NPU_TOTAL_RE = None


def _nvml():
    """Initialise NVML once. Absence is not fatal -- the host-RSS half still answers."""
    global _NVML
    if _NVML is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML = (pynvml, [pynvml.nvmlDeviceGetHandleByIndex(i)
                              for i in range(pynvml.nvmlDeviceGetCount())])
        except Exception:
            _NVML = (None, [])
    return _NVML


def npu_sample(pids):
    """(ours_mib, total_used_mib, capacity_mib) or (None, None, None) if unreadable.

    Returning None rather than 0 matters: "the NPU used no memory" and "we could not ask"
    are different findings, and the second is the one worth reporting.
    """
    import re as _re
    import subprocess
    try:
        out = subprocess.run(["mblt-status", "-q", "-s", "MEMORY,PROCESS_INFO"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None, None, None
    used = total = None
    m = _re.search(r"Usage\s*:\s*([0-9.]+)\s*MB", out)
    if m:
        used = float(m.group(1))
    m = _re.search(r"Total\s*:\s*([0-9.]+)\s*MB", out)
    if m:
        total = float(m.group(1))
    # The process table is multi-line, not row-per-process:
    #     Process ID            : 89256
    #         Process Name      : ... -m runtime.bg_entry --model llama1b ...
    #         Memory Usage
    #             NPU           : 1866 MB
    # so the pid has to be carried forward to the NPU line that follows it. A row-oriented
    # regex silently attributed nothing to us and reported 0 while the device held 1918 MB.
    ours, cur = 0.0, None
    for line in out.splitlines():
        m = _re.match(r"\s*Process ID\s*:\s*(\d+)", line)
        if m:
            cur = int(m.group(1))
            continue
        m = _re.match(r"\s*NPU\s*:\s*([0-9.]+)\s*MB", line)
        if m and cur is not None:
            if cur in pids:
                ours += float(m.group(1))
            cur = None
    return ours, used, total


def tree_pids(root_pid):
    if psutil is None:
        return []
    try:
        p = psutil.Process(root_pid)
    except psutil.Error:
        return []
    out = [p]
    try:
        out += p.children(recursive=True)
    except psutil.Error:
        pass
    return out


def sample_once(root_pid, npu=False):
    procs = tree_pids(root_pid)
    pids, rss = set(), 0
    for p in procs:
        try:
            rss += p.memory_info().rss
            pids.add(p.pid)
        except Exception:
            continue
    pynvml, handles = _nvml()
    vram_ours = vram_total = 0
    if pynvml is not None:
        for h in handles:
            try:
                vram_total += pynvml.nvmlDeviceGetMemoryInfo(h).used
            except Exception:
                pass
            for fn in ("nvmlDeviceGetComputeRunningProcesses_v3",
                       "nvmlDeviceGetComputeRunningProcesses"):
                get = getattr(pynvml, fn, None)
                if get is None:
                    continue
                try:
                    for pr in get(h):
                        if pr.pid in pids and pr.usedGpuMemory:
                            vram_ours += pr.usedGpuMemory
                    break
                except Exception:
                    continue
    npu_ours = npu_used = npu_cap = None
    if npu:
        npu_ours, npu_used, npu_cap = npu_sample(pids)
    return (len(pids), rss / 1048576.0, vram_ours / 1048576.0, vram_total / 1048576.0,
            npu_ours, npu_used, npu_cap)


def sample_into(root_pid, out_path, hz=10.0, stop=lambda: False, npu=False):
    """Sample until stop() is true. Header first so a killed run still parses."""
    period = 1.0 / hz
    t0 = time.time()
    fmt = lambda v: ("" if v is None else f"{v:.1f}")
    with open(out_path, "w") as f:
        f.write("t_s,epoch,n_procs,rss_mib,vram_ours_mib,vram_total_mib,"
                "npu_ours_mib,npu_total_used_mib,npu_capacity_mib\n")
        while not stop():
            n, rss, vo, vt, no, nu, nc = sample_once(root_pid, npu=npu)
            now = time.time()
            f.write(f"{now - t0:.3f},{now:.3f},{n},{rss:.1f},{vo:.1f},{vt:.1f},"
                    f"{fmt(no)},{fmt(nu)},{fmt(nc)}\n")
            f.flush()
            time.sleep(max(0.0, period - (time.time() - now)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--npu", action="store_true", help="also poll mblt-status for NPU DRAM")
    a = ap.parse_args()
    end = (time.time() + a.seconds) if a.seconds else None
    alive = lambda: (psutil is not None and psutil.pid_exists(a.pid))
    sample_into(a.pid, a.out, a.hz, npu=a.npu,
                stop=lambda: (not alive()) or (end is not None and time.time() > end))
    return 0


if __name__ == "__main__":
    sys.exit(main())
