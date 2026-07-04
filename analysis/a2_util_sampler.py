"""A2 combined GPU (pynvml) + host-CPU (psutil) utilization sampler.

Superset of accv_experiments/scripts/gpu_util_sampler.py: same GPU columns plus
per-sample host CPU util and memory, so Task 1's CPU-contention coordinate
(host CPU %) and the GPU coordinate share one timeline. Passive read-only; runs
NO inference. Writes until SIGTERM/SIGINT.

Usage: python a2_util_sampler.py <out.csv> <interval_s>
"""
import sys
import signal
import time

import psutil
import pynvml

OUT = sys.argv[1]
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2

_run = {"go": True}
def _stop(*a): _run["go"] = False
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
psutil.cpu_percent(percpu=False)  # prime
with open(OUT, "w", buffering=1) as f:
    f.write("epoch,util_gpu,util_mem,mem_used_mib,power_w,cpu_pct,cpu_nbusy,ram_used_mib\n")
    while _run["go"]:
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        try:
            pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            pw = -1.0
        percpu = psutil.cpu_percent(percpu=True)
        cpu_pct = sum(percpu) / len(percpu)
        nbusy = sum(1 for c in percpu if c > 50.0)   # cores >50% busy
        ram = psutil.virtual_memory().used // (1024 * 1024)
        f.write(f"{time.time():.3f},{u.gpu},{u.memory},{mem.used//(1024*1024)},"
                f"{pw:.1f},{cpu_pct:.1f},{nbusy},{ram}\n")
        time.sleep(INTERVAL)
pynvml.nvmlShutdown()
