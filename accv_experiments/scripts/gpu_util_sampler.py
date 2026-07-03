"""Standalone GPU utilization sampler (pynvml). Passive read-only NVML poll —
performs NO inference. Writes timestamped samples until SIGTERM/SIGINT.
Usage: taskset -c <core> python gpu_util_sampler.py <out.csv> <interval_s>
"""
import sys, time, signal
import pynvml

OUT = sys.argv[1]
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2

_run = {"go": True}
def _stop(*a): _run["go"] = False
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
with open(OUT, "w", buffering=1) as f:
    f.write("epoch,util_gpu,util_mem,mem_used_mib,power_w\n")
    while _run["go"]:
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        try: pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception: pw = -1.0
        f.write(f"{time.time():.3f},{u.gpu},{u.memory},{mem.used//(1024*1024)},{pw:.1f}\n")
        time.sleep(INTERVAL)
pynvml.nvmlShutdown()
