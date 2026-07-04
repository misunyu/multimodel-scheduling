"""A2 CPU stress worker — pure-Python busy spin pinned to one core.

Launched as a SUBPROCESS (not a thread) so it occupies a real core despite the
GIL. Pins itself to the core id given as argv[1] via sched_setaffinity, then
spins on integer arithmetic (no allocation, no numpy/BLAS, NO GPU) until killed.
"""
import os
import sys

core = int(sys.argv[1])
try:
    os.sched_setaffinity(0, {core})
except Exception:
    pass

x = 0
while True:
    # tight integer loop; ~pure ALU, keeps one core pinned at ~100%
    for _ in range(1_000_000):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
