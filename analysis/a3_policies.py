"""A3 placement policies (Phase 1).

Prior-work / practice placement-selection policies compared against a post-hoc
Oracle. Each policy is a PURE function: (profiling / runtime signals) -> number
of foreground streams to place on the NPU (0 = All-GPU, N = All-NPU).

Streams are migrated to the NPU in a fixed offline order given by each stream's
large-object fraction (descending), identical to the paper's Oracle and to the
split family (`phase_rev29_split.py` / `phase_rev30_clean.py`: MOVE_ORDER=[3,21,22,2]).
So "m streams on NPU" corresponds exactly to the measured `split{m}` placement.

All numeric inputs come from logged measurements; nothing here fabricates data.

Isolated single-stream profile (paper Table `tab:single-stream`,
source `rev19_table1_threads4.csv` / `rev7_single_stream.csv`):
"""
from __future__ import annotations
from dataclasses import dataclass

# ---- Isolated profiling constants (Table 2) ----------------------------------
ISO_SAP = {"GPU": 0.196, "NPU": 0.186}      # sAP_overall, negligible deadline miss
ISO_LAT_MS = {"GPU": 8.3, "NPU": 10.0}      # end-to-end mean latency, single stream
FRAME_BUDGET_MS = 33.3                        # 30 FPS deadline
CROSSOVER_DM_PCT = 48.0                       # measured homogeneous crossover (Fig sweeps)


@dataclass
class Decision:
    n_npu: int          # number of foreground streams moved to NPU (0..N)
    triggered: bool     # whether the policy actively moved anything off All-GPU
    rationale: str


# ---- P1: Isolated-Accuracy (vendor-style FP32 preference) --------------------
def p1_isolated_accuracy(N: int, iso_sap: dict = ISO_SAP) -> Decision:
    """Place all streams on the device with the higher isolated single-stream sAP."""
    best = max(iso_sap, key=iso_sap.get)
    n_npu = N if best == "NPU" else 0
    return Decision(n_npu, best == "NPU",
                    f"isolated sAP GPU={iso_sap['GPU']} vs NPU={iso_sap['NPU']} -> all-{best}")


# ---- P2: Latency-Profiling (TOD-style) ---------------------------------------
def p2_latency_profiling(N: int, iso_lat: dict = ISO_LAT_MS,
                         budget: float = FRAME_BUDGET_MS) -> Decision:
    """Place all streams on the lowest-latency device that meets the frame budget."""
    feasible = {d: l for d, l in iso_lat.items() if l <= budget}
    pool = feasible or iso_lat
    best = min(pool, key=pool.get)
    n_npu = N if best == "NPU" else 0
    return Decision(n_npu, best == "NPU",
                    f"isolated latency GPU={iso_lat['GPU']}ms vs NPU={iso_lat['NPU']}ms "
                    f"(budget {budget}ms) -> all-{best}")


# ---- P3: Utilization-Threshold Migration (interference-aware provisioning) ----
def p3_util_threshold(N: int, gpu_util_pct: float, theta_pct: float) -> Decision:
    """Migrate streams (large-object-first) to the NPU while measured GPU
    utilization exceeds theta.

    Because the GPU-pinned co-tenant -- not the foreground streams -- dominates
    GPU utilization, moving a foreground stream to the NPU does not appreciably
    lower the measured utilization (paper Fig `fig:sweeps`: util stays ~68% while
    deadline misses climb). Under a utilization signal that stays above theta, the
    rule therefore cascades to All-NPU once it triggers; if utilization never
    exceeds theta it stays All-GPU. This is exactly why a utilization threshold is
    a poor contention signal in the flat-utilization / rising-deadline-miss regime.
    """
    if gpu_util_pct is None:
        return Decision(0, False, "gpu_util missing")
    if gpu_util_pct > theta_pct:
        return Decision(N, True, f"GPU util {gpu_util_pct:.1f}% > theta {theta_pct:.0f}% -> migrate all")
    return Decision(0, False, f"GPU util {gpu_util_pct:.1f}% <= theta {theta_pct:.0f}% -> stay All-GPU")


# ---- P4: Deadline-Miss-Aware (this paper's recommended signal) ---------------
def p4_deadline_miss_aware(N: int, gpu_dm_pct: float,
                           crossover: float = CROSSOVER_DM_PCT) -> Decision:
    """Move to All-NPU once the measured foreground GPU deadline-miss rate exceeds
    the homogeneous crossover (~48%); otherwise stay All-GPU. Simplified policized
    form of the paper's recommendation (which additionally reports NPU-side DM)."""
    if gpu_dm_pct is None:
        return Decision(0, False, "gpu_dm missing")
    if gpu_dm_pct > crossover:
        return Decision(N, True, f"GPU DM {gpu_dm_pct:.1f}% > crossover {crossover:.0f}% -> All-NPU")
    return Decision(0, False, f"GPU DM {gpu_dm_pct:.1f}% <= crossover {crossover:.0f}% -> All-GPU")


# Convenience registry (P3 instantiated per-theta in the driver)
POLICIES = {
    "P1_isolated_accuracy": p1_isolated_accuracy,
    "P2_latency_profiling": p2_latency_profiling,
    "P3_util_threshold": p3_util_threshold,
    "P4_deadline_miss_aware": p4_deadline_miss_aware,
}
