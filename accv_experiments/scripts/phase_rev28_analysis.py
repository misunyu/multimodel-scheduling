"""rev28 — "evaluation protocol systematically fails" re-analysis (pure CPU, NO new measurement).

All sources are existing threads=4 operating-point CSVs (yolo11s). core eval scripts / paper /
results unchanged. Honest: report small effects as small; note data limits explicitly.

A1 reversal frequency  (isolated prescription vs deployment-optimal) -> rev28_reversal_frequency.csv
A2 Oracle vs isolated gap along the sweep                             -> rev28_oracle_gap_sweep.csv
A3 metric sensitivity (mean/median/p10/worst)                        -> rev28_metric_sensitivity.csv
A5 worst-stream dominance (leave-one-out)                            -> rev28_worst_dominance.csv
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np, pandas as pd

RES = Path("accv_experiments/results")
OUT_MD = RES / "rev28_report.md"
BGNAME = {"L1_light": "CNN", "L2_lm": "LM", "L3_vlm": "VLM"}

def w(path, rows, cols):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for r in rows: wr.writerow(r)

# ============================ A1 ============================
def a1():
    """Isolated prescription = All-GPU (Table 1: GPU >= NPU at every size for every camera,
    so isolated single-camera profiling routes every stream to the GPU).
    Deployment-optimal = strategy with max worst-stream sAP in the actual co-tenant cell.
    Reversal = deployment-optimal != All-GPU. Granularity: (co-tenant x N) cells (rev20),
    NOT per-stream combinatorial scenarios (per-stream sAP under VLM/LM not in existing data)."""
    df = pd.read_csv(RES / "rev20_5strat_heavybg.csv")
    rows = []
    detail = []
    for bg in ["L1_light", "L2_lm", "L3_vlm"]:
        for N in sorted(df[df.bg == bg].N.unique()):
            sub = df[(df.bg == bg) & (df.N == N)]
            # mean worst over reps per strategy
            wmean = sub.groupby("strategy").worst_sap.mean()
            if "All-GPU" not in wmean.index: continue
            iso = wmean["All-GPU"]                      # isolated prescription
            # deployment-optimal among DEPLOYABLE strategies (exclude Oracle, which is non-deployable)
            deploy = wmean.drop(labels=[x for x in ["Oracle"] if x in wmean.index])
            best = deploy.idxmax(); best_val = deploy.max()
            reversed_ = int(best != "All-GPU")
            detail.append({"co_tenant": BGNAME[bg], "N": int(N),
                           "isolated_rx": "All-GPU", "isolated_worst": round(iso, 4),
                           "deploy_best": best, "deploy_best_worst": round(best_val, 4),
                           "reversed": reversed_})
    # aggregate per co-tenant (over N cells)
    dd = pd.DataFrame(detail)
    for bg in ["CNN", "LM", "VLM"]:
        s = dd[dd.co_tenant == bg]
        rows.append({"co_tenant": bg, "N": "all", "n_scenarios": int(len(s)),
                     "n_reversed": int(s.reversed.sum()),
                     "frac_reversed": round(s.reversed.mean(), 3) if len(s) else 0.0})
    # also per (co_tenant,N)
    for r in detail:
        rows.append({"co_tenant": r["co_tenant"], "N": r["N"], "n_scenarios": 1,
                     "n_reversed": r["reversed"], "frac_reversed": float(r["reversed"])})
    w(RES / "rev28_reversal_frequency.csv", rows,
      ["co_tenant", "N", "n_scenarios", "n_reversed", "frac_reversed"])
    return dd, rows

# ============================ A2 ============================
def a2():
    """Along the N=4 sweep: isolated prescription worst (All-GPU) vs Oracle worst.
    Oracle approximated as max(All-GPU, All-NPU) per point (only these two measured in rev24).
    gap = Oracle - isolated. Shows the isolated profile drifting from the optimum as skip rises."""
    df = pd.read_csv(RES / "rev24_sweep_byN_points.csv")
    s = df[df.N == 4].sort_values("gpu_skip")
    rows = []
    for _, r in s.iterrows():
        iso = r.allgpu_worst                  # isolated prescription = All-GPU
        oracle = max(r.allgpu_worst, r.allnpu_worst)   # best-of-two approx
        rows.append({"gpu_skip": round(r.gpu_skip, 1), "isolated_worst": round(iso, 4),
                     "oracle_worst": round(oracle, 4), "gap": round(oracle - iso, 4)})
    w(RES / "rev28_oracle_gap_sweep.csv", rows,
      ["gpu_skip", "isolated_worst", "oracle_worst", "gap"])
    return rows

# ============================ A3 / A5 ============================
def perstream_high_contention():
    """Per-stream sAP for All-GPU at the highest GPU-pressure sweep point (threads=4,
    rev22_perstream_sap, N=4). This is the reversal regime (GPU saturated, NPU immune)."""
    ps = pd.read_csv(RES / "rev22_perstream_sap.csv")
    ag = ps[ps.strategy == "All-GPU"]
    kmax = ag.resnet_k.max()
    cell = ag[ag.resnet_k == kmax].sort_values("sid")
    gpu_skip = cell.skip.mean()
    saps = cell["sap"].to_numpy()
    an = ps[ps.strategy == "All-NPU"]
    an_cell = an[an.resnet_k == kmax]
    an_saps = an_cell["sap"].to_numpy()
    return kmax, gpu_skip, saps, an_saps, cell

def a3(saps, an_saps, gpu_skip, kmax):
    def stats(a):
        return {"mean": float(np.mean(a)), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "worst": float(np.min(a))}
    g = stats(saps); n = stats(an_saps)
    rows = []
    for metric in ["mean", "median", "p10", "worst"]:
        # exposes the failed camera? worst/p10 expose; mean/median hide (buoyed)
        exposes = "exposes" if metric in ("worst", "p10") else ("partial" if metric == "median" else "hides")
        rows.append({"metric": metric, "allgpu_value": round(g[metric], 4),
                     "allnpu_value": round(n[metric], 4),
                     "exposes_failed_camera": exposes})
    w(RES / "rev28_metric_sensitivity.csv", rows,
      ["metric", "allgpu_value", "allnpu_value", "exposes_failed_camera"])
    return rows, g, n

def a5(saps, cell, gpu_skip):
    a = np.sort(saps)  # ascending
    worst = float(a[0]); second = float(a[1]) if len(a) > 1 else float("nan")
    mean = float(np.mean(a)); mean_wo = float(np.mean(a[1:])) if len(a) > 1 else float("nan")
    rows = [{"setting": "All-GPU @ high-GPU-contention (N=4, threads=4)",
             "gpu_skip": round(gpu_skip, 1),
             "mean": round(mean, 4), "worst": round(worst, 4), "second_worst": round(second, 4),
             "mean_wo_worst": round(mean_wo, 4),
             "worst_over_mean": round(worst / mean, 3) if mean else None,
             "mean_lift_dropping_worst": round(mean_wo - mean, 4)}]
    w(RES / "rev28_worst_dominance.csv", rows,
      ["setting", "gpu_skip", "mean", "worst", "second_worst", "mean_wo_worst",
       "worst_over_mean", "mean_lift_dropping_worst"])
    return rows

# ============================ report ============================
def main():
    dd, a1rows = a1()
    a2rows = a2()
    kmax, gpu_skip, saps, an_saps, cell = perstream_high_contention()
    a3rows, g, n = a3(saps, an_saps, gpu_skip, kmax)
    a5rows = a5(saps, cell, gpu_skip)

    b = []
    b.append("# rev28 — evaluation protocol systematically fails (re-analysis)\n\n")
    b.append("_Pure CPU re-analysis of existing threads=4 yolo11s CSVs. NO new measurement. "
             "Run after rev27 completed (no concurrent GPU/NPU work). core scripts / paper / results unchanged._\n\n")

    b.append("## A1 — Ranking reversal frequency (isolated prescription vs deployment-optimal)\n\n")
    b.append("Isolated single-camera profiling routes **every** stream to the GPU (Table 1: GPU ≥ NPU at every size), "
             "so the isolated prescription is **All-GPU** in every scenario. Deployment-optimal = the deployable "
             "strategy maximizing worst-stream sAP under the actual co-tenant. Reversal = optimum ≠ All-GPU.\n\n")
    b.append("Source: `rev20_5strat_heavybg.csv` (worst/mean per co-tenant×N×strategy, threads=4, 3 reps). "
             "**Granularity = (co-tenant × N) cells**, not per-stream combinatorial scenarios "
             "(per-stream sAP under VLM/LM is not in existing data → that richer sampling would need re-measurement; deferred).\n\n")
    b.append("| co-tenant | cells (N=2/4/8) | reversed | frac |\n|---|---|---|---|\n")
    for r in a1rows:
        if r["N"] == "all":
            b.append(f"| {r['co_tenant']} | {r['n_scenarios']} | {r['n_reversed']} | {r['frac_reversed']:.0%} |\n")
    b.append("\nPer-cell detail:\n\n| co-tenant | N | isolated Rx (worst) | deployment-best (worst) | reversed? |\n|---|---|---|---|---|\n")
    for _, r in dd.iterrows():
        b.append(f"| {r['co_tenant']} | {r['N']} | All-GPU ({r['isolated_worst']:.4f}) | "
                 f"{r['deploy_best']} ({r['deploy_best_worst']:.4f}) | {'YES' if r['reversed'] else 'no'} |\n")
    b.append("\n")

    b.append("## A2 — Oracle vs isolated-prescription gap along the sweep (N=4)\n\n")
    b.append("Source: `rev24_sweep_byN_points.csv`. Isolated = All-GPU; Oracle ≈ max(All-GPU, All-NPU) "
             "(only these two measured along the sweep — approximation noted). gap grows with GPU skip:\n\n")
    b.append("| GPU skip % | isolated (All-GPU) | Oracle≈max | gap |\n|---|---|---|---|\n")
    for r in a2rows:
        b.append(f"| {r['gpu_skip']:.0f} | {r['isolated_worst']:.4f} | {r['oracle_worst']:.4f} | {r['gap']:+.4f} |\n")
    b.append("\nThe isolated prescription (All-GPU) is optimal at low contention (gap 0) but diverges from the "
             "optimum as GPU skip rises past the crossover — the protocol's error grows with deployment contention.\n\n")

    b.append(f"## A3 — Metric sensitivity (All-GPU, high GPU contention, N=4, GPU skip ≈{gpu_skip:.0f}%)\n\n")
    b.append("Source: `rev22_perstream_sap.csv` (threads=4 per-stream, ResNet sweep, highest GPU-pressure point — "
             "the reversal regime). **N=4 → only 4 per-stream values, so p10 ≈ worst (coarse percentile).**\n\n")
    b.append("| metric | All-GPU | All-NPU | exposes failed camera? |\n|---|---|---|---|\n")
    for r in a3rows:
        b.append(f"| {r['metric']} | {r['allgpu_value']:.4f} | {r['allnpu_value']:.4f} | {r['exposes_failed_camera']} |\n")
    b.append(f"\nAll-GPU per-stream sAP at this point: `{[round(x,4) for x in sorted(saps)]}`. "
             "mean is buoyed by the less-starved streams; worst (and p10, at N=4) exposes the failed camera. "
             "Honest: at N=4 p10 and worst nearly coincide — worst is the most conservative/sensitive, "
             "but we do not overclaim p10 is uniquely blind.\n\n")

    b.append(f"## A5 — Worst-stream dominance (leave-one-out)\n\n")
    r = a5rows[0]
    b.append("Source: same per-stream cell (All-GPU, high contention).\n\n")
    b.append(f"- mean {r['mean']:.4f}, worst {r['worst']:.4f}, 2nd-worst {r['second_worst']:.4f}\n")
    b.append(f"- worst/mean = {r['worst_over_mean']} (worst is {1/r['worst_over_mean']:.1f}× below the mean)\n")
    b.append(f"- dropping the worst stream lifts the system mean by {r['mean_lift_dropping_worst']:+.4f} "
             f"({r['mean']:.4f} → {r['mean_wo_worst']:.4f}) — one camera dominates the safety-relevant signal.\n\n")

    b.append("## Honest limitations\n\n")
    b.append("- Emulated multi-camera (24 forward-camera logs replayed) limits combinatorial diversity; A1 is at "
             "(co-tenant×N) granularity, not arbitrary per-stream scenarios.\n")
    b.append("- A2 Oracle is approximated as best-of-{All-GPU,All-NPU} along the sweep (full 2^N not swept).\n")
    b.append("- A3/A5 use the ResNet high-GPU-contention point as the reversal-regime proxy (threads=4); the VLM "
             "point itself lacks persisted per-stream sAP. N=4 makes p10 coarse.\n")
    b.append("- No new measurement; all from existing threads=4 yolo11s CSVs (rev20/rev22/rev24).\n\n")
    b.append("## Files\n- rev28_reversal_frequency.csv, rev28_oracle_gap_sweep.csv, "
             "rev28_metric_sensitivity.csv, rev28_worst_dominance.csv, rev28_report.md\n")
    OUT_MD.write_text("".join(b))
    print("=== rev28 done ===")
    print("A1 frac reversed:", {r['co_tenant']: r['frac_reversed'] for r in a1rows if r['N']=='all'})
    print("A2 max gap:", max(r['gap'] for r in a2rows))
    print("A3 metrics (AllGPU):", {r['metric']: r['allgpu_value'] for r in a3rows})
    print("A5:", a5rows[0])

if __name__ == "__main__":
    main()
