"""M2 analysis — applies only the rules pre-registered in log/m2_run.py.

Units: raw sAP [0,1]. The submitted table reports sAP x100 ("6.2" = 0.062);
conversion happens only where explicitly labelled.

Excluded: block_kind == "warmup" (pre-declared) and manifest status FAILED.
Nothing is excluded on the basis of measured values.

n=3: no p-value, no equivalence test, no inferential CI. Paired GPU-NPU
differences are formed per block first; the mean and all three individual values
are reported.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
CONDS = ["baseline", "L2LM", "L3VLM"]

# submitted values (raw sAP for worst-sAP; percent for DM and CPU util)
SUBMITTED = {
    "L2LM": {"gpu_dm": 83.0, "npu_dm": 76.0, "worst_gpu": 0.062, "worst_npu": 0.042,
             "cpu_util_allnpu": 93.0},
    "L3VLM": {"gpu_dm": 100.0, "npu_dm": 1.0, "worst_gpu": 0.016, "worst_npu": 0.083},
    "baseline": {"cpu_util_allnpu": 17.0},
}

frames = {}
for c in CONDS:
    f = OUT / f"m2_{c}_raw.csv"
    if not f.exists():
        print(f"[missing] {f.name}")
        continue
    df = pd.read_csv(f)
    frames[c] = df[df.block_kind == "measure"].copy()

# ---------------------------------------------------------------- per-cell table
cells = pd.concat(frames.values(), ignore_index=True)
cells_out = cells[["condition", "bg_level", "block", "order", "cell_index", "placement",
                   "gpu_dm_pct", "npu_dm_pct", "cpu_sys_mean", "cpu_proc_mean",
                   "gpu_util_mean", "worst_sap", "mean_sap", "wall_s", "per_stream_json"]]
cells_out.to_csv(OUT / "m2_cells.csv", index=False)

# ---------------------------------------------------------------- condition summary
rows = []
for c, df in frames.items():
    g = df[df.placement == "GGGG"].sort_values("block")
    n = df[df.placement == "NNNN"].sort_values("block")
    d = (g.set_index("block").worst_sap - n.set_index("block").worst_sap).sort_index()
    rows.append({
        "condition": c, "n_blocks": len(d),
        "gpu_dm_mean": round(g.gpu_dm_pct.mean(), 2), "gpu_dm_vals": list(g.gpu_dm_pct),
        "npu_dm_mean": round(n.npu_dm_pct.mean(), 2), "npu_dm_vals": list(n.npu_dm_pct),
        "allgpu_worst_mean": round(g.worst_sap.mean(), 5), "allgpu_worst_vals": list(g.worst_sap),
        "allnpu_worst_mean": round(n.worst_sap.mean(), 5), "allnpu_worst_vals": list(n.worst_sap),
        "allgpu_mean_sap_mean": round(g.mean_sap.mean(), 5),
        "allnpu_mean_sap_mean": round(n.mean_sap.mean(), 5),
        "cpu_allgpu_mean": round(g.cpu_sys_mean.mean(), 2), "cpu_allgpu_vals": list(g.cpu_sys_mean),
        "cpu_allnpu_mean": round(n.cpu_sys_mean.mean(), 2), "cpu_allnpu_vals": list(n.cpu_sys_mean),
        "cpu_proc_allnpu_mean": round(n.cpu_proc_mean.mean(), 1),
        "gpu_util_allgpu_mean": round(g.gpu_util_mean.mean(), 1),
        "gpu_util_allnpu_mean": round(n.gpu_util_mean.mean(), 1),
        "d_paired_mean": round(float(d.mean()), 5),
        "d_paired_vals": [round(v, 5) for v in d],
        "better_placement": "All-GPU" if d.mean() > 0 else "All-NPU",
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "m2_summary.csv", index=False)

pd.set_option("display.width", 250)
print("=== M2 condition summary (raw sAP; DM and CPU in %) ===")
for r in rows:
    print(f"\n-- {r['condition']} (n={r['n_blocks']} blocks) --")
    print(f"  GPU DM      mean {r['gpu_dm_mean']:6.2f}   vals {r['gpu_dm_vals']}")
    print(f"  NPU DM      mean {r['npu_dm_mean']:6.2f}   vals {r['npu_dm_vals']}")
    print(f"  All-GPU worst  mean {r['allgpu_worst_mean']:.5f}   vals {r['allgpu_worst_vals']}")
    print(f"  All-NPU worst  mean {r['allnpu_worst_mean']:.5f}   vals {r['allnpu_worst_vals']}")
    print(f"  CPU util All-GPU {r['cpu_allgpu_mean']:5.2f}  All-NPU {r['cpu_allnpu_mean']:5.2f} "
          f"(proc All-NPU {r['cpu_proc_allnpu_mean']}%)")
    print(f"  GPU util  All-GPU {r['gpu_util_allgpu_mean']:5.1f}  All-NPU {r['gpu_util_allnpu_mean']:5.1f}")
    print(f"  paired d = worst(All-GPU) - worst(All-NPU): mean {r['d_paired_mean']:+.5f}  "
          f"vals {r['d_paired_vals']}  -> better: {r['better_placement']}")
print("\n  (n=3: no p-value, no equivalence test, no inferential CI — per protocol.)")

# ---------------------------------------------------------------- decision (a)
S = {r["condition"]: r for r in rows}
l2 = S.get("L2LM")
base = S.get("baseline")
verdict = None
if l2:
    regime = (l2["gpu_dm_mean"] >= 50.0) and (l2["npu_dm_mean"] >= 50.0)
    verdict = {
        "rule": "(a) L2LM high-staleness regime check",
        "criterion": "3-block mean GPU DM >= 50% AND mean NPU DM >= 50%",
        "gpu_dm_mean": l2["gpu_dm_mean"], "npu_dm_mean": l2["npu_dm_mean"],
        "result": ("submitted high-NPU-DM / both-paths-stale regime strongly reproduced"
                   if regime else
                   "qualitative reproduction of that regime NOT established "
                   "(mechanism is NOT rejected by this)"),
        "regime_reproduced": bool(regime),
        "note": "50% is a pre-set criterion for the submitted regime, not a physical threshold",
    }
    if not regime and base:
        verdict["fallback_four_values"] = {
            "npu_dm_change_vs_baseline": round(l2["npu_dm_mean"] - base["npu_dm_mean"], 2),
            "allnpu_worst_change_vs_baseline": round(l2["allnpu_worst_mean"] - base["allnpu_worst_mean"], 5),
            "gpu_npu_ranking": l2["better_placement"],
            "cpu_util_allnpu_vs_placement_matched_baseline":
                [l2["cpu_allnpu_mean"], base["cpu_allnpu_mean"]],
        }
    print("\n=== DECISION (a) — L2LM high-staleness regime check ===")
    print(f"  criterion : {verdict['criterion']}")
    print(f"  measured  : GPU DM {verdict['gpu_dm_mean']}%   NPU DM {verdict['npu_dm_mean']}%")
    print(f"  result    : {verdict['result']}")
    if "fallback_four_values" in verdict:
        print("  fallback values (rule (a) requires these when NPU DM < 50%):")
        for k, v in verdict["fallback_four_values"].items():
            print(f"    {k}: {v}")

# corroborating diagnostics, not pass/fail
if l2 and base:
    print("\n  corroborating (not pass/fail): "
          f"GPU-NPU DM gap {l2['gpu_dm_mean'] - l2['npu_dm_mean']:+.2f} pp; "
          f"CPU util All-NPU {base['cpu_allnpu_mean']} -> {l2['cpu_allnpu_mean']} "
          f"(placement-matched baseline)")

# ---------------------------------------------------------------- (b) descriptive
print("\n=== (b) quantitative consistency vs submitted — DESCRIPTIVE ONLY, no pass/fail ===")
print("    (not an equivalence test, not a reproduction test; n=3 cannot support that)")
comp = []


def add(cond, metric, submitted, new, unit):
    absd = new - submitted
    reld = (absd / submitted * 100.0) if submitted else float("nan")
    comp.append({"condition": cond, "metric": metric, "unit": unit,
                 "submitted": submitted, "new_3block_mean": round(new, 5),
                 "abs_diff": round(absd, 5), "rel_diff_pct": round(reld, 1)})


for c in ("L2LM", "L3VLM"):
    if c not in S:
        continue
    s, r = SUBMITTED[c], S[c]
    add(c, "GPU deadline miss", s["gpu_dm"], r["gpu_dm_mean"], "%")
    add(c, "NPU deadline miss", s["npu_dm"], r["npu_dm_mean"], "%")
    add(c, "All-GPU worst sAP", s["worst_gpu"], r["allgpu_worst_mean"], "raw sAP")
    add(c, "All-NPU worst sAP", s["worst_npu"], r["allnpu_worst_mean"], "raw sAP")
    if "cpu_util_allnpu" in s:
        add(c, "CPU util (All-NPU)", s["cpu_util_allnpu"], r["cpu_allnpu_mean"], "%")
if "baseline" in S:
    add("baseline", "CPU util (All-NPU)", SUBMITTED["baseline"]["cpu_util_allnpu"],
        S["baseline"]["cpu_allnpu_mean"], "%")
cdf = pd.DataFrame(comp)
cdf.to_csv(OUT / "m2_vs_submitted.csv", index=False)
print(cdf.to_string(index=False))

out = {"decision_a": verdict, "conditions": rows}
(OUT / "m2_verdict.json").write_text(json.dumps(out, indent=2, default=str))
print("\nwrote m2_cells.csv, m2_summary.csv, m2_vs_submitted.csv, m2_verdict.json")
