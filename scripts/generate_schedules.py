#!/usr/bin/env python3
"""v27 schedule generation from ranking artefacts (configuration (b)).

The candidate ORDER is read from rankings/ranking_*.json and never written by hand --
hand-written candidate orders in YAML are the root of the phantom incident. Each emitted
schedule carries provenance (ranking path + sha256, alpha/beta, working set, date) as
header comments, so a schedule can always be traced back to the ranking that produced it.

Structure per scenario (mirrors the published runs, docs/gate_b_schedule_provenance.md):

    combination_stable  : all models on the accelerator, low rate  (feasible)
    combination_burst   : the ranking's top-1 placement, burst rate (the misprediction)
    cand_1 .. cand_N    : ranking ranks 1..N, burst rate

The four Q6 ablation corners and the Adaptive baseline share this one schedule and are
distinguished only by the executor's adaptive mode / trigger flags, never by the file.
"""
import argparse, hashlib, json, os, sys
from datetime import date

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT, "schedules")

# Documented configuration of the published (vision-3) runs -- docs/gate_d_config_attribution.md
SLO_UNIFORM_MS = 15
GEN_FPS = 2
# failure phase per scenario:
#   "top1"   -- the misprediction scenarios: the ranking's top choice IS the failure
#               (generative kept on the accelerator).
#   "allcpu" -- Q1.3 and the 4b control: the failure is everything on the CPU. Confirmed
#               from the logs (gpu_/npu_Static run all-CPU) and from
#               c2_reactive_baseline_report ("burst=all-CPU로 재실행"). Emitting top-1 here
#               would emit the RECOVERED state as the failure and no violation would occur.
FAILURE = {"q13_persistence_gpu": "allcpu", "q13_persistence_npu": "allcpu",
           "q4b_control": "allcpu",
           # Q1.5's contrast axis is "relocate vs stay put", so the failure must be the
           # all-CPU state Static is frozen in; emitting top-1 would freeze Static in the
           # RECOVERED placement and there would be nothing to compare.
           "q15_buffer_gpu": "allcpu", "q15_buffer_npu": "allcpu",
           # Q1.1 never leaves combination_stable; the field is irrelevant but set for
           # consistency with the other vision-3 scenarios.
           "q11_steady_gpu": "allcpu", "q11_steady_npu": "allcpu",
           # C3: the accumulation phase IS the all-CPU placement -- that is where
           # dQ/dt = lambda - mu_cpu is measured. Emitting top-1 would put accumulation on
           # the accelerator, where nothing accumulates.
           "c3_fluid_gpu_070": "allcpu", "c3_fluid_gpu_090": "allcpu",
           "c3_fluid_gpu_095": "allcpu", "c3_fluid_npu_070": "allcpu",
           "c3_fluid_npu_090": "allcpu", "c3_fluid_npu_095": "allcpu",
           "c3_mustar_gpu": "allcpu", "c3_mustar_npu": "allcpu",
           "c3_mudrain_gpu": "allcpu", "c3_mudrain_npu": "allcpu"}

SCENARIOS = {
    # scenario   : (ranking file,                                   lam_stable, lam_burst)
    "q3_misprediction": ("rankings/ranking_Q3Q6_paper_v3_cpu-gpu.json", 25, 80),
    "q6_ablation":      ("rankings/ranking_Q3Q6_paper_v3_cpu-gpu.json", 25, 80),
    "q5_misprediction": ("rankings/ranking_Q5_paper_v3_cpu-npu.json",   25, 80),
    # v28: lambdas are the documented ones (docs/c2_reactive_baseline_report.md section 2,
    # docs/q4_experiment_report.md). These working sets carry NO background generative --
    # recorded explicitly as "none" below so a missing background is distinguishable from
    # an omitted one (v26 lost a whole sweep to that ambiguity).
    "q4b_control":      ("rankings/ranking_Q4b_control_v3_cpu-gpu.json", 25, 45),
    "q4_infeasible":    ("rankings/ranking_Q4_heavy4_cpu-gpu.json",      25, 90),
    # v29: Q1.3 (failure persistence). Same vision-3 working set as the 4b control, no
    # background. Its lambda was never recorded, so lam_burst below is a PLACEHOLDER that
    # the v29 search replaces; the buffer is likewise unrecorded and the search uses the
    # 4b value (2), documented for this same working set.
    # v29 confirmed: the condition holds across the whole tested range (30..400 GPU,
    # 30..250 NPU) because buffer=2 turns excess arrivals into drops rather than latency,
    # so no upper bound exists and the midpoint rule is undefined. Fallback rule applied:
    # lambda = 0.9*mu*, the project's documented target ratio (docs/b2_buffer_sweep_report.md
    # "λ=0.9μ*(GPU 123 / NPU 93 fps)"). Reproduced 2/2 on both platforms.
    # v32: lambda re-determined with the DROP condition added. The v29 values (123/93)
    # applied 0.9*mu* PER VIEW although mu* is a shared pipeline rate, so demand was ~2.7x
    # capacity; V stayed low only because buffer=2 turned the excess into drops, and the
    # executor's saturation guard then fired on every candidate (25/25). Confirmed
    # zero-drop intervals: GPU [30,55], NPU [30,52]; 45 lies in both and unifies with the
    # 4b control on the identical vision-3 workload (v26 task 116 rule 3).
    "q13_persistence_gpu": ("rankings/ranking_Q1_3_persist_v3_cpu-gpu.json", 25, 45),
    "q13_persistence_npu": ("rankings/ranking_Q1_3_persist_v3_cpu-npu.json", 25, 45),
    # v34 Q1.5 (tab:b2 buffer sweep). Same vision-3 working set and ranking as Q1.3 -- the
    # published B2 schedule evaporated, so its working set is UNKNOWN beyond "yolo11s was
    # in it" (docs/supplementary_conditions_v34.md). lam_burst is 0.9*mu* expressed as TOTAL
    # demand and divided across the three views (GPU 123/3=41, NPU 93/3=31); the published
    # 123/93 were applied per view, i.e. 2.7x capacity -- the same error v32 found in Q1.3.
    "q15_buffer_gpu": ("rankings/ranking_Q1_3_persist_v3_cpu-gpu.json", 25, 41),
    "q15_buffer_npu": ("rankings/ranking_Q1_3_persist_v3_cpu-npu.json", 25, 31),
    # v34 Q1.1 (steady-state overhead). Only combination_stable is executed; the burst and
    # candidates exist in the file but are never reached (--stop-after). The rate is low and
    # uniform so V(t) ~ 0 and any delta between execution modes is monitoring cost.
    "q11_steady_gpu": ("rankings/ranking_Q1_3_persist_v3_cpu-gpu.json", 4, 4),
    "q11_steady_npu": ("rankings/ranking_Q1_3_persist_v3_cpu-npu.json", 4, 4),
    # v35 C3 fluid-model revalidation. Same vision-3 working set as the old C3
    # (docs/c3_stress_report.md: "vision-only x cpu_gpu/cpu_npu", bottleneck yolo11s).
    #
    # lambda is anchored on the BOTTLENECK VIEW's pipeline service rate measured IN THE
    # DRAIN REGIME on this system: yolo11s 121 fps (GPU) / 86 fps (NPU). The inherited mu*
    # (137/103) does not hold here -- a pilot at 0.95*137 drained at ~237 fps where the
    # model predicts 7 -- and the saturated measurement (view1 79/65 fps) is taken under
    # 1200 fps of contention, so it is not the rate that applies while a backlog drains.
    # mu* is load-dependent; docs/c3_revalidation_conditions.md carries the full record.
    #
    #   GPU 121 fps -> 0.70/0.90/0.95 = 85/109/115 per view
    #   NPU  86 fps -> 0.70/0.90/0.95 = 60/77/82  per view
    #
    # Both phases carry the SAME rate: the experiment measures accumulation and drain
    # slopes at one arrival rate, so varying lambda between phases would confound the slope
    # with a rate change.
    "c3_fluid_gpu_070": ("rankings/ranking_Q1_3_persist_v3_cpu-gpu.json", 85, 85),
    "c3_fluid_gpu_090": ("rankings/ranking_Q1_3_persist_v3_cpu-gpu.json", 109, 109),
    "c3_fluid_gpu_095": ("rankings/ranking_Q1_3_persist_v3_cpu-gpu.json", 115, 115),
    "c3_fluid_npu_070": ("rankings/ranking_Q1_3_persist_v3_cpu-npu.json", 60, 60),
    "c3_fluid_npu_090": ("rankings/ranking_Q1_3_persist_v3_cpu-npu.json", 77, 77),
    "c3_fluid_npu_095": ("rankings/ranking_Q1_3_persist_v3_cpu-npu.json", 82, 82),
    # mu* measurement runs (not part of the sweep; they FIX its scale).
    #   c3_mustar_*  : all-accelerator driven far past capacity -> saturated throughput
    #   c3_mudrain_* : lambda=20/view, so the drain slope plus lambda gives the per-view
    #                  service rate in the regime the sweep actually operates in
    "c3_mustar_gpu": ("rankings/ranking_Q1_3_persist_v3_cpu-gpu.json", 400, 400),
    "c3_mustar_npu": ("rankings/ranking_Q1_3_persist_v3_cpu-npu.json", 400, 400),
    "c3_mudrain_gpu": ("rankings/ranking_Q1_3_persist_v3_cpu-gpu.json", 20, 20),
    "c3_mudrain_npu": ("rankings/ranking_Q1_3_persist_v3_cpu-npu.json", 20, 20),
}
N_CAND = 5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def placement_of(entry, ws):
    return {v["model"]: v["device"] for v in entry["placement"].values()}


def emit_combo(name, placement, ws, lam, gens):
    lines = [f"{name}:"]
    vi = 0
    for m in ws:
        if m in gens:
            continue
        vi += 1
        lines += [f"    m{vi}:",
                  f"        display: view{vi}",
                  f"        execution: {placement[m]}",
                  f"        infps: {lam}",
                  f"        slo_ms: {SLO_UNIFORM_MS}",
                  f"        model: {m}"]
    for gi, m in enumerate(gens, 1):
        lines += [f"    bg{gi}:",
                  f"        display: none",
                  f"        execution: {placement[m]}",
                  f"        infps: {GEN_FPS}",
                  f"        model: {m}"]
    return lines


def generate(scenario, ranking_rel, lam_stable, lam_burst):
    rpath = os.path.join(PROJECT, ranking_rel)
    art = json.load(open(rpath))
    prov = art["_provenance"]
    ws = prov["workload"]["working_set"]
    plat = prov["platform"]
    acc = {"cpu-gpu": "gpu", "cpu-npu": "npu"}[plat]
    gens = [m for m in ws if m in ("llama1b", "qwen2_vl")]
    R = art["ranking"]

    head = [
        f"# GENERATED by scripts/generate_schedules.py on {date.today().isoformat()} -- do not hand-edit.",
        f"# scenario      : {scenario}",
        f"# platform      : {plat}",
        f"# working set   : {ws}  (foreground vision = {[m for m in ws if m not in gens]}, "
        f"background generative = {gens})",
        f"# candidate order: READ FROM the ranking below (ranks 1..{N_CAND}); never hand-written.",
        f"# ranking file  : {ranking_rel}",
        f"# ranking sha256: {sha256(rpath)}",
        f"# alpha / beta  : {prov['targets']['alpha']} / {prov['targets']['beta']}",
        f"# lambda        : stable {lam_stable} fps/view, burst {lam_burst} fps/view "
        f"(generative {GEN_FPS} req/s); uniform L_SLO {SLO_UNIFORM_MS} ms",
        f"# active background: {gens if gens else 'none'}",
        f"# failure phase : {FAILURE.get(scenario, 'top1')}"
        f"{'  (combination_burst = all models on CPU)' if FAILURE.get(scenario)=='allcpu' else '  (combination_burst = ranking top-1)'}",
        "#",
        "# The Q6 ablation corners and the Adaptive baseline use THIS file unchanged and are",
        "# distinguished by executor mode/trigger flags only.",
        "",
    ]

    body = []
    all_accel = {m: acc for m in ws}
    body += emit_combo("combination_stable", all_accel, ws, lam_stable, gens) + [""]
    fail_mode = FAILURE.get(scenario, "top1")
    burst_placement = ({m: "cpu" for m in ws} if fail_mode == "allcpu"
                       else placement_of(R[0], ws))
    body += emit_combo("combination_burst", burst_placement, ws, lam_burst, gens) + [""]
    for i in range(min(N_CAND, len(R))):
        body += emit_combo(f"cand_{i+1}", placement_of(R[i], ws), ws, lam_burst, gens) + [""]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{scenario}_{plat}.yaml")
    open(out, "w").write("\n".join(head + body))

    # Static never reconfigures (adaptive mode 3), so it must START in the failure
    # placement. Give it a schedule containing ONLY that phase -- running the normal
    # schedule left Static on the healthy stable placement and V(t) never rose
    # (caught by the stage-1 gate, v31).
    fo = os.path.join(OUT_DIR, f"{scenario}_{plat}_failure_only.yaml")
    open(fo, "w").write("\n".join(
        head[:-1]
        + ["# STATIC-ONLY: only the failure phase; Static must start here.", ""]
        + emit_combo("combination_burst", burst_placement, ws, lam_burst, gens) + [""]))

    # v33-C: variant C (dwell kept, progress REMOVED) needs the candidate list to be the
    # SAME placement five times -- "progress 제거 = top-1 재적용" (c2_reactive_baseline
    # §2). Giving C the normal ranked list let it descend to cand_5 and behave exactly
    # like BoundGuard. The placement is READ FROM THE RANKING (rank 1), never hardcoded.
    ra = os.path.join(OUT_DIR, f"{scenario}_{plat}_reapply_top1.yaml")
    top1 = placement_of(R[0], ws)
    ra_body = emit_combo("combination_stable", all_accel, ws, lam_stable, gens) + [""]
    ra_body += emit_combo("combination_burst", burst_placement, ws, lam_burst, gens) + [""]
    for i in range(N_CAND):
        ra_body += emit_combo(f"cand_{i+1}", top1, ws, lam_burst, gens) + [""]
    open(ra, "w").write("\n".join(
        head[:-1]
        + ["# RE-APPLY-TOP1: cand_1..cand_5 are all the ranking's rank-1 placement.",
           "# For variant C (dwell without progress). Same ranking/lambda/buffer as the",
           "# normal schedule; only the candidate list differs.", ""]
        + ra_body))
    return out, R[:N_CAND], ws, gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    a = ap.parse_args()
    todo = {a.scenario: SCENARIOS[a.scenario]} if a.scenario else SCENARIOS
    for sc, (rk, ls_, lb) in todo.items():
        out, top, ws, gens = generate(sc, rk, ls_, lb)
        print(f"[gen] {sc}: {os.path.relpath(out, PROJECT)}")
        for i, e in enumerate(top, 1):
            tag = "".join('a' if v["device"] != "cpu" else 'c'
                          for m in ws for v in e["placement"].values() if v["model"] == m)
            print(f"        cand_{i} = rank {e['rank']}  {tag}  (score {e['pred_score']})")


if __name__ == "__main__":
    sys.exit(main())
