"""Phase N1 sections 4-5: reference placements (fixed from the isolated profile only, before any pilot run), pilot
schedules and the run plan. Writes revision_profiling_npu/pilot/PILOT_FAMILIES.md and pilot/schedules/*.yaml.

R1 (rule written before the pilot): every instance i has a fixed variant v_i (from the candidate set). Its options are
  cpu (common CPU ONNX), npu0 and npu1 (its own variant binary), each with the isolated closed-loop throughput.
  Instances are taken in descending order of their best option's throughput (ties: instance name); each takes its
  best option still available (an NPU holds at most one instance; the CPU is never exhausted).
R2: all instances on the CPU.

  python3 -B code/pilot_plan_npu.py
"""
import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import yaml

from gen_schedules_npu import build_family, read_family, split_name

ROOT = Path(__file__).resolve().parent.parent.parent / "revision_profiling_npu"
FAMS = ("rbybys", "rbrsybys")        # N = 3 and N = 4 from the evaluation set (section 0-1)
BETAS = (0.1, 0.25)
MULTS = (1, 4)
REPEAT_MULT, REPEATS = 2, 3
SEC_PER_RUN_EST = 45.0


def options(entries, name):
    m, v = split_name(name)
    opts = {"cpu": entries[f"{m}_cpu"]["throughput_fps"]}
    for d in ("npu0", "npu1"):
        opts[d] = entries[f"{m}_{v}_{d}"]["throughput_fps"]
    return opts


def r1(entries, names):
    opts = {n: options(entries, n) for n in names}
    order = sorted(names, key=lambda n: (-max(opts[n].values()), n))
    free, place = {"npu0", "npu1"}, {}
    for n in order:
        for d, _ in sorted(opts[n].items(), key=lambda kv: -kv[1]):
            if d == "cpu" or d in free:
                place[n] = d
                free.discard(d)
                break
    return place, order, opts


def find_combo(base, place):
    hits = [c for c, devs in base.items() if devs == place]
    assert len(hits) == 1, (place, hits)
    return hits[0]


def main():
    prof = json.loads((ROOT / "isolated" / "offline_profile.json").read_text())
    entries, cmax = prof["entries"], prof["cmax"]
    sd = ROOT / "pilot" / "schedules"
    sd.mkdir(parents=True, exist_ok=True)
    L = ["# Pilot families and reference placements (Phase N1 §4-§5) — fixed before any pilot run", "",
         "Source: `revision_profiling_npu/isolated/offline_profile.json` only.", "",
         "Cmax (req/s): " + ", ".join(f"{m} {c['throughput_fps']:.2f} ({c['device']}/{c['variant']})" for m, c in cmax.items()), "",
         "R1 rule: instances in descending order of their best isolated option (cpu / npu0 / npu1 with the instance's own "
         "variant binary); each takes its best option still free (NPU <= 1 instance, CPU unlimited). R2: all CPU.", ""]
    plan = {}
    n_runs = 0
    for fam in FAMS:
        base, _ = read_family(fam)
        names = sorted(next(iter(base.values())))
        place1, order, opts = r1(entries, names)
        c1 = find_combo(base, place1)
        c2 = find_combo(base, {n: "cpu" for n in names})
        plan[fam] = {"R1": c1, "R2": c2, "R1_placement": place1, "candidates": len(base)}
        L += [f"## {fam} (N = {len(names)}, {len(base)} candidates)", "",
              "| instance | cpu | npu0 | npu1 | (req/s, isolated) |", "|---|---|---|---|---|"]
        for n in names:
            L.append(f"| {n} | {opts[n]['cpu']:.2f} | {opts[n]['npu0']:.2f} | {opts[n]['npu1']:.2f} | |")
        L += ["", f"- order: {', '.join(order)}",
              f"- **R1 = {c1}**: " + ", ".join(f"{n}:{place1[n]}" for n in names),
              f"- **R2 = {c2}**: all cpu", ""]
        for b in BETAS:
            for k in MULTS + (REPEAT_MULT,):
                y = build_family(fam, b, k, cm={m: c["throughput_fps"] for m, c in cmax.items()})
                (sd / f"{fam}_b{b}_x{k}.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
            rates = {n: round(b * cmax[split_name(n)[0]]["throughput_fps"] * k, 3) for n in names for k in (1,)}
            L.append(f"- beta {b}: rate per instance at 1x = {rates} (x4: four times)")
        n_runs += len(base) * len(BETAS) * len(MULTS)
        L.append("")
    rep = REPEATS * len(BETAS)
    L += ["## Run plan", "",
          f"- main: (13 + 21) candidates x beta {list(BETAS)} x rate {list(MULTS)} = {n_runs} runs",
          f"- repeat: rbrsybys R1 ({plan['rbrsybys']['R1']}) at {REPEAT_MULT}x, {REPEATS} repeats x {len(BETAS)} beta = {rep} runs",
          f"- total {n_runs + rep} runs; estimate {SEC_PER_RUN_EST:.0f} s/run -> {(n_runs + rep) * SEC_PER_RUN_EST / 3600:.2f} h "
          "(to be replaced by the measured run time)", "",
          "## Criteria (fixed before the pilot; values are not changed after seeing results)", "",
          "- (A) every run: timestamp conservation (gen, flow) = 0, trace_check ok, failed = 0",
          "- (B) per workload (family, beta, k), per instance: generated spread over candidates <= 2 % of the max "
          "(instances with < 200 generated: <= 2 requests)",
          "- (C1) 1x: R1 D_macro <= 0.02 and every instance completed/generated >= 0.95; "
          "(C2) 4x: R1 has an instance with q_i >= 0.05 (each pilot family)",
          "- (D) lowest-rate instance generated >= 30 in the window; (E) realization >= 0.95 for every instance",
          "- (F) residency violations 0, NPU load failures 0",
          "- (G, reference) repeat: F relative range <= 1 %, D_macro absolute range <= 0.01",
          "- beta: among betas passing A, B, F, the smallest passing C1, C2, D, E; none -> stop and report"]
    (ROOT / "pilot" / "PILOT_FAMILIES.md").write_text("\n".join(L) + "\n")
    (ROOT / "pilot" / "pilot_plan.json").write_text(json.dumps(plan, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
