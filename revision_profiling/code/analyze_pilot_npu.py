"""Phase N1 section 5: pilot verdict per beta with the criteria fixed in pilot/PILOT_FAMILIES.md. Reads run JSONs only.
  python3 -B code/analyze_pilot_npu.py
Writes pilot/REPORT_PILOT.md, pilot/pilot_runs.csv, pilot/pilot_criteria.csv.
"""
import sys

sys.dont_write_bytecode = True

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "revision_profiling_npu" / "pilot"
BETAS = (0.1, 0.25)


def load(d):
    out = []
    for p in sorted(Path(d).glob("*__rep*.json")):
        if ".rejected." in p.name or ".npu." in p.name:
            continue
        r = json.loads(p.read_text())
        m = re.match(r"(\w+)_b([\d.]+)_x(\d+)__(combination_\d+)__rep(\d+)", p.stem)
        fam, beta, k, comb, rep = m.group(1), float(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5))
        side = {json.loads(s.read_text())["instance"]: json.loads(s.read_text()) for s in Path(d).glob(p.stem + ".npu.*.json")}
        out.append({"file": p.name, "fam": fam, "beta": beta, "k": k, "comb": comb, "rep": rep, "r": r, "side": side})
    return out


def main():
    plan = json.loads((ROOT / "pilot_plan.json").read_text())
    runs = load(ROOT / "runs")
    reps = load(ROOT / "repeat")
    rows = []
    for x in runs + reps:
        r = x["r"]
        for i in r["instances"]:
            w = i["window"]
            rows.append({"file": x["file"], "fam": x["fam"], "beta": x["beta"], "k": x["k"], "comb": x["comb"], "rep": x["rep"],
                         "instance": i["name"], "device": i["execution"], "rate": i["rate"], "generated": w["generated"],
                         "completed": w["completed"], "dropped": w["dropped_full"], "failed": w["failed"],
                         "cons_gen": w["conservation_gen"], "cons_flow": w["conservation_flow"], "trace": w["trace_check"],
                         "realization": w["realization"], "q_i": w["q_i"], "service_p50": w["service_ms_p50"],
                         "F": r["system"]["throughput_fps"], "D_macro": r["system"]["drop_macro"], "valid": r["valid"],
                         "invalid": ";".join(r["invalid_reasons"]),
                         "close_rc": x["side"].get(i["name"], {}).get("close_rc", "")})
    with open(ROOT / "pilot_runs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    verdict, crit_rows, L = {}, [], ["# Pilot report (Phase N1 §5)", "",
                                      f"Runs: main {len(runs)}, repeat {len(reps)}. Criteria fixed in `PILOT_FAMILIES.md` before the runs.", ""]
    for b in BETAS:
        R = [x for x in runs if x["beta"] == b]
        I = [y for y in rows if y["beta"] == b and y["file"] in {x["file"] for x in R}]
        c = {}
        badA = [y for y in I if y["cons_gen"] or y["cons_flow"] or y["trace"] != "ok" or y["failed"]]
        badA += [{"file": x["file"], "instance": "-"} for x in R if not x["r"]["valid"]]
        c["A"] = (not badA, f"{len(badA)} bad instance-runs")
        worstB, badB = 0.0, []
        g = defaultdict(list)
        for y in I:
            g[(y["fam"], y["k"], y["instance"])].append(y["generated"])
        for key, v in g.items():
            mx, mn = max(v), min(v)
            ok = (mx - mn) <= 2 if mx < 200 else (mx - mn) <= 0.02 * mx
            worstB = max(worstB, (mx - mn) / mx if mx else 0)
            if not ok:
                badB.append(f"{key}:{mn}-{mx}")
        c["B"] = (not badB, f"max relative spread {worstB:.4f}; violations {badB[:4]}")
        c1, c2, info = True, True, []
        for fam, p in plan.items():
            for k, need in ((1, "C1"), (4, "C2")):
                rr = [x for x in R if x["fam"] == fam and x["k"] == k and x["comb"] == p["R1"]]
                if not rr:
                    c1 = c1 and need != "C1"
                    c2 = c2 and need != "C2"
                    info.append(f"{fam} {k}x R1 missing")
                    continue
                s = rr[0]["r"]
                inst = s["instances"]
                if k == 1:
                    comp = min(i["window"]["completed"] / i["window"]["generated"] for i in inst)
                    ok = s["system"]["drop_macro"] <= 0.02 and comp >= 0.95
                    c1 = c1 and ok
                    info.append(f"{fam} 1x R1 D_macro {s['system']['drop_macro']:.4f}, min completion {comp:.3f} -> {ok}")
                else:
                    qmax = max(i["window"]["q_i"] for i in inst)
                    ok = qmax >= 0.05
                    c2 = c2 and ok
                    info.append(f"{fam} 4x R1 max q_i {qmax:.4f} -> {ok}")
        c["C1"] = (c1, "; ".join(t for t in info if " 1x " in t))
        c["C2"] = (c2, "; ".join(t for t in info if " 4x " in t))
        gmin = min(y["generated"] for y in I)
        c["D"] = (gmin >= 30, f"min generated {gmin}")
        rmin = min(y["realization"] for y in I)
        c["E"] = (rmin >= 0.95, f"min realization {rmin:.4f}")
        badF = [x["file"] for x in R if any("residency" in t or "load" in t for t in x["r"]["invalid_reasons"])]
        badF += [f"{y['file']}:{y['instance']}" for y in I if y["device"] != "cpu" and y["close_rc"] not in (0, "0")]
        c["F"] = (not badF, f"{len(badF)} problems")
        ref = {}
        for k in (1, 4):
            for fam in plan:
                kk = [x for x in R if x["fam"] == fam and x["k"] == k]
                ref[f"{fam} {k}x: candidates with drop > 0"] = f"{sum(1 for x in kk if x['r']['system']['dropped_full'] > 0)}/{len(kk)}"
                r2 = [x for x in kk if x["comb"] == plan[fam]["R2"]]
                if r2:
                    ref[f"{fam} {k}x: R2 D_macro"] = f"{r2[0]['r']['system']['drop_macro']:.4f}"
        rp = [x for x in reps if x["beta"] == b]
        g_info = {}
        if rp:
            F = np.array([x["r"]["system"]["throughput_fps"] for x in rp])
            D = np.array([x["r"]["system"]["drop_macro"] for x in rp])
            g_info = {"n": len(rp), "F": F.round(3).tolist(), "F_rel_range": float((F.max() - F.min()) / F.mean()),
                      "F_se": float(F.std(ddof=1) / np.sqrt(len(F))), "D_macro": D.round(4).tolist(),
                      "D_abs_range": float(D.max() - D.min()), "D_se": float(D.std(ddof=1) / np.sqrt(len(D))),
                      "q_i_range": {i["name"]: [round(min(x2["window"]["q_i"] for x in rp for x2 in x["r"]["instances"] if x2["name"] == i["name"]), 4),
                                                round(max(x2["window"]["q_i"] for x in rp for x2 in x["r"]["instances"] if x2["name"] == i["name"]), 4)]
                                    for i in rp[0]["r"]["instances"]}}
            c["G(ref)"] = (g_info["F_rel_range"] <= 0.01 and g_info["D_abs_range"] <= 0.01,
                           f"F rel range {g_info['F_rel_range']:.4f}, D_macro abs range {g_info['D_abs_range']:.4f}")
        base_ok = c["A"][0] and c["B"][0] and c["F"][0]
        sel_ok = base_ok and all(c[k][0] for k in ("C1", "C2", "D", "E"))
        verdict[b] = sel_ok
        L += [f"## beta = {b}", "", "| criterion | pass | detail |", "|---|---|---|"]
        for k, (ok, det) in c.items():
            L.append(f"| {k} | {'PASS' if ok else '**FAIL**'} | {det} |")
            crit_rows.append({"beta": b, "criterion": k, "pass": ok, "detail": det})
        L += ["", "Reference (not criteria):", ""] + [f"- {k}: {v}" for k, v in ref.items()]
        if g_info:
            L += [f"- repeat (rbrsybys R1, 2x, n={g_info['n']}): F {g_info['F']} (rel range {g_info['F_rel_range']:.4f}, SE {g_info['F_se']:.4f}); "
                  f"D_macro {g_info['D_macro']} (abs range {g_info['D_abs_range']:.4f}, SE {g_info['D_se']:.4f}); q_i range {g_info['q_i_range']}"]
        L.append("")
    passing = [b for b in BETAS if verdict[b]]
    choice = min(passing) if passing else None
    L += ["## Selection", "", f"- passing betas: {passing}",
          f"- **selected beta: {choice}**" if choice is not None else "- **no beta passes -> stop and report (user decision)**"]
    with open(ROOT / "pilot_criteria.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["beta", "criterion", "pass", "detail"])
        wr.writeheader()
        wr.writerows(crit_rows)
    (ROOT / "REPORT_PILOT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
