"""Section 5.1 pass table for single-instance 1x runs. Criteria are fixed by the spec; nothing is tuned here.
  python3 -B code/analyze_smoke.py <runs_dir> <out_csv>"""
import sys

sys.dont_write_bytecode = True
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
prof = json.loads((ROOT / "offline_profile" / "offline_profile.json").read_text())["models"]
rows = []
for p in sorted(Path(sys.argv[1]).glob("*__rep*.json")):
    if ".rejected." in p.name:
        continue
    r = json.loads(p.read_text())
    i = r["instances"][0]
    w = i["window"]
    dev, m = i["execution"], i["model"]
    rm = i["rate"]
    ccpu = prof[m]["cpu"]["throughput_fps"]
    off_p50 = prof[m][dev]["service_ms_p50"]
    comp_rate = w["completed"] / r["timing"]["window_sec_actual"]
    lat_ratio = w["service_ms_p50"] / off_p50 if w["service_ms_p50"] else None
    q = w["dropped_full"] / w["generated"] if w["generated"] else None
    c = {
        "modules_sha": (not r["module_violations"]) and all(v["match"] for v in i["files_checked"].values()),
        "no_failed": w["failed"] == 0 and w["consumer_failed"] == 0,
        "conservation_inflight": w["conservation_gen"] == 0 and w["conservation_flow"] == 0 and w.get("trace_check", "ok") == "ok" and w["inflight_start"] <= 1 and w["inflight_end"] <= 1 and i["inflight_sequential_ok"],
        "realization": w["realization"] is not None and w["realization"] >= 0.95,
        "latency_pm20": lat_ratio is not None and 0.8 <= lat_ratio <= 1.2,
    }
    if dev == "gpu":
        regime = "gpu"
        c["completion"] = comp_rate >= 0.9 * rm
    else:
        not_over = comp_rate >= 0.9 * rm and q is not None and q < 0.01
        over = 0.8 * ccpu <= comp_rate <= 1.2 * ccpu and w["dropped_full"] > 0
        if rm < 0.8 * ccpu:
            regime, c["completion"] = "cpu_not_overloaded", not_over
        elif rm >= 1.2 * ccpu:
            regime, c["completion"] = "cpu_overloaded", over
        else:
            regime, c["completion"] = "cpu_boundary(" + ("not_overloaded_rule" if not_over else "overloaded_rule" if over else "none") + ")", (not_over or over)
    rows.append({"model": m, "device": dev, "r_m": round(rm, 3), "c_cpu": round(ccpu, 3), "regime": regime,
                 "scheduled_ticks": w["scheduled_ticks"], "generated": w["generated"], "missed_ticks": w["missed_ticks"],
                 "realization": round(w["realization"], 4), "enqueued": w["enqueued"], "dropped_full": w["dropped_full"],
                 "completed": w["completed"], "completed_per_s": round(comp_rate, 3), "failed": w["failed"],
                 "consumer_failed": w["consumer_failed"], "pending_start": w["pending_start"], "pending_end": w["pending_end"],
                 "conservation_flow": w["conservation_flow"], "service_p50_ms": round(w["service_ms_p50"], 3) if w["service_ms_p50"] else None,
                 "offline_p50_ms": round(off_p50, 3), "latency_ratio": round(lat_ratio, 3) if lat_ratio else None,
                 "service_p95_over_p50": round(w["service_ms_p95"] / w["service_ms_p50"], 2) if w["service_ms_p50"] else None,
                 "tick_delay_p95_ms": round(w["tick_delay_ms_p95"], 3), "tick_delay_max_ms": round(w["tick_delay_ms_max"], 3),
                 "gpu_util_mean": r["gpu_samples"]["util_mean"],
                 **{"ok_" + k: v for k, v in c.items()}, "pass_all": all(c.values()), "valid": r["valid"], "execution_id": r["execution_id"]})
with open(sys.argv[2], "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
for x in rows:
    print(f"{x['model']:22s} {x['device']} r={x['r_m']:8.2f} ccpu={x['c_cpu']:7.2f} {x['regime']:34s} sched={x['scheduled_ticks']:5d} gen={x['generated']:5d} miss={x['missed_ticks']:3d} "
          f"drop={x['dropped_full']:4d} comp={x['completed']:5d} lat={x['latency_ratio']} "
          + " ".join(k[3:] + "=" + ("Y" if x[k] else "N") for k in x if k.startswith("ok_")) + f" PASS={x['pass_all']}")
