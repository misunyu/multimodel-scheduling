"""Phase N1 section 3 smoke table: counters, conservation, trace_check, end-to-end service time, driver-call time
(sidecar), ResNet big vs small. Reads run JSONs only.
  python3 -B code/analyze_smoke_npu.py <runs_dir> <out_md>
"""
import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path


def main():
    runs = Path(sys.argv[1])
    rows = []
    for p in sorted(runs.glob("*__rep*.json")):
        if ".rejected." in p.name or ".npu." in p.name:
            continue
        r = json.loads(p.read_text())
        i = r["instances"][0]
        w = i["window"]
        side = list(runs.glob(p.stem + ".npu.*.json"))
        s = json.loads(side[0].read_text()) if side else {}
        rows.append({"run": p.stem.split("__")[0].replace("smoke_", ""), "key": i["model"], "dev": i["execution"],
                     "gen": w["generated"], "comp": w["completed"], "drop": w["dropped_full"], "fail": w["failed"],
                     "cons": (w["conservation_gen"], w["conservation_flow"]), "trace": w["trace_check"],
                     "real": w["realization"], "svc50": w["service_ms_p50"], "lat50": w["latency_ms_p50"],
                     "pre": w["preprocess_ms_mean"], "drv50": s.get("driver_call_ms_p50"), "close": s.get("close_rc"),
                     "valid": r["valid"], "inv": r["invalid_reasons"]})
    L = ["# Smoke (Phase N1 §3): 1 instance, 2 req/s, warm-up 5 s + window 10 s", "",
         "| run | registry key | device | generated | completed | dropped | failed | conservation (gen, flow) | trace_check | realization | service p50 ms (end-to-end) | latency p50 ms | preprocess ms | driver call p50 ms | Close rc | valid |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
        L.append(f"| {x['run']} | {x['key']} | {x['dev']} | {x['gen']} | {x['comp']} | {x['drop']} | {x['fail']} | {x['cons']} | {x['trace']} | "
                 f"{f(x['real'])} | {f(x['svc50'])} | {f(x['lat50'])} | {f(x['pre'])} | {f(x['drv50'])} | {f(x['close'])} | {x['valid']} |")
    by = {(x["run"]): x for x in rows}
    L += ["", "## ResNet50 big vs small (end-to-end service p50 / driver call p50, ms)", ""]
    for m in ("resnet50", "yolov3"):
        for d in ("npu0", "npu1"):
            b, s = by.get(f"{m}_big_{d}"), by.get(f"{m}_small_{d}")
            if b and s:
                L.append(f"- {m} {d}: big {b['svc50']:.2f} / {b['drv50']:.2f}, small {s['svc50']:.2f} / {s['drv50']:.2f}, "
                         f"diff (big - small) {b['svc50'] - s['svc50']:+.2f} / {b['drv50'] - s['drv50']:+.2f}")
    ok = all(x["valid"] and x["cons"] == (0, 0) and x["trace"] == "ok" and x["fail"] == 0 for x in rows)
    L += ["", f"All runs valid, conservation 0, trace_check ok, failed 0: **{ok}** ({len(rows)} runs)"]
    Path(sys.argv[2]).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
