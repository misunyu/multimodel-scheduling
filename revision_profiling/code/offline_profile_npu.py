"""Phase N1 section 4: isolated capacity per (model, variant, device) from the closed-loop runs
(revision_profiling_npu/isolated/runs, warm-up 5 s + window 30 s), and Cmax per model = max over (device, variant).
Output: revision_profiling_npu/isolated/offline_profile.json and OFFLINE_PROFILE.md.
  python3 -B code/offline_profile_npu.py
"""
import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent / "revision_profiling_npu" / "isolated"


def main():
    entries = {}
    for p in sorted((ROOT / "runs").glob("closed_*__rep0.json")):
        r = json.loads(p.read_text())
        i = r["instances"][0]
        w = i["window"]
        tag = p.stem.split("__")[0][len("closed_"):]              # resnet50_big_npu0 | resnet50_cpu
        parts = tag.split("_")
        model, dev = parts[0], parts[-1]
        variant = parts[1] if len(parts) == 3 else "common"
        side = list((ROOT / "runs").glob(p.stem + ".npu.*.json"))
        s = json.loads(side[0].read_text()) if side else {}
        entries[tag] = {"model": model, "variant": variant, "device": dev, "registry_key": i["model"],
                        "throughput_fps": r["system"]["throughput_fps"], "completed": w["completed"],
                        "service_ms_p50": w["service_ms_p50"], "service_ms_p95": w["service_ms_p95"],
                        "service_ms_mean": w["service_ms_mean"], "preprocess_ms_mean": w["preprocess_ms_mean"],
                        "driver_call_ms_p50": s.get("driver_call_ms_p50"), "window_sec": r["timing"]["window_sec_actual"],
                        "valid": r["valid"], "invalid_reasons": r["invalid_reasons"], "execution_id": r["execution_id"]}
    assert len(entries) == 10, sorted(entries)
    cmax = {}
    for m in ("resnet50", "yolov3"):
        best = max((e for e in entries.values() if e["model"] == m), key=lambda e: e["throughput_fps"])
        cmax[m] = {"throughput_fps": best["throughput_fps"], "device": best["device"], "variant": best["variant"]}
    prof = {"definition": "closed-loop single instance, warm-up 5 s + window 30 s; throughput = completed / window; "
                          "service = dequeue -> output put (preprocess + host stages + NPU round trip + postprocess). "
                          "Cmax[m] = max throughput over (device, variant) of model m.",
            "entries": entries, "cmax": cmax}
    (ROOT / "offline_profile.json").write_text(json.dumps(prof, indent=1))
    cpu = {m: entries[f"{m}_cpu"]["throughput_fps"] for m in ("resnet50", "yolov3")}
    L = ["# Isolated profile (Phase N1 §4)", "", prof["definition"], "",
         "| model | variant | device | req/s | service p50 ms | service p95 ms | driver call p50 ms | NPU / CPU | valid |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, e in sorted(entries.items(), key=lambda kv: (kv[1]["model"], kv[1]["variant"], kv[1]["device"])):
        d = e["driver_call_ms_p50"]
        L.append(f"| {e['model']} | {e['variant']} | {e['device']} | {e['throughput_fps']:.2f} | {e['service_ms_p50']:.2f} | "
                 f"{e['service_ms_p95']:.2f} | {'-' if d is None else f'{d:.2f}'} | {e['throughput_fps'] / cpu[e['model']]:.2f} | {e['valid']} |")
    L += ["", "## Cmax (reference rate = beta x Cmax x k)", ""]
    for m, c in cmax.items():
        L.append(f"- {m}: {c['throughput_fps']:.2f} req/s at {c['device']} / {c['variant']}")
    L += ["", "## npu0 vs npu1 (PCIe Gen4x8 vs Gen2x1)", ""]
    for m in ("resnet50", "yolov3"):
        for v in ("big", "small"):
            a, b = entries[f"{m}_{v}_npu0"], entries[f"{m}_{v}_npu1"]
            L.append(f"- {m} {v}: npu0 {a['throughput_fps']:.2f}, npu1 {b['throughput_fps']:.2f} req/s "
                     f"(npu1/npu0 = {b['throughput_fps'] / a['throughput_fps']:.3f})")
    (ROOT / "OFFLINE_PROFILE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
