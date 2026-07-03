"""rev6 STEP 0 — binary inventory + mode-toggle probe.

For each YOLO11 detector in {s, m, l, x}:
  1. List every mxq the workstation can resolve for that detector
     (legacy HF cache, model_zoo per-user cache, legacy backup).
  2. Compute SHA256 + size + mtime for each.
  3. Decide which binary each phase in B3 actually loaded.
  4. Probe whether the legacy/global8-compiled binary can be loaded in
     single mode (so multistream could reuse it). Probe is per-model.

Outputs:
  results/binary_inventory_rev6.csv     per-(detector, candidate-path) row
  results/binary_plan_rev6.json         re-measurement plan keyed by detector
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

RES = Path("accv_experiments/results")
RES.mkdir(parents=True, exist_ok=True)

DETECTORS = [
    {"name": "yolo11s", "cls": "YOLO11s"},
    {"name": "yolo11m", "cls": "YOLO11m"},
    {"name": "yolo11l", "cls": "YOLO11l"},
    {"name": "yolo11x", "cls": "YOLO11x"},
]
HF_HUB = Path.home() / ".cache/huggingface/hub"
ZOO = Path.home() / ".mblt_model_zoo/vision/aries"
LEGACY_BACKUP = Path("models/mobilint_backup")


def find_paths(name):
    cands = []
    # HF cache (legacy / mode-suffixed)
    cands += sorted(glob.glob(str(HF_HUB / f"models--mobilint--YOLO11*/snapshots/*/aries/{name}.mxq")))
    for mode in ["global8", "single", "multi"]:
        cands += sorted(glob.glob(str(HF_HUB / f"models--mobilint--YOLO11*/snapshots/*/aries/{mode}/{name}.mxq")))
    # model_zoo cache (per mode)
    for mode in ["global8", "single", "multi"]:
        p = ZOO / mode / f"{name}.mxq"
        if p.exists():
            cands.append(str(p))
    # Legacy backup directory
    p = LEGACY_BACKUP / f"{name}.mxq"
    if p.exists():
        cands.append(str(p))
    # Dedup by realpath
    seen = set(); uniq = []
    for c in cands:
        rp = os.path.realpath(c)
        if rp in seen: continue
        seen.add(rp); uniq.append(c)
    return uniq


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_load(path, infer_mode):
    """Try to load mxq at `path` with the given infer_mode. Returns (ok, err)."""
    try:
        from mblt_model_zoo import vision as mv
        # We need the right class for the detector; infer from filename
        name = Path(path).name.replace(".mxq", "")
        cls_name = name.replace("yolo", "YOLO")  # yolo11s -> YOLO11s
        cls = getattr(mv, cls_name, None)
        if cls is None:
            return False, f"no class for {cls_name}"
        m = cls(local_path=path, infer_mode=infer_mode, product="aries")
        # Light warmup to confirm engine is alive
        import numpy as np
        from step0_compare_devices import CONF, IOU
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        x = m.preprocess(dummy); o = m(x); m.postprocess(o, conf_thres=CONF, iou_thres=IOU)
        try: m.dispose()
        except Exception: pass
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def main():
    inv_rows = []
    plan = {}
    for det in DETECTORS:
        name = det["name"]
        print(f"\n=== {name} ===")
        paths = find_paths(name)
        if not paths:
            print(f"  no mxq found.")
            plan[name] = {"available": [], "decision": "TBD (no mxq)"}
            continue
        # Compute hashes
        info_per_path = []
        for p in paths:
            try:
                sz = os.path.getsize(p); mt = os.path.getmtime(p); sha = sha256(p)
                tag = "legacy" if "mobilint_backup" in p else ("hf_cache" if "huggingface" in p else ("model_zoo" if "mblt_model_zoo" in p else "other"))
                if "global8" in p: tag += ":global8"
                elif "/single/" in p: tag += ":single"
                elif "/multi/" in p: tag += ":multi"
                info = {"detector": name, "tag": tag, "path": p,
                        "sha256": sha, "size_bytes": sz, "mtime": mt}
                inv_rows.append(info)
                info_per_path.append(info)
                print(f"  {tag:<20s}  sha={sha[:12]}  {sz/1e6:>6.2f} MB  {p}")
            except Exception as e:
                print(f"  ERR: {p}: {e}")
        # Decide which binary to standardize on
        # Rule: s -> legacy (b2441f9d); m/l/x -> probe whether global8 binary
        # supports single-mode load. If yes, use global8 binary for all phases.
        # If no, fall back to single binary as the unified one (re-measure baseline+ladder).
        if name == "yolo11s":
            # find legacy binary (preserved in legacy_backup OR hf_cache un-suffixed)
            legacy = next((info for info in info_per_path
                           if "legacy" in info["tag"] or info["tag"] == "hf_cache"), None)
            if legacy is None:
                plan[name] = {"available": [info["sha256"][:12] for info in info_per_path],
                              "decision": "FAIL: no legacy file"}
                continue
            # Test that the legacy file can be loaded in both global8 and single
            ok_g8, err_g8 = probe_load(legacy["path"], "global8")
            print(f"  probe legacy in global8: {'ok' if ok_g8 else 'fail '+err_g8}")
            ok_sg, err_sg = probe_load(legacy["path"], "single")
            print(f"  probe legacy in single : {'ok' if ok_sg else 'fail '+err_sg}")
            plan[name] = {
                "available": [info["sha256"][:12] for info in info_per_path],
                "unified_binary": legacy["path"],
                "unified_sha256": legacy["sha256"],
                "global8_works": ok_g8, "single_works": ok_sg,
                "decision": "re-measure baseline+ladder (global8 legacy) + multistream (single legacy)" if (ok_g8 and ok_sg) else
                            "re-measure with whichever mode probes ok",
            }
        else:
            # Pick the global8 binary (already used for baseline+ladder)
            g8 = next((info for info in info_per_path if info["tag"].endswith(":global8")), None)
            sg = next((info for info in info_per_path if info["tag"].endswith(":single")), None)
            if g8 is None or sg is None:
                plan[name] = {
                    "available": [info["sha256"][:12] for info in info_per_path],
                    "decision": "TBD (missing global8 or single binary)",
                }
                continue
            # Are they already the same hash?
            same_hash = g8["sha256"] == sg["sha256"]
            # Probe whether global8 binary can be loaded in single mode (save baseline re-measure)
            ok_g8_in_single, err_g8_in_single = probe_load(g8["path"], "single")
            print(f"  probe global8-binary in single mode: {'ok' if ok_g8_in_single else 'fail '+err_g8_in_single}")
            # And single-binary in global8 mode (alternative direction)
            ok_sg_in_g8, err_sg_in_g8 = probe_load(sg["path"], "global8")
            print(f"  probe single-binary in global8 mode: {'ok' if ok_sg_in_g8 else 'fail '+err_sg_in_g8}")
            if same_hash:
                decision = "no re-measurement needed (binaries already identical)"
                unified = g8["path"]
            elif ok_g8_in_single:
                decision = "re-measure multistream only with global8 binary forced via local_path; baseline+ladder kept"
                unified = g8["path"]
            elif ok_sg_in_g8:
                decision = "re-measure baseline+ladder only with single binary forced; multistream kept"
                unified = sg["path"]
            else:
                decision = "neither binary works in the other mode; accept binary divergence and document"
                unified = None
            plan[name] = {
                "available": [info["sha256"][:12] for info in info_per_path],
                "global8_path": g8["path"], "global8_sha256": g8["sha256"],
                "single_path":  sg["path"], "single_sha256":  sg["sha256"],
                "global8_in_single_works": ok_g8_in_single,
                "single_in_global8_works": ok_sg_in_g8,
                "unified_binary": unified,
                "decision": decision,
            }

    # Save
    out_csv = RES / "binary_inventory_rev6.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["detector", "tag", "path",
                                          "sha256", "size_bytes", "mtime"])
        w.writeheader()
        for r in inv_rows:
            w.writerow(r)
    out_json = RES / "binary_plan_rev6.json"
    with open(out_json, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\nsaved {out_csv}")
    print(f"saved {out_json}")
    print("\n=== plan ===")
    for det, info in plan.items():
        print(f"  {det}: {info['decision']}")


if __name__ == "__main__":
    main()
