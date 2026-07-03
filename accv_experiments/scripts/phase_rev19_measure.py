"""rev19 — threads=4 operating-point measurement.

Lever: torch.set_num_threads(4) (verified box-identical to threads=24; only speed).
Pinned: mxq b2441f9d, global8, yolo11s. No core eval script modified.

STEP 2: Table 1 rebuild — single-stream L0, 24 logs, GPU+NPU, per-size sAP AND
        offline mAP + infer + skip, 3 reps. (CPU reused from rev7 — device is
        thread-independent.)
STEP 4: Table 3 expansion — N in {2,4} at L1_light, 5 strategies incl Oracle
        (full placement enumeration: N=2->4, N=4->16), worst & mean sAP, 3 reps.
        N=8 named strategies already in rev11_n8.csv (threads=4, 3 reps) — reused.

Outputs:
  results/rev19_table1_threads4.csv
  results/rev19_main_comparison_threads4.csv
  results/manifest_rev19.json
"""

from __future__ import annotations

import csv, json, itertools, sys, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "minimal_pipeline"))

from _step_d_common import (load_val, load_split_for_sid)
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                set_active_npu_engines, dispose_npu_for,
                                measure_single_stream, measure_multistream,
                                DETECTORS)
from step0_compare_devices import FPS

RES = Path("accv_experiments/results")
MAN = RES / "manifest_rev19.json"
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
PERIOD = 1000.0 / FPS
THREADS = 4
N_REPS = 3
COMP_A_4 = [2, 22, 3, 21]    # 2,22 small-rich; 3,21 large-rich
COMP_A_2 = [2, 21]           # 1 small-rich + 1 large-rich
SIZE_GROUP = {2: "small", 22: "small", 3: "large", 21: "large"}


def manifest():
    if MAN.exists():
        try: return json.loads(MAN.read_text())
        except Exception: pass
    return {"done": {}}

def save_man(m): MAN.write_text(json.dumps(m, indent=2))


# ---- STEP 2 ----
T1_COLS = ["rep", "device", "n_sids", "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
            "map_5095", "map_s", "map_m", "map_l", "infer_ms", "skip_pct"]

def step2_table1(val):
    out = RES / "rev19_table1_threads4.csv"
    if out.exists(): out.unlink()
    n_sids = len(val["sequences"])
    gpu = FGModelGPUGeneric(DET["ultralytics_pt"])
    npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)[0]
    set_active_npu_engines([npu])
    measure_single_stream(0, load_split_for_sid(val, 0), "NPU", npu, "L0")  # warmup
    rows = []
    for rep in range(N_REPS):
        torch.set_num_threads(THREADS)
        for device, model in [("GPU", gpu), ("NPU", npu)]:
            acc = {k: [] for k in ["sap_5095","sap_50","sap_s","sap_m","sap_l",
                                     "map_5095","map_s","map_m","map_l","infer","skip"]}
            for sid in range(n_sids):
                split = load_split_for_sid(val, sid)
                if device == "NPU": set_active_npu_engines([npu])
                m = measure_single_stream(sid, split, device, model, "L0")
                acc["sap_5095"].append(m["sap_5095"]); acc["sap_50"].append(m["sap_50"])
                acc["sap_s"].append(m["sap_s"]); acc["sap_m"].append(m["sap_m"]); acc["sap_l"].append(m["sap_l"])
                acc["map_5095"].append(m["map_5095"]); acc["map_s"].append(m["map_s"])
                acc["map_m"].append(m["map_m"]); acc["map_l"].append(m["map_l"])
                acc["infer"].append(m["latency_mean"]); acc["skip"].append(m["frame_skip_pct"])
            rows.append({"rep": rep, "device": device, "n_sids": n_sids,
                          "sap_5095": round(np.mean(acc["sap_5095"]),4), "sap_50": round(np.mean(acc["sap_50"]),4),
                          "sap_s": round(np.mean(acc["sap_s"]),4), "sap_m": round(np.mean(acc["sap_m"]),4),
                          "sap_l": round(np.mean(acc["sap_l"]),4),
                          "map_5095": round(np.mean(acc["map_5095"]),4), "map_s": round(np.mean(acc["map_s"]),4),
                          "map_m": round(np.mean(acc["map_m"]),4), "map_l": round(np.mean(acc["map_l"]),4),
                          "infer_ms": round(np.mean(acc["infer"]),3), "skip_pct": round(np.mean(acc["skip"]),2)})
            print(f"  T1 rep{rep} {device}: sap_l={rows[-1]['sap_l']:.4f} map_l={rows[-1]['map_l']:.4f} "
                    f"infer={rows[-1]['infer_ms']:.1f} skip={rows[-1]['skip_pct']:.1f}%")
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=T1_COLS); w.writeheader()
        for r in rows: w.writerow(r)
    dispose_npu_for("yolo11s"); del gpu
    print(f"saved {out}")


# ---- STEP 4 ----
MC_COLS = ["rep","N","bg","strategy","placement","worst_sap","mean_sap","npu_skip_max"]

def all_placements(sids):
    """All 2^N device assignments; return list of (name, placement_list)."""
    out=[]
    for combo in itertools.product(["GPU","NPU"], repeat=len(sids)):
        out.append(combo)
    return out

def named_strategy(sids, kind):
    if kind=="All-GPU": return ["GPU"]*len(sids)
    if kind=="All-NPU": return ["NPU"]*len(sids)
    if kind=="Isolated":  # large-rich->NPU, small-rich->GPU
        return ["NPU" if SIZE_GROUP[s]=="large" else "GPU" for s in sids]
    if kind=="Cont-aware":  # small-rich->NPU
        return ["NPU" if SIZE_GROUP[s]=="small" else "GPU" for s in sids]

def step4_maincmp(val):
    out = RES / "rev19_main_comparison_threads4.csv"
    if out.exists(): out.unlink()
    rows=[]
    gpu_models=[FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models=load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu_models)
    for N, sids in [(2, COMP_A_2), (4, COMP_A_4)]:
        splits=[load_split_for_sid(val,s) for s in sids]
        for rep in range(N_REPS):
            torch.set_num_threads(THREADS)
            # enumerate all placements (for Oracle) + tag named ones
            best_worst=-1; best_pl=None; best_mean=None
            named_results={}
            for placement in all_placements(sids):
                ng=sum(1 for d in placement if d=="GPU"); nn=sum(1 for d in placement if d=="NPU")
                agg=measure_multistream(sids, splits, list(placement),
                                         gpu_models[:ng], npu_models[:nn], "L1_light")
                npu_skips=[s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="NPU"]
                worst=agg["worst_sap"]; mean=agg["mean_sap"]
                # track oracle
                if worst>best_worst:
                    best_worst=worst; best_pl=placement; best_mean=mean
                # match named
                for kind in ["All-GPU","Isolated","Cont-aware","All-NPU"]:
                    if list(placement)==named_strategy(sids,kind):
                        named_results[kind]=(worst,mean,max(npu_skips) if npu_skips else 0.0,placement)
            # write named + oracle
            for kind,(w,mn,sk,pl) in named_results.items():
                rows.append({"rep":rep,"N":N,"bg":"L1_light","strategy":kind,
                              "placement":"".join("N" if d=="NPU" else "G" for d in pl),
                              "worst_sap":round(w,4),"mean_sap":round(mn,4),"npu_skip_max":round(sk,1)})
            rows.append({"rep":rep,"N":N,"bg":"L1_light","strategy":"Oracle",
                          "placement":"".join("N" if d=="NPU" else "G" for d in best_pl),
                          "worst_sap":round(best_worst,4),"mean_sap":round(best_mean,4),"npu_skip_max":0.0})
            print(f"  MC N={N} rep{rep}: "
                    +" ".join(f"{k}={named_results[k][0]:.4f}" for k in named_results)
                    +f" Oracle={best_worst:.4f}({''.join('N' if d=='NPU' else 'G' for d in best_pl)})")
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=MC_COLS); w.writeheader()
        for r in rows: w.writerow(r)
    dispose_npu_for("yolo11s")
    for m in gpu_models: del m
    print(f"saved {out}")


def main():
    torch.set_num_threads(THREADS)
    print(f"=== rev19 threads={THREADS} (period={PERIOD:.1f}ms) ===")
    val=load_val(); print(f"{len(val['sequences'])} logs")
    print("\n--- STEP 2: Table 1 threads=4 ---")
    step2_table1(val)
    print("\n--- STEP 4: main-comparison N={2,4} threads=4 ---")
    step4_maincmp(val)
    print("\n=== rev19 measurement done ===")


if __name__ == "__main__":
    main()
