"""M2 — claim-validity check for the submitted L2LM / L3VLM co-tenancy results.

Usage:  python log/m2_run.py <baseline|L2LM|L3VLM>     # one process per condition

Purpose: NOT to replace the submitted numbers. This checks whether those numbers
and the paper's Section 4.3 mechanism can still be used as the basis of the
rebuttal. If values disagree, the discrepancy is reported as measured; no setting
is adjusted and no submitted value is overwritten.

=============================================================================
PRE-REGISTERED — fixed before any measurement; not revised after seeing results.
=============================================================================

UNITS
  Raw sAP on [0,1] throughout. x100 for display only; the two are never mixed.

CO-TENANT CONFIGURATION (submitted setup, unchanged)
  baseline = no co-tenant                       (bg level "L0")
  L2LM     = 1x ResNet50 + 1x TinyLLaMA-1.1B    (bg level "L2_lm")
  L3VLM    = 1x ResNet50 + 1x Qwen2-VL-2B       (bg level "L3_vlm")
  TinyLLaMA is NOT added to L3VLM. All loops keep the existing back-to-back,
  pre-generated-input form (step_h2_robustness.py:93-99, _step_d_common.py:342-358).
  As in the submitted run (phase_rev20_heavybg.py:89), all three background models
  are preloaded (max_level="L3") regardless of the condition being measured, so
  resident GPU memory matches.

FOUR-LOG WORKLOAD (verified, not chosen)
  sids [2, 22, 3, 21] — PANELS[4] in phase_rev20_heavybg.py:33, the producer of
  the submitted rev20_5strat_heavybg.csv L2LM/L3VLM rows. Identical to M1's panel.

MEASUREMENT FUNCTION
  phase_rev6_sweep.measure_multistream — the exact function the submitted L2LM /
  L3VLM runs called (phase_rev20_heavybg.py:112), imported unchanged. Not
  reimplemented, and not swapped for rev30's variant.

RESIDUAL DIFFERENCE FROM THE SUBMITTED RUN (recorded, not corrected)
  A submitted rev20 N=4 block enumerated all 16 placements per rep
  (phase_rev20_heavybg.py:101-102, itertools.product over 2^4), with All-GPU
  first and All-NPU last. An M2 block contains only the two compared placements,
  as this protocol specifies. The GPU/thermal history entering each cell
  therefore differs.

EXECUTION ORDER (n=3 cannot be balanced; approximately counterbalanced)
  baseline : GPU-first 2 / NPU-first 1
  L2LM     : GPU-first 2 / NPU-first 1
  L3VLM    : GPU-first 1 / NPU-first 2
  The baseline split is not specified by the protocol; it is fixed here, before
  measurement, and recorded. No statistical claim of balance is made, and the
  conditions are not pooled to claim balance. Order is generated from a recorded
  seed before the run.

WARM-UP (technical exclusion, pre-declared)
  L2LM and L3VLM are different co-tenant configurations, so each condition —
  baseline included, since it runs as its own process — begins with one warm-up
  paired block under its own configuration. It is excluded from analysis and
  written to the manifest as "warmup, pre-declared". The exclusion is declared
  here, before measurement, not decided from performance values.
  Basis: recheck_prefix showed cold start biases the first measurement
  (GGGG DM 29.73 -> 24.19, paired d +0.00782 -> +0.01478).

FAILED RUNS
  Every attempted run — crash, timeout, invalid — is written to the manifest.
  Pre-defined technical-failure exclusions, with reason recorded:
    (a) the measurement call raised an exception (status FAILED);
    (b) the warm-up block (status "warmup, pre-declared").
  Nothing is excluded on the basis of measured performance.

CPU UTILIZATION
  phase_cpuload.Sampler is imported and used unchanged — the same logger, in the
  same place (a separate daemon thread started immediately before the
  measure_multistream call and stopped immediately after), that produced the
  submitted 17->93% figures (phase_cpuload.py:21-34, 47-56). No new synchronous
  psutil call is inserted into the foreground inference loop: the 0.1 s blocking
  sample would itself change the workload this experiment exists to check.
  It records system-wide cpu_percent(interval=0.1, percpu=True) averaged over
  cores, the process-specific Process.cpu_percent, max-core, and pynvml GPU util.
  CPU util is recorded per placement. The submitted 17->93% was measured on
  All-NPU placement only (phase_cpuload.py:17-18: CONDS baseline/+LM/+VLM, all
  measured with ["NPU"]*4), so an L2LM All-NPU value is compared against the
  no-co-tenant All-NPU value, never against All-GPU.

ANALYSIS RULE (overrides the standing "difference first, then CI" instruction
for M2 only)
  GPU-NPU differences are summarised by computing the per-block paired difference
  first and reporting both its mean and all three individual values. At n=3, no
  p-value, no equivalence test, and no inferential CI is produced.

DECISION RULE (a) — L2LM high-staleness regime check, primary
  3-block mean GPU DM >= 50% AND 3-block mean NPU DM >= 50%
    -> the submitted high-NPU-DM / both-paths-stale regime is strongly reproduced
  mean NPU DM < 50%
    -> qualitative reproduction of that regime is not established. The mechanism
       itself is NOT rejected here. Instead these four are reported as measured:
         - change in NPU DM vs baseline
         - change in All-NPU worst sAP vs baseline
         - GPU-NPU ranking (which placement is better)
         - CPU utilization vs the placement-matched baseline
  50% is not a physically special threshold; it is a pre-set criterion for
  identifying the submitted high-staleness regime.
  GPU-NPU DM difference and CPU utilization are reported alongside but are not
  separate pass/fail criteria. CPU util is a corroborating diagnostic for the
  host-saturation reading and does not by itself overturn the verdict above.

DECISION RULE (b) — quantitative consistency, descriptive only
  Each submitted metric is placed beside the new 3-block mean with absolute and
  relative difference. No pass/fail. This is not called an equivalence test or a
  reproduction test; n=3 cannot support such a claim.
"""
from __future__ import annotations
import csv, hashlib, json, os, platform, random, subprocess, sys, time
from pathlib import Path

COND_BG = {"baseline": "L0", "L2LM": "L2_lm", "L3VLM": "L3_vlm"}
COND_ORDER_SPEC = {                       # (n GPU-first, n NPU-first), pre-fixed
    "baseline": (2, 1),                   # not specified by protocol; fixed here
    "L2LM": (2, 1),
    "L3VLM": (1, 2),
}
if len(sys.argv) != 2 or sys.argv[1] not in COND_BG:
    sys.exit(f"usage: python log/m2_run.py <{'|'.join(COND_BG)}>")
COND = sys.argv[1]

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "accv_experiments/scripts"))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))

RAW = OUT / f"m2_{COND}_raw.csv"
META = OUT / f"m2_{COND}_meta.json"
MANIFEST = OUT / f"m2_{COND}_manifest.csv"

N_BLOCKS = 3
BASE_SEED = 20260902
SEED = BASE_SEED + {"baseline": 10, "L2LM": 20, "L3VLM": 30}[COND]


def build_order(seed, n_gpu_first, n_npu_first):
    o = ["GPU_FIRST"] * n_gpu_first + ["NPU_FIRST"] * n_npu_first
    random.Random(seed).shuffle(o)
    return o


ORDER = build_order(SEED, *COND_ORDER_SPEC[COND])
assert len(ORDER) == N_BLOCKS


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<{type(e).__name__}>"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


import torch  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

meta = {
    "experiment": "M2", "condition": COND, "bg_level": COND_BG[COND],
    "purpose": "claim-validity check for submitted L2LM/L3VLM; not a replacement",
    "units": "raw sAP [0,1]; x100 display only",
    "n_blocks": N_BLOCKS,
    "t_wall_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "boot_time": sh("uptime -s"),
    "git_commit": sh("git rev-parse HEAD"),
    "git_dirty_measurement_path": bool(sh("git status --porcelain -- accv_experiments")),
    "python": platform.python_version(), "interpreter": sys.executable,
    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    "onnxruntime_gpu": ort.__version__, "numpy": np.__version__,
    "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "shim_sourced": "vendor_ort_cu12" in os.environ.get("LD_LIBRARY_PATH", ""),
    "base_seed": BASE_SEED, "seed": SEED,
    "order_spec_gpu_first_npu_first": COND_ORDER_SPEC[COND],
    "generated_order": ORDER,
    "order_note": ("n=3 cannot be balanced; approximately counterbalanced per protocol. "
                   "baseline split not specified by protocol, fixed here before measurement. "
                   "No statistical claim of balance across conditions."),
    "warmup_policy": "one excluded warm-up paired block per condition process; pre-declared",
    "measure_fn": "phase_rev6_sweep.measure_multistream (the submitted L2LM/L3VLM path)",
    "cpu_logger": ("phase_cpuload.Sampler, imported unchanged — the logger that produced "
                   "the submitted 17->93%; separate daemon thread around the measure call"),
    "submitted_cpu_util_placement": "All-NPU (phase_cpuload.py CONDS use ['NPU']*4)",
    "residual_difference_vs_submitted": ("rev20 N=4 block enumerated all 16 placements "
                                         "(phase_rev20_heavybg.py:101-102); M2 block has 2"),
}
import importlib  # noqa: E402
for name in ("qbruntime", "mblt_model_zoo", "ultralytics", "psutil", "transformers"):
    try:
        meta[name] = str(getattr(importlib.import_module(name), "__version__", "?"))
    except Exception as e:
        meta[name] = f"<{type(e).__name__}>"

# ---- documented init order: ORT co-tenant sessions BEFORE any torch CUDA model
from _step_d_common import preload_background_models, _BG_PRELOADED   # noqa: E402
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, measure_multistream,  # noqa: E402
                              set_active_npu_engines, dispose_npu_for, DETECTORS)
from _step_d_common import load_val, load_split_for_sid                # noqa: E402
from step0_compare_devices import FPS                                  # noqa: E402
import phase_cpuload as CPUL                                           # noqa: E402

DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
THREADS = 4
PERIOD = 1000.0 / FPS
PANEL4 = [2, 22, 3, 21]          # phase_rev20_heavybg.py:33 PANELS[4]

torch.set_num_threads(THREADS)
print(f"[M2 {COND}] bg={COND_BG[COND]} seed={SEED} order={ORDER}", flush=True)

t0 = time.time()
preload_background_models(max_level="L3")     # as the submitted run did
meta["ort_providers"] = _BG_PRELOADED["resnet"][0].get_providers()
meta["resnet_cotenant_on_gpu"] = "CUDAExecutionProvider" in meta["ort_providers"]
print(f"[gate] ORT providers = {meta['ort_providers']}", flush=True)

val = load_val()
gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(8)]
npu_models = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 8)
set_active_npu_engines(npu_models)
meta.update({
    "mxq_path": DET["multistream_mxq"],
    "mxq_sha256": sha256(ROOT / DET["multistream_mxq"]),
    "npu_infer_mode": DET["multistream_mode"],
    "panel4_sids": PANEL4, "threads": THREADS, "period_ms": round(PERIOD, 3),
    "detector": "yolo11s",
})
print(f"[load] ready ({time.time()-t0:.1f}s) mxq={meta['mxq_sha256'][:16]}", flush=True)
META.write_text(json.dumps(meta, indent=2))

splits = [load_split_for_sid(val, s) for s in PANEL4]

RAW_COLS = ["run_id", "condition", "bg_level", "block", "block_kind", "order", "cell_index",
            "placement", "gpu_dm_pct", "npu_dm_pct", "worst_sap", "mean_sap",
            "cpu_sys_mean", "cpu_sys_p99", "cpu_proc_mean", "cpu_proc_p99",
            "cpu_maxcore_p99", "gpu_util_mean", "n_util_samples",
            "per_stream_json", "t_start", "t_end", "wall_s", "period_ms", "fps",
            "detector", "dataset", "timestamp"]
MAN_COLS = ["run_id", "condition", "block", "block_kind", "order", "cell_index", "placement",
            "status", "detail", "t_start", "t_end"]


def app(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def dev_dm(agg):
    g = [s["frame_skip_pct"] for s in agg["per_stream"] if s["device"] == "GPU"]
    n = [s["frame_skip_pct"] for s in agg["per_stream"] if s["device"] == "NPU"]
    return (round(float(np.mean(g)), 2) if g else 0.0,
            round(float(np.mean(n)), 2) if n else 0.0)


run_id = f"m2_{COND}_{int(time.time())}"
bg = COND_BG[COND]
GGGG = ["GPU"] * 4
NNNN = ["NPU"] * 4


def run_cell(block, block_kind, order, cell_index, placement):
    ps = "".join("N" if d == "NPU" else "G" for d in placement)
    ng = sum(1 for d in placement if d == "GPU")
    nn = sum(1 for d in placement if d == "NPU")
    torch.set_num_threads(THREADS)
    man = {"run_id": run_id, "condition": COND, "block": block, "block_kind": block_kind,
           "order": order, "cell_index": cell_index, "placement": ps}
    smp = CPUL.Sampler()                      # the submitted logger, unchanged
    smp.start()
    ta = time.time()
    man["t_start"] = round(ta, 3)
    try:
        agg = measure_multistream(PANEL4, splits, placement, gpu_models[:ng], npu_models[:nn], bg)
        tb = time.time()
    except Exception as e:
        tb = time.time()
        smp.stop.set(); smp.join(timeout=5)
        man.update({"status": "FAILED", "detail": f"{type(e).__name__}: {e}"[:300],
                    "t_end": round(tb, 3)})
        app(MANIFEST, MAN_COLS, man)
        print(f"  !! {COND}/b{block}/{ps} FAILED: {type(e).__name__}: {e}", flush=True)
        return
    smp.stop.set(); smp.join(timeout=5)
    sysm, _, sysp99 = CPUL.stats(smp.sys)
    pm, _, pp99 = CPUL.stats(smp.proc)
    _, _, mcore = CPUL.stats(smp.maxcore)
    gm, _, _ = CPUL.stats(smp.gpu)
    gdm, ndm = dev_dm(agg)
    row = {"run_id": run_id, "condition": COND, "bg_level": bg, "block": block,
           "block_kind": block_kind, "order": order, "cell_index": cell_index, "placement": ps,
           "gpu_dm_pct": gdm, "npu_dm_pct": ndm,
           "worst_sap": round(agg["worst_sap"], 5), "mean_sap": round(agg["mean_sap"], 5),
           "cpu_sys_mean": sysm, "cpu_sys_p99": sysp99, "cpu_proc_mean": pm,
           "cpu_proc_p99": pp99, "cpu_maxcore_p99": mcore, "gpu_util_mean": gm,
           "n_util_samples": len(smp.sys),
           "per_stream_json": json.dumps(agg["per_stream"]),
           "t_start": round(ta, 3), "t_end": round(tb, 3), "wall_s": round(tb - ta, 2),
           "period_ms": round(PERIOD, 3), "fps": FPS,
           "detector": "yolo11s", "dataset": "argoverse-hd-val", "timestamp": int(tb)}
    app(RAW, RAW_COLS, row)
    man.update({"status": "warmup, pre-declared" if block_kind == "warmup" else "OK",
                "detail": "excluded from analysis (pre-declared technical warm-up)"
                          if block_kind == "warmup" else "",
                "t_end": round(tb, 3)})
    app(MANIFEST, MAN_COLS, man)
    tag = "WARMUP " if block_kind == "warmup" else ""
    print(f"  {tag}{COND}/b{block}/{ps}: gpuDM={gdm} npuDM={ndm} worst={row['worst_sap']} "
          f"cpu_sys={sysm} gpu_util={gm} ({tb-ta:.1f}s)", flush=True)


try:
    run_cell(-1, "warmup", "GPU_FIRST", 0, GGGG)
    run_cell(-1, "warmup", "GPU_FIRST", 1, NNNN)
    for b, order in enumerate(ORDER):
        seq = (GGGG, NNNN) if order == "GPU_FIRST" else (NNNN, GGGG)
        for ci, placement in enumerate(seq):
            run_cell(b, "measure", order, ci, placement)
finally:
    try:
        dispose_npu_for("yolo11s")
    except Exception as e:
        print(f"  (dispose warning: {e})", flush=True)

meta["t_wall_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
META.write_text(json.dumps(meta, indent=2))
print(f"=== M2 {COND} done ===", flush=True)
