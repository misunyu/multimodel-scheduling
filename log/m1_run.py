"""M1 — three representative contention points at n=10, one environment.

Reviewer LZun asked for repetition counts and confidence intervals. This measures
k = 1, 2, 3 (ResNet50 GPU co-tenant instance count) with n=10 blocks each, in the
current environment, and reports raw per-stream data plus block-wise paired
differences.

Usage:  python log/m1_run.py <K>            # one process per k (required, see warm-up)

=============================================================================
PRE-REGISTERED PROTOCOL — fixed before any measurement; do not revise after
seeing results.
=============================================================================

UNITS
  All statistics are computed in raw sAP on [0, 1].
  0.005 raw sAP = 0.5 sAP point in paper notation. Multiply by 100 for display
  only. The two units are never mixed.

BLOCK
  block = the same four Argoverse-HD logs (PANEL4 = sid 2, 22, 3, 21) under the
  same co-tenant configuration (k ResNet50 ORT-CUDA loops). Within one block the
  two placements All-GPU (GGGG) and All-NPU (NNNN) are measured consecutively.
  d_i = worst_sAP(All-GPU, i) - worst_sAP(All-NPU, i), formed INSIDE block i.
  CIs are computed from the sample of d_i, never by subtracting two aggregate
  means.

  Note a residual difference from rev30 that this design does not remove: a
  rev30 block contained five placements (GGGG, GGNG, GGNN, GNNN, NNNN) measured
  in a fixed order, whereas an M1 block contains only the two compared
  placements. This is recorded, not corrected.

EXECUTION ORDER
  Balanced randomized: of the 10 blocks, 5 run All-GPU first and 5 run All-NPU
  first. The order is generated BEFORE execution from a recorded seed and stored
  in the metadata. It is not alternating and it is not rev30's fixed
  All-GPU-first order.

WARM-UP (technical exclusion, pre-declared)
  Each measurement process begins with one warm-up block at the same k and the
  same co-tenant configuration. Its result is excluded from all analysis. The
  exclusion is declared here, before measurement, and every warm-up cell is
  written to the manifest with status "warmup, pre-declared". It is not decided
  by looking at performance values.
  Basis: the recheck_prefix experiment showed cold start biases the first
  measurement (GGGG DM 29.73 -> 24.19, paired d +0.00782 -> +0.01478).

FAILED RUNS
  Every attempted run — including crash, timeout, and invalid — is written to the
  manifest. A run is excluded only under the pre-defined technical-failure
  criteria below, with the reason recorded:
    (a) the measurement call raised an exception (status FAILED);
    (b) the warm-up block (status "warmup, pre-declared").
  No run is ever excluded on the basis of its measured performance values.

DECISION RULE — primary, k=1 only
  TOST for equivalence of the block-wise paired difference against rev30:
    Delta_d = mean(d_new) - mean(d_rev30)
    H01: Delta_d <= -0.005      H02: Delta_d >= +0.005
    alpha = 0.05, two independent samples, Welch degrees of freedom.
    Both one-sided p < 0.05  -> equivalence established within +/-0.005
    otherwise                -> equivalence not established
  "Equivalence not established" must NOT be read as "the two measurements
  differ". In that case the new values are reported without claiming agreement.
  A Welch two-sample difference test is deliberately NOT used: failure to reject
  is not evidence of equivalence, so it cannot support the needed conclusion.

  Equivalence margin +/-0.005 — sole basis: rev30's k=1 d = +0.01497 (about
  +1.5 sAP points), so a change within +/-0.5 point does not reverse the
  qualitative placement ordering at this operating point. Power is not a basis
  and is not invoked.

  The margin is NOT applied to k=2 or k=3: both are near the crossover where d
  approaches 0, so +/-0.5 point could flip the sign.

  TOST is performed at k=1 only, because k=1 is the only point where rev30 has
  n=10 (k=2 has 7 reps, k=3 has 7). The other two points report new values only.

SECONDARY (descriptive, no tests)
  GPU deadline miss, NPU deadline miss, and per-cell worst sAP are tabulated
  beside the paper values.

BRACKET FAILURE (pre-registered)
  If k = 1, 2, 3 fail to bracket the crossover in the new environment, the points
  are neither replaced nor extended; the result is reported as measured. Adding
  k=4 after seeing the result would be post-hoc adaptation.
  Reporting language:
    lower-DM point's d CI entirely > 0 and higher-DM point's entirely < 0
      -> "sign-resolved interval between the tested points"
    a middle point's CI contains 0
      -> "preference unresolved at X% DM; NPU-favoring at Y%"
  The phrase "confidence interval of the crossover" is not used: what is computed
  is the CI of the difference at each tested point, not a CI of the crossover
  point itself.

CONDITIONS HELD IDENTICAL TO rev30
  30-frame warm-up inside per_stream_sap, threads=4, conf 0.25 / IoU 0.45 /
  max_det 300, infer_mode "single", PANEL4 sids, and the same frame-range /
  replay-termination logic — phase_rev30_clean.measure_multistream_full is
  imported and called unchanged, never reimplemented. No fixed-duration timer is
  introduced: the "14 s window" is a property of the chosen sids' frame ranges,
  not a code constant, and is not implemented as a timer.

LOGGING
  Per-stream raw sAP (worst-stream is the min over 4 streams, so aggregates alone
  are insufficient), GPU and NPU deadline-miss rates, and GPU utilization /
  memory / power from the sampler rev30 already used.
  Host CPU utilization is deliberately NOT logged here: rev30's sweep had no CPU
  util, and adding psutil.cpu_percent(0.1, ...) would introduce a 0.1 s blocking
  call and extra CPU activity — a new difference. M1's whole point is condition
  identity with rev30.
"""
from __future__ import annotations
import csv, hashlib, json, os, platform, random, signal, subprocess, sys, time
from pathlib import Path

if len(sys.argv) != 2 or sys.argv[1] not in ("1", "2", "3"):
    sys.exit("usage: python log/m1_run.py <K>   where K is 1, 2 or 3")
K = int(sys.argv[1])

OUT = Path(__file__).resolve().parent            # log/
ROOT = OUT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "accv_experiments/scripts"))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))

RAW = OUT / f"m1_k{K}_raw.csv"
UTIL = OUT / f"m1_k{K}_util.csv"
META = OUT / f"m1_k{K}_meta.json"
MANIFEST = OUT / f"m1_k{K}_manifest.csv"

N_BLOCKS = 10
BASE_SEED = 20260902                              # fixed before the run; recorded in meta
SEED = BASE_SEED + K                              # distinct order per k, both recorded
SAMPLER_CORE = "23"
SAMPLE_INTERVAL = 0.2


# ---- pre-generated balanced randomized order (before any measurement) -------
def build_order(seed: int, n: int) -> list[str]:
    """n/2 blocks start with All-GPU, n/2 with All-NPU; shuffled, not alternating."""
    assert n % 2 == 0
    order = ["GPU_FIRST"] * (n // 2) + ["NPU_FIRST"] * (n // 2)
    random.Random(seed).shuffle(order)
    return order


ORDER = build_order(SEED, N_BLOCKS)
assert ORDER.count("GPU_FIRST") == 5 and ORDER.count("NPU_FIRST") == 5
IS_ALTERNATING = all(ORDER[i] != ORDER[i + 1] for i in range(len(ORDER) - 1))


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
    "experiment": "M1",
    "k": K,
    "n_blocks": N_BLOCKS,
    "purpose": "three representative contention points at n=10 in one environment",
    "units": "all statistics in raw sAP [0,1]; x100 for display only",
    "t_wall_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "boot_time": sh("uptime -s"),
    "git_commit": sh("git rev-parse HEAD"),
    "git_dirty_measurement_path": bool(sh("git status --porcelain -- accv_experiments")),
    "python": platform.python_version(),
    "interpreter": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "onnxruntime_gpu": ort.__version__,
    "numpy": np.__version__,
    "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "shim_sourced": "vendor_ort_cu12" in os.environ.get("LD_LIBRARY_PATH", ""),
    "base_seed": BASE_SEED,
    "seed": SEED,
    "generated_order": ORDER,
    "order_is_alternating": IS_ALTERNATING,
    "order_balance": {"GPU_FIRST": ORDER.count("GPU_FIRST"), "NPU_FIRST": ORDER.count("NPU_FIRST")},
    "warmup_policy": "one excluded warm-up block per process at the same k; pre-declared",
    "cpu_util_logged": False,
    "cpu_util_reason": "rev30 sweep had none; adding psutil sampling would change conditions",
    "tost": {
        "applies_to": "k=1 only (rev30 has n=10 only at k=1)",
        "margin_raw_sap": 0.005,
        "alpha": 0.05,
        "df": "Welch",
        "H01": "Delta_d <= -0.005",
        "H02": "Delta_d >= +0.005",
        "margin_basis": "rev30 k=1 d=+0.01497; +/-0.5 sAP point does not reverse ordering here",
        "not_applied_to": "k=2, k=3 (near crossover, d near 0)",
    },
}
for name, mod in (("qbruntime", "qbruntime"), ("mblt_model_zoo", "mblt_model_zoo"),
                  ("ultralytics", "ultralytics")):
    try:
        import importlib
        meta[name] = str(getattr(importlib.import_module(mod), "__version__", "?"))
    except Exception as e:
        meta[name] = f"<{type(e).__name__}>"

# ---- documented init order: ORT co-tenant sessions BEFORE any torch CUDA model
import phase_rev30_clean as R30                       # noqa: E402
from _step_d_common import _BG_PRELOADED              # noqa: E402
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,    # noqa: E402
                              set_active_npu_engines, dispose_npu_for)

torch.set_num_threads(R30.THREADS)
print(f"[M1 k={K}] seed={SEED} order={ORDER}", flush=True)
print(f"[M1 k={K}] torch={meta['torch']} ort={meta['onnxruntime_gpu']} "
      f"driver={meta['driver']} shim={meta['shim_sourced']}", flush=True)

t0 = time.time()
R30.preload_background_models(max_level="L1")
sess = _BG_PRELOADED["resnet"][0]
meta["ort_providers"] = sess.get_providers()
meta["resnet_cotenant_on_gpu"] = "CUDAExecutionProvider" in sess.get_providers()
print(f"[gate] ORT providers = {meta['ort_providers']}", flush=True)

val = R30.load_val()
n_models = 8                                          # rev30's full run loaded 8 of each
gpu_models = [FGModelGPUGeneric(R30.DET["ultralytics_pt"]) for _ in range(n_models)]
npu_models = load_npu_engines(R30.DET, R30.DET["multistream_mxq"], R30.NPU_MODE, n_models)
set_active_npu_engines(npu_models)
meta.update({
    "mxq_path": R30.DET["multistream_mxq"],
    "mxq_sha256": sha256(ROOT / R30.DET["multistream_mxq"]),
    "npu_infer_mode": R30.NPU_MODE,
    "n_engines_loaded": n_models,
    "panel4_sids": R30.PANEL4,
    "threads": R30.THREADS,
    "period_ms": round(R30.PERIOD, 3),
    "detector": "yolo11s",
    "conf_iou_maxdet": "0.25 / 0.45 / 300 (unchanged from rev30 defaults)",
})
print(f"[load] {n_models} GPU + {n_models} NPU ready ({time.time()-t0:.1f}s) "
      f"mxq={meta['mxq_sha256'][:16]} mode={R30.NPU_MODE}", flush=True)
META.write_text(json.dumps(meta, indent=2))

sp4 = [R30.load_split_for_sid(val, s) for s in R30.PANEL4]

samp = subprocess.Popen(["taskset", "-c", SAMPLER_CORE, sys.executable,
                         str(ROOT / "accv_experiments/scripts/gpu_util_sampler.py"),
                         str(UTIL), str(SAMPLE_INTERVAL)])
time.sleep(2.0)

RAW_COLS = ["run_id", "k", "block", "block_kind", "order", "cell_index", "placement",
            "gpu_skip_pct", "npu_skip_pct", "worst_sap", "mean_sap", "median_sap", "p10_sap",
            "sap_small", "sap_medium", "sap_large", "deadline_margin_ms_mean",
            "per_stream_json", "t_start", "t_end", "period_ms", "fps",
            "detector", "dataset", "timestamp"]
MAN_COLS = ["run_id", "k", "block", "block_kind", "order", "cell_index", "placement",
            "status", "detail", "t_start", "t_end"]


def app(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


run_id = f"m1_k{K}_{int(time.time())}"
bg = R30.reg_level(K)
GGGG = ["GPU"] * 4
NNNN = ["NPU"] * 4


def run_cell(block, block_kind, order, cell_index, placement):
    ps = R30.pstr(placement)
    ng = sum(1 for d in placement if d == "GPU")
    nn = sum(1 for d in placement if d == "NPU")
    torch.set_num_threads(R30.THREADS)
    ta = time.time()
    man = {"run_id": run_id, "k": K, "block": block, "block_kind": block_kind,
           "order": order, "cell_index": cell_index, "placement": ps,
           "t_start": round(ta, 3)}
    try:
        agg = R30.measure_multistream_full(R30.PANEL4, sp4, placement,
                                           gpu_models[:ng], npu_models[:nn], bg)
        tb = time.time()
    except Exception as e:                              # technical failure (a)
        tb = time.time()
        man.update({"status": "FAILED", "detail": f"{type(e).__name__}: {e}"[:300],
                    "t_end": round(tb, 3)})
        app(MANIFEST, MAN_COLS, man)
        print(f"  !! b{block}/{ps} FAILED: {type(e).__name__}: {e}", flush=True)
        return
    s = R30.summarize(agg, placement)
    row = {"run_id": run_id, "k": K, "block": block, "block_kind": block_kind,
           "order": order, "cell_index": cell_index, "placement": ps,
           "t_start": round(ta, 3), "t_end": round(tb, 3),
           "per_stream_json": json.dumps(agg["per_stream"]),
           "period_ms": round(R30.PERIOD, 3), "fps": R30.FPS,
           "detector": "yolo11s", "dataset": R30.DATASET, "timestamp": int(tb)}
    row.update(s)
    app(RAW, RAW_COLS, row)                             # warm-up rows are written too,
    man.update({"status": "warmup, pre-declared" if block_kind == "warmup" else "OK",
                "detail": "excluded from analysis (pre-declared technical warm-up)"
                          if block_kind == "warmup" else "",
                "t_end": round(tb, 3)})
    app(MANIFEST, MAN_COLS, man)                        # and flagged block_kind=warmup
    tag = "WARMUP " if block_kind == "warmup" else ""
    print(f"  {tag}k{K}/b{block}/{ps}: gpu_skip={s['gpu_skip_pct']} "
          f"npu_skip={s['npu_skip_pct']} worst={s['worst_sap']} ({tb-ta:.1f}s)", flush=True)


try:
    # --- warm-up block: same k, same co-tenant configuration, excluded ---------
    run_cell(-1, "warmup", "GPU_FIRST", 0, GGGG)
    run_cell(-1, "warmup", "GPU_FIRST", 1, NNNN)

    # --- 10 measurement blocks in the pre-generated balanced order -------------
    for b, order in enumerate(ORDER):
        seq = (GGGG, NNNN) if order == "GPU_FIRST" else (NNNN, GGGG)
        for ci, placement in enumerate(seq):
            run_cell(b, "measure", order, ci, placement)
finally:
    samp.send_signal(signal.SIGTERM)
    try:
        samp.wait(timeout=5)
    except Exception:
        samp.kill()
    try:
        dispose_npu_for("yolo11s")
    except Exception as e:
        print(f"  (dispose warning: {e})", flush=True)

meta["t_wall_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
META.write_text(json.dumps(meta, indent=2))
print(f"=== M1 k={K} done ===", flush=True)
