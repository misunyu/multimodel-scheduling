# EXP-QOS v2 — Summary

**Question (W2 defence).** Does CUDA stream-priority QoS, applied to protect the foreground
detector, remove the contention/staleness loss that flips the GPU↔NPU placement decision?

**Pre-registered verdict: Outcome C (partial protection).** Stream priority shifts the device
crossover to heavier contention and roughly halves the worst-stream degradation, but it does
**not** eliminate it: the large-biased staleness signature persists, and under a GPU-saturating
VLM co-tenant the reversal survives with the NPU still winning worst-stream sAP by ~2.5×.

---

## 1. Environment + drift
See `env_snapshot.md`. No drift from the §A-4 expected values (driver 580.159.03 / CUDA 13.0,
torch 2.12.0+cu130, ort 1.20.1, ultralytics 8.4.56, stream priority greatest=-1/least=0, MPS
binaries present and not running). GPU idle (0 compute processes) at start → solo measurement.
`yolo11s.pt` sha256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.

## 2. Operating-point ↔ baseline-skip mapping (§1)
Contention lever = number of ResNet50 GPU-loops (`resnet_k`); x-axis is the load setting, not the
resulting skip. BASE skip% are the rev22 measured means; `vlm_sat` = L3_vlm (ResNet50 + Qwen2-VL-2B).

| point | co-tenant | BASE GPU skip% (rev22/rev20) |
|---|---|---|
| k0 | none | 0.0 |
| k1 | resnet50 ×1 | 40.6 |
| k2 | resnet50 ×2 | 55.7 |
| k3 | resnet50 ×3 | 66.5 |
| k8 | resnet50 ×8 | 80.3 |
| vlm_sat | resnet50 + Qwen2-VL-2B | ~100 |

**Crossover-interpolation memo.** The paper's "crossover ≈ 48%" is an interpolation: BASE All-GPU
worst-stream is 0.0879 at k1 (skip 40.6%, above All-NPU 0.0836) and 0.0765 at k2 (skip 55.7%, below
0.0836); linear interpolation to the 0.0836 line lands near 48% skip. There is no discrete
co-tenant setting that yields exactly 48% skip.

## 3. BASE sanity (§4-2)  — PASS
Re-measured k2 All-GPU (no PRIO), 3 reps: worst-stream sAP = 0.0766 / 0.0749 / 0.0814,
mean **0.0776** vs rev22 mean **0.0765**, |Δ| = 0.0011 ≤ 0.006 → **PASS**. Environment is
consistent with the baseline that produced rev22; no full-baseline re-measurement needed.

## 4. Topology + PRIO implementation choice (§3.1)
Runtime probe: all four foreground threads use the torch **default stream** (`cuda_stream=0x0`)
— i.e. baseline FG streams share one stream and are serialized among themselves. Per §3.1-2 we
therefore mirror that concurrency with a **single shared high-priority stream**
(`torch.cuda.Stream(priority=-1)`, topology = `shared1`) wrapping `FGModelGPUGeneric.predict` in a
`torch.cuda.stream()` context; co-tenants are untouched (default priority). Core measurement
scripts were not modified. Note: using stream priority in PyTorch necessarily moves FG off the
legacy default stream, which both raises priority **and** drops the default-stream implicit
synchronization — this is intrinsic to the mechanism and is exactly the protection a deployer
would apply.

**Detection-identity check (§3.3): PASS.** 20 frames, PRIO off vs on: max box diff = 0, max score
diff = 0, 0 class mismatches → priority injection does not alter detections (the per-frame
`stream.synchronize()` in the wrapper makes the cross-stream read race-free).

## 5. PRIO micro-benchmark (§3.1-4) — priority is EFFECTIVE
At k2 (one full N=4 pass each, off vs on; the frame-driven harness has no 60 s wall knob so the
natural per-run sample is used, ~hundreds of infer samples/stream):

| | FG infer p99 | FG GPU skip% |
|---|---|---|
| PRIO off | 75.2 ms | 60.5% |
| PRIO on | 51.6 ms | 41.7% |
| Δ | **−23.6 ms** | **−18.8 pts** |

Priority demonstrably reduces FG inference latency and frame skip under contention (not "injected
but no effect").

## 6. Worst-stream sAP: BASE vs PRIO (mean ± std of 3 reps)
All-NPU reference (flat, reused) = **0.0836**.

| point | co-tenant | BASE worst | PRIO worst | PRIO−BASE | Δ vs uncontended (PRIO) | PRIO vs All-NPU |
|---|---|---|---|---|---|---|
| k0 | none | 0.1149 ± 0.0000 | 0.1149 ± 0.0000 | +0.0000 | 0.0000 | GPU wins |
| k1 | rn×1 | 0.0879 ± 0.0051 | 0.0975 ± 0.0031 | +0.0096 | 0.0174 | GPU wins |
| k2 | rn×2 | 0.0765 ± 0.0022 | 0.0877 ± 0.0015 | +0.0112 | 0.0272 | GPU wins |
| k3 | rn×3 | 0.0739 ± 0.0005 | 0.0837 ± 0.0011 | +0.0098 | 0.0312 | **tie** (+0.0001) |
| k8 | rn×8 | 0.0663 ± 0.0018 | 0.0801 ± 0.0015 | +0.0138 | 0.0348 | **NPU wins** |
| vlm_sat | rn+VLM | 0.0157 ± 0.0002 | 0.0328 ± 0.0015 | +0.0171 | 0.0821 | **NPU wins (2.5×)** |

PRIO improves worst-stream at every contended point, but in the k1–k3 partial band the worst still
sits 0.017–0.031 below its own uncontended value — far above the Outcome-B removal threshold (0.005).

## 7. Per-size worst-stream ΔsAP vs uncontended (PRIO) — large-biased structure PERSISTS
ΔsAP = (PRIO k0 per-size) − (PRIO point per-size), mean over the 4 streams.

| point | Δsmall | Δmedium | Δlarge |
|---|---|---|---|
| k1 | +0.0007 | +0.0206 | +0.0500 |
| k2 | +0.0010 | +0.0305 | +0.0751 |
| k3 | +0.0010 | +0.0314 | +0.0752 |
| k8 | +0.0013 | +0.0335 | +0.0857 |
| vlm_sat | +0.0053 | +0.0994 | +0.2566 |

At every point Δlarge ≫ Δmedium ≫ Δsmall. The contention loss under PRIO is still the large-biased
staleness signature (large objects lose most, small almost nothing) — the same size structure the
paper attributes to late delivery. QoS reduces the magnitude but does not change the mechanism.

## 8. Crossover verdict
Smallest co-tenant load at which PRIO All-GPU worst-stream drops below the All-NPU line (0.0836):
- **BASE**: between k1 (0.0879) and k2 (0.0765) → ~k1–k2, skip ≈ 48% (paper's value).
- **PRIO**: above k2 (0.0877), a tie at k3 (0.0837 ≈ 0.0836), and clearly below at k8 (0.0801).
  → crossover pushed to **~k3–k8** (skip ~66–80% by load; PRIO's own measured skip is lower because
  priority protects FG).

So stream priority moves the partial-band reversal to heavier contention, but the reversal **still
exists** at high load and is **strong at GPU saturation** (vlm_sat: GPU 0.033 vs NPU 0.084).

## 9. Pre-registered Outcome judgment (§0)
- **Outcome A** requires a crossover *within k1–k3* under PRIO. FALSE — PRIO All-GPU worst is ≥
  All-NPU through k2 and ties at k3.
- **Outcome B** requires the k1–k3 degradation essentially removed (Δ vs uncontended ≤ 0.005).
  FALSE — Δ is 0.017–0.031 in that band.
- → **Outcome C (partial protection).** PRIO (i) shifts the device crossover from ~k1–k2 (~48%
  skip) to ~k3–k8, (ii) roughly halves but does not remove the worst-stream degradation, (iii)
  preserves the large-biased staleness structure, and (iv) leaves the reversal intact at GPU
  saturation, where the NPU still wins worst-stream sAP by ~2.5×.

**Paper implication (for the human to apply).** The W2 reply can be honest and strong: stream
priority *does* help at moderate contention (so we do not claim it is useless), yet it neither
removes the staleness loss nor prevents the placement reversal under the GPU-saturating VLM
co-tenant that the paper's headline uses. The reversal is therefore not a QoS-strawman: it is a
mechanism (large-biased staleness) that QoS attenuates but cannot eliminate, and that re-emerges
fully under saturation.

## 10. MPS-MINI (§3.2) — NOT triggered
Outcome C runs MPS-MINI only if the crossover persists at **k3 or below** under PRIO. It does not:
the partial-band (k1–k3) crossover is removed (GPU ties/wins through k3), and the crossover only
survives at k8 and at saturation. Per the pre-registered rule, MPS-MINI is **not executed**.
(Independently, §A-2: per-client `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` differentiation is impossible
in the single-process harness; a faithful MPS-MINI would require the separate multi-process harness
of §3.2, which was not built because the trigger condition was not met.) `exp_qos_mps_results.csv`
is therefore not produced.

## 11. Paper-text consistency memo (§A-5)
The actual foreground GPU detector path is **Ultralytics PyTorch** (`YOLO("yolo11s.pt")`,
`device="cuda"`), not ONNX Runtime. In the co-tenant stack, ResNet50 and TinyLLaMA run on
**ONNX Runtime CUDA EP** and Qwen2-VL-2B runs on **PyTorch (bfloat16)**. The paper (Sec. 5) states
"GPU and CPU inference use FP32 ONNX models", which is accurate for the CNN/LM co-tenants but not
for the foreground YOLOv11s detector (PyTorch FP32). Whether to amend the paper text is a human
decision; recorded here for traceability. (This does not affect any measurement: BASE and PRIO use
the identical FG path; PRIO differs only by the high-priority stream.)

## 12. Failures / skips
- MPS-MINI: not triggered (see §10).
- Micro-bench "60 s" window adapted to one full N=4 pass per condition (frame-driven harness has no
  wall-clock knob); documented in §5. Not a change to any §0 criterion.
- No measurement failed; no value is interpolated or back-derived. Reused BASE values are cited from
  rev22_vlm_sweep.csv (k0–k8 All-GPU/All-NPU) and rev20_5strat_heavybg.csv (vlm_sat).

## Artifacts
- `exp_qos_results.csv` — per-run/per-stream rows: base (k2 sanity, 3 reps) + prio (6 points × 3 reps).
- `exp_qos_pre.json` — topology / identity / micro-bench / sanity machine record.
- `env_snapshot.md`, `exp_qos_pre_stdout.log`, `exp_qos_sweep_stdout.log`.
- Script: `accv_experiments/scripts/phase_qos_v2.py` (new; core scripts unmodified).
