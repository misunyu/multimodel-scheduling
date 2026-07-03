# rev 5 diagnosis — mxq drift / eval-path / relative loss

No new hardware measurement was taken. This file aggregates the verdicts of
D1 (provenance), D2 (eval-path audit), and D3 (relative loss recomputation).

## D1 — mxq provenance

| source | sha256 (first 16) | size | mtime | HF revision |
|---|---|---|---|---|
| `~/.cache/huggingface/hub/.../snapshots/8e62b1a9/aries/yolo11s.mxq` (= step_a / Table 1) | `b2441f9da7ec7dd6...` | 11.87 MB | 2026-05-27 18:30 | `8e62b1a9` |
| `models/mobilint_backup/yolo11s.mxq` (legacy backup) | `b2441f9da7ec7dd6...` | 11.87 MB | 2026-05-27 18:30 | — |
| `~/.mblt_model_zoo/vision/aries/global8/yolo11s.mxq` (B3 baseline / ladder) | `4ff08b089d4ced5c...` | 10.67 MB | 2026-06-02 10:26 | — |
| `~/.mblt_model_zoo/vision/aries/single/yolo11s.mxq` (B3 multistream) | `3ade8e507e51a771...` | 10.26 MB | 2026-06-02 11:05 | — |

**Verdict: mxq DRIFT confirmed.** Three distinct SHA256 hashes for `yolo11s.mxq`. The original anchor (`step_a` + Table 1) used hash `b2441f9d` (11.87 MB); B3 downloaded two *different* binaries via the new `mblt_model_zoo` clone (`4ff08b08` global8 = 10.67 MB; `3ade8e50` single = 10.26 MB). The HF repository at `mobilint/YOLO11s` has been republished with different mxq binaries since the Table 1 era — likely with revised INT8 calibration, different layer fusion, or a different compiler version. All three files load cleanly under the same SDK API, so the loader masks the underlying weight change.

Within B3 itself, the v11s baseline+ladder ran on hash `4ff08b08` (global8 binary) and the v11s multistream ran on hash `3ade8e50` (single binary). They are different binaries, not merely the same binary with different runtime modes.

Per-file CSV: `results/mxq_provenance.csv`.

## D2 — eval-path audit

Full table: `results/evalpath_audit.md`.

| parameter | step_a (Table 1) | step_k (gen_decomp old) | phase_b3 (gen_decomp new) | same? |
|---|---|---|---|---|
| COCO areaRng (32², 96²) | pycocotools default | pycocotools default | pycocotools default | yes |
| IoU range | $[.50:.95]$ via `e.stats[3..5]` | same | same | yes |
| AP_small/medium/large metric | `per_stream_sap.sap_{small,medium,large}` | `per_stream_sap.sap_{s,m,l}` (same fields, shorter names) | same | **yes** |
| 24-log aggregation | per-log eval → paired mean (NPU − GPU) per size | per-log eval → paired mean | per-log eval → paired mean | yes |
| Warmup-skip | 30 frames | 30 frames | 30 frames | yes |
| mxq binary actually loaded | `b2441f9d` (11.87 MB) | `b2441f9d` (single-mode fallback to un-suffixed) | `4ff08b08` (new global8) / `3ade8e50` (new single) | **no** |
| `infer_mode` parameter | `global8` | `single` | `global8` (single-stream phase), `single` (multistream phase) | mixed |

**Verdict: eval-path is identical.** The two mismatched rows are both consequences of the mxq layer:
- The mxq-binary row is the primary cause (D1 confirmed).
- The `infer_mode` row, in the *pre-B3 world*, did not change the binary loaded (`_find_yolo11s_mxq` fell back to a single un-suffixed `.mxq`), but it changed runtime resource allocation enough to alter per-frame sAP. From B3 onward, `infer_mode` *also* selects between different binaries.

Per-line audit: `results/evalpath_audit.md`.

## D3 — relative Q and L

GPU FP32 baseline AP per size, per detector (mxq-independent — GPU side never touches the NPU):

| detector | GPU_FP32 AP_s | AP_m | AP_l |
|---|---|---|---|
| YOLO11s | 0.0159 | 0.1839 | 0.4768 |
| YOLO11m | 0.0282 | 0.2152 | 0.5192 |
| YOLO11l | 0.0340 | 0.2350 | 0.5358 |
| YOLO11x | 0.0420 | 0.2568 | 0.5516 |

**Single-stream relative loss** (Q and L from B3 sweep, divided by per-detector per-size GPU FP32 baseline):

| detector | rel_Q_s | rel_Q_m | rel_Q_l | rel_L_s | rel_L_m | rel_L_l |
|---|---|---|---|---|---|---|
| YOLO11s | $-39.8\%$ | $-34.0\%$ | $-20.5\%$ | $+9.3\%$ | $+13.4\%$ | $+13.9\%$ |
| YOLO11m | $-44.4\%$ | $-29.1\%$ | $-24.3\%$ | $+17.9\%$ | $+23.2\%$ | $+22.2\%$ |
| YOLO11l | $-41.2\%$ | $-33.6\%$ | $-26.2\%$ | $+22.3\%$ | $+25.9\%$ | $+25.9\%$ |
| YOLO11x | $-48.0\%$ | $-36.6\%$ | $-30.1\%$ | $+24.4\%$ | $+26.4\%$ | $+26.8\%$ |
| mean abs (across detectors) | **43.4%** | 33.3% | 25.3% | 18.5% | 22.2% | 22.2% |

**Multi-stream concurrency-induced L** (N=4 L1_light, GPU-resident streams, averaged across detectors and placements):

| size | mean abs `rel_L_multi` |
|---|---|
| small | 7.7% |
| medium | 13.3% |
| large | 11.9% |

### Verdicts

- **(Q1) relative_Q is SMALL-CONCENTRATED** ✓ (43% > 33% > 25%, monotone across detectors). Matches the paper §3.3 prediction that the INT8 quantization loss concentrates on small objects in relative terms.
- **(Q2) relative_L is NOT small-concentrated.** Single-stream: 18% / 22% / 22% (s/m/l) — medium and large carry equal-to-larger share. Multi-stream: 8% / 13% / 12% — medium concentrated, not small. The paper's §3.3 phrasing "small boxes age out of alignment soonest, so their $L_i(C)$ climbs fastest" describes the *intuition* about the crossing point of $L$ with $Q$, not the absolute or relative magnitude of $L$ itself. The data is consistent with "small-rich cameras cross $C_i^\star$ first" (small $Q_i$ is small enough that even a modest $L_i$ catches it) without requiring $L$ to be small-concentrated.

CSVs: `results/relative_QL.csv` (single-stream per-detector), `results/relative_L_multistream.csv` (multi-stream aggregated per detector × placement × size), `results/relative_L_multistream_raw.csv` (per-stream raw).

## Diagnosis

| issue | attribution |
|---|---|
| Issue A — Table 1 vs tab:gen-decomp medium/large mismatch | **mxq drift (primary, D1)** + `infer_mode`-induced runtime variation on the original binary (secondary). The eval-path is clean (D2). |
| Issue B — relative $Q$ ordering | small-concentrated (matches §3.3) |
| Issue B — relative $L$ ordering | medium / large weighted ($\sim 22\%$ in single-stream, $\sim 13\%$ medium in multi-stream); NOT small-concentrated |

## Recommendation

For Issue A (preferred order, pick one):

1. **Pin to the legacy mxq for the v11s anchor and re-run B3 on the legacy mxq.** The original `b2441f9d` binary is preserved at `models/mobilint_backup/yolo11s.mxq` and in the HF cache snapshot `8e62b1a9`. Pass `local_path=` to the `mblt_model_zoo` class constructor to bypass the new download. Re-run `phase_b3_v11_global8.py` overriding the v11s path; v11m/l/x mxq files have no legacy backup so they continue to use the new model_zoo binaries. Outcome: Table 1 reproducible, but YOLO11s within the family uses a different binary than YOLO11m/l/x.
2. **Standardize on the new mxq.** Re-measure Table 1 / `tab:partA` / `tab:main-comparison` using the new `4ff08b08` (global8) binary, replacing the published v11s anchors. Outcome: family-wide consistency, but the published Table 1 values change.
3. **Document both as separate snapshots.** Keep Table 1 as-is (legacy snapshot) and present `tab:gen-decomp` as a separate snapshot of a republished mxq series. Outcome: minimal disruption, but the reader is asked to accept two non-comparable numerical anchors.

For Issue B: amend the §3.3 sentence "small boxes age out of alignment soonest, so their $L_i(C)$ climbs fastest" to "small boxes age out of alignment soonest *relative to their own baseline*, so the *crossing condition* $L_i(C) > Q_i$ is satisfied first there", clarifying that the absolute / relative magnitude of $L$ is largest on medium / large but the *gap to $Q$* closes first for small-rich. The within-detector ordering claim ($C_i^\star$ smallest for small-rich at low-to-mid capacity, flipping at YOLO11x) is preserved.
