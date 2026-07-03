# D2 eval-path audit (Table 1 source vs gen_decomp Q)

| parameter | Table 1 (step_a) | gen_decomp Q (step_k) | B3 (phase_b3) | same | note |
|---|---|---|---|---|---|
| COCO areaRng (small <32², medium <96², large ≥96²) | pycocotools default | pycocotools default |  | yes | Neither pipeline sets e.params.areaRng; pycocotools uses [[0,1e10],[0,32**2],[32**2,96**2],[96**2,1e10]] default |
| IoU range | [.50:.95] via e.stats[3..5] for size-stratified AP | [.50:.95] via e.stats[3..5] for size-stratified mAP and sAP |  | yes | _step_d_common.per_stream_sap uses e.stats[0]=ap_5095 and e.stats[3..5]=size-stratified |
| Metric for AP_small/medium/large (Q definition) | Streaming sAP (per_stream_sap reports sap_small/medium/large) | Streaming sAP from step_a baseline (NPU side) + offline mAP from gen_decomp (NPU side, single mode) |  | **no** | gen_decomp_v2 Q used per_stream_sap sap_s/m/l on a per-sid basis, paired GPU-NPU mean. step_a uses identical per_stream_sap. → same metric. BUT they used DIFFERENT mxq (D1 finding). |
| Aggregation across 24 logs | mean±std across 24 per-log per_stream_sap values | mean across 24 per-log per_stream_sap values (paired GPU-NPU mean per size) |  | yes | Same per-log eval, same averaging step |
| Warmup-skip frames | 30 (WARMUP_FRAMES in _step_d_common) | 30 (WARMUP_FRAMES in _step_d_common) |  | yes | Shared constant |
| yolo11s.mxq actually loaded | sha256 b2441f9d... (11.87 MB, HF revision 8e62b1a9, mtime 2026-05-27) | sha256 b2441f9d... (same file via _find_yolo11s_mxq fallback to un-suffixed) | (same as gen_decomp Q) | yes | BOTH step_a and step_k used the OLD mxq (b2441f9d). The DIFFERENT mxq (4ff08b08 global8 / 3ade8e50 single) only enters with B3's phase_b3_v11_global8.py via the new model_zoo class. |
| infer_mode at construction time |  |  | global8 (NEW mxq 4ff08b08) | **no** | Pre-B3 there was only one mxq file (b2441f9d) used in both global8 and single mode -- infer_mode controls runtime resource allocation, not weights. From B3 onward, model_zoo downloads mode-specific mxq binaries, so infer_mode now selects which BINARY is loaded. |
