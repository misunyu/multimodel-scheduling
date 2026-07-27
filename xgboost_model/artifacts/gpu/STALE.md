# STALE — legacy (pre-MLA100) predictor artifacts

These `xgb_model_x3_*` / `xgb_model_random_*` bundles are the **pre-MLA100**
two-target predictor, trained on the **old working set** (resnext50, vgg19,
yolov4, ...). They produced the legacy paper figures whose candidate rankings
are now superseded (see `docs/phantom_model_audit.md`, `docs/ranking_regeneration_report.md`).

**Do not use for new rankings.** New rankings use the three-target MLA100 bundle
in `xgboost_model/artifacts/{cpu_gpu,cpu_npu}/` via `deploy_predictor_logic.DeployPredictor`.
Kept for audit history only — do not delete.

The old `xgboost_model/prediction_result/` score dumps referenced by v20 were
already removed from the tree in an earlier commit; they are not restored here.
