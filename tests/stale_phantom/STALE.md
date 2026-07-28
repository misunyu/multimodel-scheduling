# STALE — phantom-vocabulary schedules (quarantined)

These schedules name prior-generation models the runtime cannot resolve
(`mnasnet, resnext50, vgg19, shufflenet-v2-12, squeezenet1.0-12, yolov4`, and the
alias-collapsed `resnet50_big/_small`, `yolov3_*`). v21 fail-fast halts any run that
loads them.

**They are NOT the source of the paper's Q3/Q6/Q5 numbers** — Gate B
(`docs/gate_b_schedule_provenance.md`) established from the raw logs that those runs
used generated temp schedules over Table I vocabulary only, with every model producing
inferences. These files are kept for audit history. Do not use, do not delete.

## Scripts pointing at these paths

Several scripts still reference the old `tests/<name>.yaml` paths
(`ml_misprediction_validation.py`, `dynamic_load*_validation.py`,
`run_bounded_recovery_sweep.py`, `runtime_overhead_analysis.py`, `qos_recovery_validation.py`,
`q5_run_infeasible.py`, ...). Those scripts were **already non-runnable**: v21 fail-fast
halts any run whose schedule names an unresolvable model. Moving the files turns a
mid-run halt into an earlier missing-file error; nothing that previously worked breaks.
They will be repointed when new-vocabulary schedules are created (task 85, pending the
two open decisions in docs/gate_b_schedule_provenance.md).
