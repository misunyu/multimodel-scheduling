# schedules/ (Phase 2 campaign)

- File: `<family>_x<m>.yaml`, m in 1..4. Family key = model names sorted and joined by `+`.
- family_index: 1..22 in the order N ascending, then key (see RUN_PLAN.md section 2).
- Instance order = sorted model names. Instance name = model name (one instance per model).
- Combination id = bit mask, `combination_<mask:03d>`; bit i = 1 puts instance i on GPU. `combination_000` = all-CPU, the largest = all-GPU.
  `placement` is the same mask as a string, instance 0 first (`G`/`C`).
- Rate of an instance: `rate = infps = 0.1 * base_rate * multiplier`, `base_rate` = GPU isolated throughput of the model in
  `offline_profile/offline_profile.json`. The rate does not depend on the placement.
- Layout: `meta` + `combinations.<id>.instances.<name>.{model, execution, base_rate, beta, multiplier, rate, infps}` (executor layout).
  The dataset schedule CSVs (`dataset/*_schedules_x3.csv`) carry the same numbers in the layout of the existing parser
  (`combination_xxx -> instance -> model, execution, infps`); see `dataset/README.md`.
- All 2^N placements of every family are listed; nothing is sampled.
