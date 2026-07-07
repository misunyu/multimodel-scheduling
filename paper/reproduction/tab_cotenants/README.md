# tab:cotenants — GPU co-tenant workloads

**Paper label:** `tab:cotenants`
**Paper location:** Section 4.1 (Experimental Setup → Co-tenant workloads), the first table in the paper.
**Caption:** "GPU co-tenant workloads."

## Nature of this table — no data source

This is a **purely descriptive** table. It lists the three GPU co-tenant levels used
throughout the paper and characterizes each in words. It contains **no measured numbers**,
so there is **no generator script and no input data to reproduce**.

Its exact content is defined literally in `paper/main_vision.tex` (the `tab:cotenants` block):

| Level | Models | Characterization |
|---|---|---|
| L1_CNN | ResNet50 | Light GPU co-tenant |
| L2_LM  | ResNet50 + TinyLLaMA-1.1B | GPU and host-CPU pressure |
| L3_VLM | ResNet50 + Qwen2-VL-2B | GPU-intensive vision-language workload |

These three levels (`\Lcnn`, `\Llm`, `\Lvlm`) are the co-tenant conditions that appear as
rows/points in the quantitative artifacts:
- `tab:main` (worst/mean sAP at each co-tenant level) — see `../tab_main/`
- `tab:policy-comparison` (selection-policy regret at each level) — see `../tab_policy_comparison/`
- `fig:sweeps` (the ResNet50 = L1_CNN contention sweep) — see `../fig_sweeps/`

## Reproduction status

**No script needed (descriptive table).** Nothing to run. The table is verified by
inspection against the workload definitions used by the measurement scripts (ResNet50 as
the CNN co-tenant, TinyLLaMA-1.1B as the host-heavy LM probe, Qwen2-VL-2B as the GPU VLM).
