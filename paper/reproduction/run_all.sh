#!/usr/bin/env bash
# Regenerate every reproducible figure/table in paper/main_vision.tex.
#
# Usage:
#   cd paper/reproduction
#   ./run_all.sh                 # uses ./ .venv or the repo .venv if present
#   PY=/path/to/python ./run_all.sh
#
# Each generator reads only its own ./<item>/data/ inputs, writes its output
# into ./<item>/, and cross-checks the numbers against the read-only paper
# (../main_vision.tex). Nothing outside paper/reproduction/ is modified.
#
# Expected total runtime: well under 1 minute — every item now runs from bundled
# data. fig_quant_conf renders from a bundled aggregated intermediate
# (data/quant_conf_matched_scores.csv), so it no longer needs the Argoverse-HD
# val.json. To REBUILD that intermediate from val.json + the raw dumps, run the
# optional STAGE-1 step (needs the dataset — see fig_quant_conf/README.md):
#     # cd fig_quant_conf && python extract_data.py && cd ..
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Pick an interpreter: explicit $PY > repo .venv > python3
if [ -n "${PY:-}" ]; then
  :
elif [ -x "$HERE/../../.venv/bin/python" ]; then
  PY="$HERE/../../.venv/bin/python"
else
  PY="python3"
fi
echo "Interpreter: $PY"
"$PY" --version

# Runnable items in paper order. fig_failures (hardware-only, GPU+NPU) and
# tab_cotenants (descriptive, no data) are intentionally excluded.
ITEMS=(
  tab_single_stream       # tab:single-stream (Table 1)
  fig_quant_conf          # fig:quant_conf   (figure from bundled intermediate; STAGE-1 extract is optional)
  fig_iou_decay           # fig:iou_decay
  tab_decomp              # tab:decomp       (Table 2)
  fig_persize_sweep       # fig:persize-sweep
  tab_main                # tab:main         (Table 3)
  fig_sweeps              # fig:sweeps       (inline TikZ coordinates)
  tab_policy_comparison   # tab:policy-comparison
  tab_metric_sensitivity  # tab:metric-sensitivity
  tab_detector_generality # tab:detector-generality
)

pass=0; fail=0
for item in "${ITEMS[@]}"; do
  echo
  echo "==================================================================="
  echo ">>> $item"
  echo "==================================================================="
  if ( cd "$item" && "$PY" generate.py ); then
    echo "--- $item: OK"
    pass=$((pass+1))
  else
    echo "--- $item: FAILED (see output above)"
    fail=$((fail+1))
  fi
done

echo
echo "==================================================================="
echo "run_all.sh done: $pass ok, $fail failed (of ${#ITEMS[@]} runnable items)"
echo "Not run here: fig_failures (GPU+NPU hardware), tab_cotenants (descriptive)"
echo "==================================================================="
[ "$fail" -eq 0 ]
