#!/bin/bash
# Phase 1b pilot, stages before beta selection. nice 0 (decided in section 2). Run from revision_profiling/.
set -u
cd "$(dirname "$0")/.."
PY="../.venv/bin/python3 -B -W ignore"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export REVPROF_NICE=0
F4="gpt2+mnasnet+resnet50+resnext50"
F7="gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19"
echo "== 5.1 smoke (beta 0.1, 1x) $(date +%T)"
$PY code/gen_schedules.py --single --beta 0.1 --mult 1 --out-dir pilot/smoke/schedules
$PY code/run_campaign.py --out-dir pilot/smoke/runs pilot/smoke/schedules/*.yaml
echo "== 5.3(B) single GPU 1x, beta 0.05 and 0.25 $(date +%T)"
for B in 0.05 0.25; do
  $PY code/gen_schedules.py --single --beta $B --mult 1 --masks all_gpu --out-dir pilot/beta_$B/single_gpu/schedules
  $PY code/run_campaign.py --out-dir pilot/beta_$B/single_gpu/runs pilot/beta_$B/single_gpu/schedules/*.yaml
done
echo "== 5.3(C) N=4 all-GPU 1x per beta $(date +%T)"
for B in 0.05 0.1 0.25; do
  $PY code/gen_schedules.py --family "$F4" --beta $B --mult 1 --masks all_gpu --out pilot/rate_range/schedules/n4_b${B}_x1.yaml
done
$PY code/run_campaign.py --out-dir pilot/rate_range/runs pilot/rate_range/schedules/*.yaml
echo "== 5.2 arrival (3 betas x {2,4}x x {N4,N7} x 3 placements x 3 repeats) $(date +%T)"
for B in 0.05 0.1 0.25; do for M in 2 4; do
  $PY code/gen_schedules.py --family "$F4" --beta $B --mult $M --masks all_gpu,all_cpu,half --out pilot/arrival/schedules/n4_b${B}_x${M}.yaml
  $PY code/gen_schedules.py --family "$F7" --beta $B --mult $M --masks all_gpu,all_cpu,half --out pilot/arrival/schedules/n7_b${B}_x${M}.yaml
done; done
$PY code/run_campaign.py --out-dir pilot/arrival/runs --repeats 3 pilot/arrival/schedules/*.yaml
echo "== done $(date +%T)"
