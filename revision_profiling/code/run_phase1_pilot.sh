#!/bin/bash
# Phase 1 pilot, stages 1-4 (spec v2.2 sections 5.1-5.3). Stage 5 (5.4) runs after beta is chosen.
# Usage: bash code/run_phase1_pilot.sh   (from revision_profiling/, long running: use nohup)
set -u
cd "$(dirname "$0")/.."
PY="../.venv/bin/python3 -B -W ignore"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
F4=g_m_r_t
F8=g_m_resnet50_resnext50_shufflenet-v2-12_squeezenet1.0-12_t_v
echo "== stage 1: offline profile (closed loop) $(date +%T)"
$PY code/offline_profile.py
echo "== stage 2: smoke 5.1 (beta 0.5, 1x, single instance, cpu+gpu) $(date +%T)"
$PY code/gen_schedules.py --single --beta 0.5 --mult 1 --out-dir pilot/smoke/schedules
$PY code/run_campaign.py --out-dir pilot/smoke/runs pilot/smoke/schedules/*.yaml
echo "== stage 3: 5.3(B) single GPU 1x for beta 0.25 and 1.0 $(date +%T)"
for B in 0.25 1.0; do
  $PY code/gen_schedules.py --single --beta $B --mult 1 --masks all_gpu --out-dir pilot/beta_$B/single_gpu/schedules
  $PY code/run_campaign.py --out-dir pilot/beta_$B/single_gpu/runs pilot/beta_$B/single_gpu/schedules/*.yaml
done
echo "== stage 4: 5.2 arrival (3 betas x {2,4}x x {N4,N8} x {all_gpu,all_cpu,half} x 2 repeats) $(date +%T)"
for B in 0.25 0.5 1.0; do for M in 2 4; do
  $PY code/gen_schedules.py --family $F4 --beta $B --mult $M --masks all_gpu,all_cpu,half --out pilot/arrival/schedules/n4_${F4}_b${B}_x${M}.yaml
  $PY code/gen_schedules.py --family $F8 --beta $B --mult $M --masks all_gpu,all_cpu,half --out pilot/arrival/schedules/n8_b${B}_x${M}.yaml
done; done
$PY code/run_campaign.py --out-dir pilot/arrival/runs --repeats 2 pilot/arrival/schedules/*.yaml
echo "== done $(date +%T)"
