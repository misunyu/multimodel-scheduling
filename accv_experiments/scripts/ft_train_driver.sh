#!/usr/bin/env bash
set -u
BASE=/home/msyu/PycharmProjects/multimodel-scheduling-video
cd "$BASE"
source .venv/bin/activate
PROJ="$BASE/accv_experiments/results/ft_runs"; mkdir -p "$PROJ"
ST="$PROJ/STATUS_FT"; LOG="$PROJ/train.log"
DATA="$BASE/accv_experiments/data/ahd_yolo/ahd_coco80.yaml"
echo "[$(date '+%F %T')] TRAINING start" > "$ST"
yolo detect train model=yolo11s.pt data="$DATA" epochs=50 imgsz=640 seed=0 \
  deterministic=True project="$PROJ" name=ft_yolo11s exist_ok=True \
  >> "$LOG" 2>&1
rc=$?
BEST="$PROJ/ft_yolo11s/weights/best.pt"
if [ $rc -eq 0 ] && [ -f "$BEST" ]; then
  echo "DONE best=$BEST rc=0 $(date '+%F %T')" > "$ST"
else
  echo "FAIL rc=$rc $(date '+%F %T')" > "$ST"
fi
