#!/usr/bin/env bash
# Detached driver: download Argoverse 1.1 tracking train tars, extract only
# ring_front_center, validate against train.json. Session-independent (setsid).
set -u
BASE=/home/msyu/PycharmProjects/multimodel-scheduling-video
DATA="$BASE/accv_experiments/data/argoverse_hd/Argoverse-1.1/argoverse-tracking"
TARDIR="$BASE/accv_experiments/data/argoverse_hd/_train_tars"
STAGE="$TARDIR/_stage"
ANNOT="$BASE/accv_experiments/data/argoverse_hd/Argoverse-HD/annotations/train.json"
LOG="$TARDIR/driver.log"; STATUS="$TARDIR/STATUS"
URLBASE="https://s3.amazonaws.com/argoverse/datasets/av1.1/tars"
PY="$BASE/.venv/bin/python"
mkdir -p "$DATA/train" "$STAGE"
say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
setstatus(){ echo "$*" > "$STATUS"; say "STATUS=$*"; }

dl(){ local n=$1; say "download train$n start"
  aria2c -c -x16 -s16 -k1M --file-allocation=none --retry-wait=5 --max-tries=30 \
    --console-log-level=warn -d "$TARDIR" -o "tracking_train${n}_v1.1.tar.gz" \
    "$URLBASE/tracking_train${n}_v1.1.tar.gz" >> "$LOG" 2>&1
  local rc=$?; [ $rc -eq 0 ] || { setstatus "DOWNLOAD_FAIL train$n rc=$rc"; exit 1; }
  say "download train$n ok"; }

extract(){ local n=$1; rm -rf "${STAGE:?}"/*; say "extract train$n front-center"
  tar -xzf "$TARDIR/tracking_train${n}_v1.1.tar.gz" -C "$STAGE" --wildcards '*/ring_front_center/*' >> "$LOG" 2>&1
  local rc=$?; [ $rc -eq 0 ] || { setstatus "EXTRACT_FAIL train$n rc=$rc"; exit 1; }
  find "$STAGE" -type d -name ring_front_center | while read -r d; do
    seq=$(basename "$(dirname "$d")"); mkdir -p "$DATA/train/$seq"
    rm -rf "$DATA/train/$seq/ring_front_center"; mv "$d" "$DATA/train/$seq/ring_front_center"
  done
  rm -f "$TARDIR/tracking_train${n}_v1.1.tar.gz" "$TARDIR/tracking_train${n}_v1.1.tar.gz.aria2"
  rm -rf "${STAGE:?}"/*; say "train$n placed + tar removed"; }

validate(){ "$PY" - "$ANNOT" "$DATA" "$TARDIR/_present" <<'PYEOF' >> "$LOG" 2>&1
import json,os,sys
ann=json.load(open(sys.argv[1])); data=sys.argv[2]
imgs=ann["images"]; seqd=ann.get("seq_dirs")
present=sum(1 for im in imgs if seqd and os.path.exists(os.path.join(data,seqd[im["sid"]],im["name"])))
open(sys.argv[3],"w").write(str(present)); print(f"VALIDATE present={present}/{len(imgs)}")
PYEOF
}

setstatus "RUNNING"
dl 4; extract 4; validate
PRES=$(cat "$TARDIR/_present" 2>/dev/null || echo 0); say "after train4 present=$PRES"
[ "$PRES" -ge 1 ] || { setstatus "VALIDATE_FAIL_train4 present=$PRES"; exit 1; }
setstatus "TRAIN4_OK present=$PRES"
for n in 1 2 3; do dl $n; extract $n; validate; say "after train$n present=$(cat "$TARDIR/_present")"; done
PRES=$(cat "$TARDIR/_present"); EXPECT=39384
if [ "$PRES" -ge $((EXPECT*98/100)) ]; then setstatus "DONE present=$PRES/$EXPECT"; else setstatus "COUNT_MISMATCH present=$PRES/$EXPECT"; fi
say "driver finished"
