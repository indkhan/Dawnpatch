#!/usr/bin/env bash
# Resume after the 8192 OOM: probe each context in its own process (an OOM can poison the CUDA
# context), then train at the largest one that fits, then evaluate. Same deadlines as pipeline.sh.
cd "$(dirname "$0")"
PY="/c/codebases/Dawnpatch/spikes/model-compat/.venv/Scripts/python.exe"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 TRAIN_DEADLINE=06:15 EVAL_DEADLINE=07:40
unset SMOKE MAX_TRAIN_SECONDS MAX_EVAL_SECONDS
LOG=work/pipeline.log
note() { echo "$(date '+%F %T') $*" >> "$LOG"; }
note "== RUN_REST start (per-process context probing)"
rm -f work/calibration.json
CHOSEN=""
for L in 4096 3072 2048 1536 1024; do
  note "== PROBE $L start"
  if CONTEXTS=$L CALIB_ONLY=1 timeout 900 "$PY" -I train.py >> "$LOG" 2>&1; then
    CHOSEN=$L; note "== PROBE $L fits"; break
  fi
  note "== PROBE $L does not fit"
done
if [ -z "$CHOSEN" ]; then note "PIPELINE STOP: no context fits"; exit 1; fi
note "== STAGE train start (context $CHOSEN)"
CONTEXTS=$CHOSEN "$PY" -I train.py >> "$LOG" 2>&1; TX=$?
note "== STAGE train end exit=$TX"
if [ -d work/adapter ] || ls -d work/checkpoints/checkpoint-* >/dev/null 2>&1; then
  note "== STAGE evaluate start"
  "$PY" -I evaluate.py >> "$LOG" 2>&1; EX=$?
  note "== STAGE evaluate end exit=$EX"
else
  note "== STAGE evaluate skipped: no adapter or checkpoint"
fi
note "== PIPELINE end"
