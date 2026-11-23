#!/usr/bin/env bash
# End-to-end run: wait for model download -> smoke tests -> extract -> train -> evaluate.
# All output and stage markers go to work/pipeline.log. A failing smoke stage stops the run.
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p work

PY="${PY:-/c/codebases/Dawnpatch/spikes/model-compat/.venv/Scripts/python.exe}"
LOG="work/pipeline.log"
WEIGHTS="work/model/model-00000-of-00001.safetensors"
EXPECTED=5033557096

unset SMOKE MAX_TRAIN_SECONDS MAX_EVAL_SECONDS
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_DEADLINE="${TRAIN_DEADLINE:-06:15}"
export EVAL_DEADLINE="${EVAL_DEADLINE:-07:40}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$(stamp) $*" >> "$LOG"; }

stage() {
  local name="$1"; shift
  note "== STAGE $name start"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  note "== STAGE $name end exit=$rc"
  return $rc
}

size_of() { [ -f "$1" ] && stat -c %s "$1" || echo 0; }

note "== PIPELINE start (TRAIN_DEADLINE=$TRAIN_DEADLINE EVAL_DEADLINE=$EVAL_DEADLINE)"

# 1. Wait for the weights download (work/download.sh), polling every 60 s.
note "== STAGE download wait start (expected $EXPECTED bytes)"
while [ "$(size_of "$WEIGHTS")" -ne "$EXPECTED" ]; do
  note "download at $(size_of "$WEIGHTS") / $EXPECTED bytes"
  sleep 60
done
note "== STAGE download wait end: complete"

# 2. Extract (skipped if already done).
if [ -f work/train.jsonl ]; then
  note "== STAGE extract skipped (work/train.jsonl exists)"
else
  stage extract "$PY" -I extract.py || { note "PIPELINE STOP: extract failed"; exit 1; }
fi

# 3. Smoke tests: train then evaluate on tiny inputs. Any failure stops the pipeline.
if ! stage smoke_train env SMOKE=1 MAX_TRAIN_SECONDS=120 "$PY" -I train.py; then
  note "PIPELINE STOP: smoke train failed (see errors above in this log)"; exit 1
fi
if ! stage smoke_eval env SMOKE=1 MAX_EVAL_SECONDS=300 "$PY" -I evaluate.py; then
  note "PIPELINE STOP: smoke eval failed (see errors above in this log)"; exit 1
fi
rm -rf work/smoke
note "== STAGE smoke passed; work/smoke removed"

# 4. Real training (stops at TRAIN_DEADLINE, always saves the adapter).
stage train "$PY" -I train.py
train_rc=$?

# 5. Real evaluation (runs if an adapter or checkpoint exists).
if [ -f work/adapter/adapter_model.safetensors ] || ls work/checkpoints/checkpoint-*/adapter_model.safetensors >/dev/null 2>&1; then
  stage evaluate "$PY" -I evaluate.py
  eval_rc=$?
else
  note "== STAGE evaluate skipped: no adapter or checkpoint (train exit=$train_rc)"
  eval_rc=1
fi

note "== PIPELINE end train_exit=$train_rc eval_exit=$eval_rc"
exit $eval_rc
