#!/usr/bin/env bash
# Runs the full Nanbeige spike: finish download, train, resume, generate, context sweep.
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
mkdir -p outputs
echo "== download $(date +%T)"
$PY -c "from huggingface_hub import snapshot_download as s; print(s('Nanbeige/Nanbeige4.2-3B', revision='b82e54bd609793562a75cbf9337970a93369eab5', allow_patterns=['*.safetensors','*.json','*.py','tokenizer*']))" || exit 1
echo "== train 1024 x3 $(date +%T)"
$PY spike.py train --seq 1024 --steps 3 --save-steps 3 --out outputs/run1 || echo "TRAIN_FAILED"
echo "== resume to 5 $(date +%T)"
$PY spike.py train --seq 1024 --steps 5 --save-steps 5 --resume --out outputs/run1 || echo "RESUME_FAILED"
echo "== generate $(date +%T)"
$PY spike.py generate --out outputs/run1 || echo "GENERATE_FAILED"
for L in 2048 4096 8192; do
  echo "== sweep $L $(date +%T)"
  if ! $PY spike.py train --seq $L --steps 2 --save-steps 100 --out outputs/seq$L; then echo "SWEEP_FAILED $L"; break; fi
done
echo "== end $(date +%T)"
