# T3 Code -> MiniCPM5-2B QLoRA experiment

A training run on the developer's own T3 Code history, to see whether a small local model
learns to reply with the diffs the developer actually kept.

- Base model: `openbmb/MiniCPM5-2B` (Llama architecture, Apache-2.0). Not the GGUF repo, which cannot be trained.
- Method: 4-bit NF4 QLoRA (double quant), LoRA r=16, alpha=32, dropout 0.05, on all attention and MLP projections.
- Task: given the developer's prompt, reply with the unified diff they kept. Loss only on the diff tokens.
- Evaluation is a proxy (held-out loss, parse rate, file overlap, line similarity). It is not the full replay and test gate from `plan.txt`.

## Data rules applied

Taken from `plan.txt` ("Principles and data rules"):

- Kept: developer-typed user messages (`createdBy = user` on the T3 user_message item) on top-level threads, and `file_change` diffs.
- Dropped and never written to disk: assistant messages, reasoning, subagent turns, dynamic tool arguments, handoff and compaction text.
- Excluded threads: those with a parent thread (subagent or fork children), and handoff or context-transfer targets.
- Forwarded or injected-looking prompts, and secret-pattern hits in prompt or diff, drop the whole example. Only counts are kept.
- An 8-word sequence shared with the same thread's assistant or reasoning text drops the example.
- Agent-written kept diffs are included (owner decision, 2026-11-21).
- Split: per workspace, the newest 20% of threads (by first message time) are held out. Whole threads stay together.
- Caps: prompt 4,000 chars; each diff 20,000 chars; examples with more than 30,000 diff chars are dropped. Examples longer than the chosen context are truncated at train time, not dropped.

Counts and drop reasons: `work/extract_stats.json`.

## Stages

| Script | What it does |
|---|---|
| `extract.py` | Reads `%USERPROFILE%\.t3\userdata\statev2.sqlite` read-only (falls back to a backup-API snapshot). Writes `work/train.jsonl`, `work/heldout.jsonl`, `work/extract_stats.json`. |
| `train.py` | Downloads the model into `work/model`. Calibrates context length (8192 down to 1024; 3 steps each; first that stays at or under 5.6 GiB reserved VRAM). Trains 3 epochs, cosine LR 2e-4, grad accumulation 8, paged AdamW 8-bit. Saves `work/adapter`. |
| `evaluate.py` | Loss on all held-out examples for base and trained (same examples and context), then greedy generation while time allows. Writes `work/REPORT.md`. |
| `pipeline.sh` | Runs the three stages in order and logs to `work/pipeline.log`. |

## How to rerun

```bash
cd /c/codebases/Dawnpatch/experiments/t3-minicpm5
bash pipeline.sh
```

Environment overrides (optional): `TRAIN_DEADLINE=HH:MM` (default 06:15), `EVAL_DEADLINE=HH:MM` (default 07:40), `PY=<python>`.
`extract.py` is skipped if `work/train.jsonl` exists. Delete it to re-extract.

Smoke test (small, writes to `work/smoke/`):

```bash
SMOKE=1 MAX_TRAIN_SECONDS=120 python -I train.py
SMOKE=1 MAX_EVAL_SECONDS=300 python -I evaluate.py
```

## Outputs (all under `work/`, git-ignored)

- `work/pipeline.log`: timestamped stage markers and full stdout/stderr
- `work/extract_stats.json`, `work/train.jsonl`, `work/heldout.jsonl`
- `work/calibration.json`: context trials and the chosen length
- `work/train_progress.jsonl`: one line per logged step (loss, lr, peak reserved GB)
- `work/checkpoints/`: checkpoints (every 200 steps, keep 3)
- `work/adapter/`: final LoRA adapter
- `work/train_summary.json`
- `work/REPORT.md`: dataset, training, loss and generation tables, 5 samples, verdict

## Timing

Launched around 00:50 local on 2026-11-22. Training stops at the 06:15 deadline at the latest.
Evaluation stops at 07:40, and `REPORT.md` is final by about 07:45.
Training normally finishes earlier, since 3 epochs over the examples is much less than the 06:15 deadline allows.

## Known limits

- Small data: about 117 training examples from 64 threads. Overfitting is the main risk.
- The model's chat template is used as-is; the system prompt is fixed.
- Diffs are used as stored in T3 (`diffStr`). They may lack `---`/`+++` file headers; the eval counts `@@` as parseable.
- Context length is chosen by calibration on the 3 longest examples. Long examples are truncated to fit (head and tail of the prompt kept; the diff cut at the end).
- Window-only VRAM spill is treated as "does not fit" (reserved above 5.6 GiB).
