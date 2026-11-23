#!/usr/bin/env python3
"""QLoRA on MiniCPM5-2B: request + file -> edit. One run per edit format (FORMAT=sr|udiff).

Reuses v1's model loading, chunked loss trainer and callbacks. Changes from v1:
  - thinking is switched off in the chat template (empty <think></think>), for training and generation
  - examples longer than the context are DROPPED, never truncated, so every target ends with <|im_end|>
  - both arms train on the same examples: an example is kept only if it fits in both formats
  - 2 epochs (v1: 3)

Env: FORMAT=sr|udiff (required), MAX_LEN (default 3072), EPOCHS (default 2),
     MAX_TRAIN_SECONDS (time limit, adapter is saved when it hits), SMOKE=1
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "t3-minicpm5"))
import train as T  # noqa: E402  v1 trainer helpers
import edits as E  # noqa: E402

FORMAT = os.environ["FORMAT"]
assert FORMAT in E.FORMATS, FORMAT
SMOKE = os.environ.get("SMOKE") == "1"
WORK = HERE / "work"
OUT = WORK / (f"smoke-{FORMAT}" if SMOKE else FORMAT)
T.OUT = OUT
T.MODEL_DIR = HERE.parent / "t3-minicpm5" / "work" / "model"
MAX_LEN = int(os.environ.get("MAX_LEN", "3072"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
END = "<|im_end|>"


def log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def prompt_text(tok, fmt, prompt):
    msgs = [{"role": "system", "content": E.FORMATS[fmt]["system"]}, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode(tok, fmt, row):
    p_ids = tok(prompt_text(tok, fmt, row["prompt"]), add_special_tokens=False)["input_ids"]
    c_ids = tok(row[f"target_{fmt}"] + END, add_special_tokens=False)["input_ids"]
    return p_ids + c_ids, [-100] * len(p_ids) + c_ids


def fits_both(tok, row, max_len):
    return all(len(encode(tok, f, row)[0]) <= max_len for f in E.FORMATS)


def main():
    import torch
    Trainer = T.chunked_trainer_class()
    T.set_seeds()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = T.load_jsonl(WORK / "train.jsonl")
    tok, model, _ = T.load_base()
    end_id = tok.convert_tokens_to_ids(END)
    assert end_id is not None and end_id != tok.unk_token_id, "no <|im_end|> token"
    kept = [r for r in rows if fits_both(tok, r, MAX_LEN)]
    if SMOKE:  # the longest examples that fit: a memory stress test
        kept = sorted(kept, key=lambda r: len(encode(tok, FORMAT, r)[0]), reverse=True)[:8]
    log(f"format={FORMAT} max_len={MAX_LEN}: {len(kept)}/{len(rows)} examples fit in both formats (rest dropped)")
    items = []
    for r in kept:
        ids, labels = encode(tok, FORMAT, r)
        items.append({"input_ids": ids, "labels": labels})
    assert items[0]["labels"][-1] == end_id and items[0]["labels"][0] == -100
    log(f"label check: first target starts {tok.decode([x for x in items[0]['labels'] if x != -100])[:80]!r}")

    limit = os.environ.get("MAX_TRAIN_SECONDS")
    deadline = dt.datetime.now() + dt.timedelta(seconds=float(limit)) if limit else dt.datetime.max
    progress = OUT / "train_progress.jsonl"
    progress.unlink(missing_ok=True)
    cbs = T.make_callbacks(deadline, progress, time.time())
    args = T.make_args(OUT / "checkpoints", num_train_epochs=EPOCHS, gradient_accumulation_steps=8,
                       warmup_steps=5, logging_steps=2, save_strategy="no")
    trainer = Trainer(model=model, args=args, train_dataset=items,
                      data_collator=T.make_collator(tok.pad_token_id), callbacks=[cbs[0]])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train()
    model.save_pretrained(str(OUT / "adapter"))
    summary = {
        "format": FORMAT, "max_length": MAX_LEN, "examples": len(items), "examples_offered": len(rows),
        "epochs_completed": trainer.state.epoch, "global_step": trainer.state.global_step,
        "train_loss": res.training_loss, "train_seconds": round(time.time() - t0, 1),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "stopped_by_time_limit": dt.datetime.now() >= deadline,
        "train_threads": sorted({r["thread_id"] for r in kept}),
    }
    (OUT / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"saved adapter: { {k: v for k, v in summary.items() if k != 'train_threads'} }")


if __name__ == "__main__":
    sys.exit(main())
