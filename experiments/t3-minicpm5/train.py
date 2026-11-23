#!/usr/bin/env python3
"""4-bit QLoRA fine-tune of openbmb/MiniCPM5-2B on T3 prompt -> kept-diff examples.

Stages inside one process:
  1. ensure the model files are in work/model (plain files, no HF cache symlinks)
  2. load 4-bit NF4 base, add LoRA r=16
  3. context calibration: 3 steps per max_length, largest first; pick the first that
     finishes and keeps torch.cuda.max_memory_reserved() <= 5.6 GiB
  4. train (cosine, lr 2e-4, 3 epochs, grad accum 8), loss on assistant tokens only
  5. save adapter to work/adapter

Env:
  TRAIN_DEADLINE=HH:MM   absolute local time; training stops there, adapter is saved
  MAX_TRAIN_SECONDS=N    relative limit (used by the smoke test); overrides TRAIN_DEADLINE
  SMOKE=1                20 train examples, outputs in work/smoke/, calibration at 1024 only
"""
import datetime as dt
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
SMOKE = os.environ.get("SMOKE") == "1"
OUT = WORK / "smoke" if SMOKE else WORK
MODEL_DIR = WORK / "model"
MODEL_ID = "openbmb/MiniCPM5-2B"
SYSTEM = "You are a coding assistant. Reply with a unified diff only."
CONTEXTS = [1024] if SMOKE else [8192, 6144, 4096, 3072, 2048, 1536, 1024]
if os.environ.get("CONTEXTS"):  # coordinator: probe one context per process
    CONTEXTS = [int(x) for x in os.environ["CONTEXTS"].split(",")]
GIB = 1024 ** 3
MEM_LIMIT = 5.6 * GIB
TRIAL_TIMEOUT_S = 600
SEED = 42
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seeds():
    import numpy as np
    import torch
    import transformers
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    transformers.set_seed(SEED)


WEIGHTS = "model-00000-of-00001.safetensors"
WEIGHTS_BYTES = 5033557096  # size reported by the Hub API for this file


def ensure_model():
    """The model is downloaded by work/download.sh (resumable curl). Only verify it here."""
    w = MODEL_DIR / WEIGHTS
    have = w.stat().st_size if w.exists() else 0
    if have != WEIGHTS_BYTES:
        raise SystemExit(f"model incomplete: {w} is {have} bytes, expected {WEIGHTS_BYTES}. "
                         "Wait for work/download.sh to finish.")
    if not (MODEL_DIR / "config.json").exists():
        raise SystemExit("model config.json missing in " + str(MODEL_DIR))


def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------- encoding
def encode(tok, prompt, diff, max_len):
    """Return (input_ids, labels). Labels are -100 except assistant (diff) tokens.

    Over-long examples are truncated, not dropped: the prompt keeps its head and tail
    (so the chat header and generation prompt survive) and the diff is cut at the end.
    """
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    p_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": diff}], tokenize=False)
    if not full.startswith(p_text):
        raise RuntimeError("chat template: full conversation does not start with the generation prompt")
    c_text = full[len(p_text):]
    p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
    c_ids = tok(c_text, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(c_ids) > max_len:
        budget_c = min(len(c_ids), max_len // 2)
        budget_p = max_len - budget_c
        if len(p_ids) > budget_p:
            head = budget_p // 2
            tail = budget_p - head
            p_ids = p_ids[:head] + p_ids[len(p_ids) - tail:]
        c_ids = c_ids[: max_len - len(p_ids)]
    return p_ids + c_ids, [-100] * len(p_ids) + c_ids


class EncodedDataset:
    def __init__(self, tok, rows, max_len):
        self.items = []
        self.truncated = 0
        for r in rows:
            ids, labels = encode(tok, r["prompt"], r["diff"], max_len)
            full_len = len(encode(tok, r["prompt"], r["diff"], 10 ** 9)[0])
            if full_len > max_len:
                self.truncated += 1
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id):
    import torch

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": att}

    return collate


def verify_masking(tok, ds):
    ex = ds[0]
    n_label = sum(1 for x in ex["labels"] if x != -100)
    label_ids = [x for x in ex["labels"] if x != -100]
    text = tok.decode(label_ids)
    log(f"masking check: total tokens={len(ex['input_ids'])}, label tokens (not -100)={n_label}")
    log(f"masking check: first 80 label chars: {text[:80]!r}")
    assert n_label > 0, "no label tokens on first example"
    assert ex["labels"][0] == -100, "prompt tokens must be masked"


# ---------------------------------------------------------------- model
def load_base():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), quantization_config=bnb, dtype=dtype, device_map={"": 0}
    )
    # Memory fix (coordinator, 01:20): prepare_model_for_kbit_training upcasts the non-quantized
    # embedding and lm_head to fp32; with a 130,560-token vocabulary that is ~2 GB on a 6 GB card.
    # Keep them in bf16/fp16, frozen, and only enable gradient checkpointing.
    for p_ in model.parameters():
        p_.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return tok, model, bf16


def snapshot_lora(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_lora(model, snap):
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(snap[n])


# ---------------------------------------------------------------- chunked loss
LOSS_CHUNK = 256


def chunked_trainer_class():
    """Trainer whose loss never materialises full-vocabulary logits for the whole sequence.
    Hidden states are taken from the base LlamaModel; only positions with a label are projected
    through lm_head, in chunks of LOSS_CHUNK tokens with activation checkpointing (fp32 CE)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import Trainer

    def _chunk_loss(h, w, y):
        logits = F.linear(h, w).float()
        return F.cross_entropy(logits, y, reduction="sum")

    class ChunkedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            labels = inputs["labels"]
            out = base.model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                             use_cache=False)
            hidden = out.last_hidden_state[:, :-1, :]
            tgt = labels[:, 1:]
            keep = tgt != -100
            h = hidden[keep]
            y = tgt[keep]
            n = max(int(y.numel()), 1)
            w = base.lm_head.weight
            total = hidden.new_zeros((), dtype=torch.float32)
            for i in range(0, h.shape[0], LOSS_CHUNK):
                total = total + checkpoint(_chunk_loss, h[i:i + LOSS_CHUNK], w, y[i:i + LOSS_CHUNK],
                                           use_reentrant=False)
            # Trainer passes num_items_in_batch = label tokens across the whole accumulated batch
            # and then does NOT divide by gradient_accumulation_steps, so normalise by it here.
            denom = num_items_in_batch if num_items_in_batch is not None else n
            if torch.is_tensor(denom):
                denom = denom.to(total.device).float().clamp(min=1)
            loss = total / denom
            return (loss, None) if return_outputs else loss

    return ChunkedTrainer


# ---------------------------------------------------------------- callbacks
def make_callbacks(deadline, progress_path, start_ts):
    import torch
    from transformers import TrainerCallback

    class ProgressCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                rec = {
                    "time": dt.datetime.now().isoformat(timespec="seconds"),
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "loss": logs["loss"],
                    "lr": logs.get("learning_rate"),
                    "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
                }
                with open(progress_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                log(f"step {rec['step']} epoch {rec['epoch']:.2f} loss {rec['loss']} lr {rec['lr']}")

        def on_step_end(self, args, state, control, **kw):
            if dt.datetime.now() >= deadline:
                log(f"deadline {deadline} reached at step {state.global_step}: stopping")
                control.should_training_stop = True

    class TrialCB(TrainerCallback):
        """Used during calibration: abort a trial that runs too long (spill to shared memory stalls)."""
        def on_step_end(self, args, state, control, **kw):
            if time.time() - start_ts > TRIAL_TIMEOUT_S:
                control.should_training_stop = True

    return ProgressCB(), TrialCB()


def make_args(out_dir, **kw):
    import torch
    from transformers import TrainingArguments

    bf16 = torch.cuda.is_bf16_supported()
    base = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        seed=SEED,
        data_seed=SEED,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=5,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        disable_tqdm=True,
        gradient_checkpointing=False,
    )
    base.update(kw)
    return TrainingArguments(**base)


# ---------------------------------------------------------------- calibration
def calibrate(tok, model, rows, bf16):
    import torch
    Trainer = chunked_trainer_class()

    snap = snapshot_lora(model)
    collate = make_collator(tok.pad_token_id)
    # The 3 longest examples are the stress test for each context length.
    longest = sorted(rows, key=lambda r: len(r["prompt"]) + len(r["diff"]), reverse=True)[:3]
    trials = []
    chosen = None
    for L in CONTEXTS:
        restore_lora(model, snap)
        ds = EncodedDataset(tok, longest, L)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        rec = {"max_length": L}
        try:
            cbs = make_callbacks(dt.datetime.max, OUT / "calib_progress.jsonl", start)
            trainer = Trainer(
                model=model,
                args=make_args(OUT / "calib", max_steps=3, gradient_accumulation_steps=1,
                               logging_steps=1, save_strategy="no"),
                train_dataset=ds,
                data_collator=collate,
                callbacks=[cbs[1]],
            )
            trainer.train()
            peak = torch.cuda.max_memory_reserved()
            rec.update(finished=True, peak_reserved_gb=round(peak / GIB, 3),
                       seconds=round(time.time() - start, 1))
            rec["fits"] = peak <= MEM_LIMIT and (time.time() - start) <= TRIAL_TIMEOUT_S
        except torch.cuda.OutOfMemoryError:
            rec.update(finished=False, fits=False, error="CUDA OOM")
        except Exception as e:  # noqa: BLE001
            rec.update(finished=False, fits=False, error=repr(e)[:300])
        trials.append(rec)
        log(f"calibration {rec}")
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        if rec.get("fits"):
            chosen = L
            break
    restore_lora(model, snap)
    (OUT / "calibration.json").write_text(json.dumps({"chosen_max_length": chosen, "trials": trials}, indent=2))
    if chosen is None:
        raise SystemExit("FAIL: no context length fits in 5.6 GiB reserved VRAM. Not training.")
    log(f"CHOSEN CONTEXT: {chosen}")
    return chosen, snap


# ---------------------------------------------------------------- main
def resolve_deadline():
    if os.environ.get("MAX_TRAIN_SECONDS"):
        return dt.datetime.now() + dt.timedelta(seconds=float(os.environ["MAX_TRAIN_SECONDS"]))
    hhmm = os.environ.get("TRAIN_DEADLINE", "06:15")
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    if d <= dt.datetime.now():
        raise SystemExit(f"TRAIN_DEADLINE {hhmm} is already past")
    return d


def main():
    import torch
    Trainer = chunked_trainer_class()

    set_seeds()
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_model()
    deadline = resolve_deadline()
    log(f"training deadline: {deadline}  smoke={SMOKE}")

    rows = load_jsonl(WORK / "train.jsonl", limit=20 if SMOKE else None)
    log(f"train examples: {len(rows)}")

    tok, model, bf16 = load_base()
    verify_masking(tok, EncodedDataset(tok, rows[:1], 4096))

    ckpt_dir = OUT / "checkpoints"
    resume = ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint-*"))
    calib_file = OUT / "calibration.json"
    if resume and calib_file.exists():
        max_len = json.loads(calib_file.read_text())["chosen_max_length"]
        log(f"resuming from checkpoint; using calibrated context {max_len}")
        snap = None
    else:
        max_len, snap = calibrate(tok, model, rows, bf16)
        set_seeds()
    if os.environ.get("CALIB_ONLY"):
        log(f"CALIB_ONLY: context {max_len} fits")
        return 0

    ds = EncodedDataset(tok, rows, max_len)
    log(f"encoded {len(ds)} examples at max_length {max_len}; truncated {ds.truncated}")
    verify_masking(tok, ds)
    if snap is not None:
        restore_lora(model, snap)

    progress = OUT / "train_progress.jsonl"
    cbs = make_callbacks(deadline, progress, time.time())
    args = make_args(
        ckpt_dir,
        num_train_epochs=3,
        gradient_accumulation_steps=8,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=make_collator(tok.pad_token_id), callbacks=[cbs[0]],
    )
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    elapsed = time.time() - t0
    adapter = OUT / "adapter"
    model.save_pretrained(str(adapter))
    summary = {
        "max_length": max_len,
        "examples": len(ds),
        "truncated_examples": ds.truncated,
        "global_step": trainer.state.global_step,
        "epochs_completed": trainer.state.epoch,
        "train_loss": result.training_loss,
        "train_seconds": round(elapsed, 1),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "stopped_by_deadline": dt.datetime.now() >= deadline,
        "adapter": str(adapter),
    }
    (OUT / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"adapter saved to {adapter}: {summary}")


if __name__ == "__main__":
    sys.exit(main())
