"""Dawnpatch model-compat spike: QLoRA on a custom-architecture model with plain transformers+TRL+PEFT+bnb.
Usage: python spike.py train --seq 1024 --steps 3 [--resume] [--bits 4|8|16] [--model ID --rev SHA]
       python spike.py generate --out outputs/run1
"""
import argparse, json, os, subprocess, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

NB = ("Nanbeige/Nanbeige4.2-3B", "b82e54bd609793562a75cbf9337970a93369eab5")

def smi_used_mb():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return -1

def load(a):
    kw = dict(trust_remote_code=True, revision=a.rev, device_map={"": 0})
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if a.bits == 4:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
    elif a.bits == 8:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    kw["dtype"] = dt
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    return tok, model

def lora_targets(model):
    import torch.nn as nn, bitsandbytes as bnb
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            leaf = n.split(".")[-1]
            if leaf != "lm_head" and "embed" not in n:
                names.add(leaf)
    return sorted(names)

EDITS = [("rename variable `x` to `count`", "def f(x):\n    x += 1\n    return x", "def f(count):\n    count += 1\n    return count"),
 ("add type hints", "def add(a, b):\n    return a + b", "def add(a: int, b: int) -> int:\n    return a + b"),
 ("replace print with logging", "print('start')", "import logging\nlogging.info('start')"),
 ("make the loop a list comprehension", "r = []\nfor i in range(5):\n    r.append(i*i)", "r = [i*i for i in range(5)]"),
 ("add a docstring", "def area(r):\n    return 3.14*r*r", 'def area(r):\n    """Return circle area."""\n    return 3.14*r*r')]
FILLER = "def helper_{i}(a, b):\n    # utility function number {i}\n    return (a * {i}) + b\n\n"

def build_dataset(tok, n, seq):
    from datasets import Dataset
    rows = []
    for i in range(n):
        t, src, dst = EDITS[i % len(EDITS)]
        ctx = ""
        # pad file context with filler code until roughly seq tokens (leave room for answer)
        if seq > 1024 or os.environ.get("FILL"):
            j = 0
            while len(tok(ctx)["input_ids"]) < seq - 200:
                ctx += "".join(FILLER.format(i=j + k) for k in range(40)); j += 40
        user = f"{ctx}\n# File above is context.\nTask: {t}\n```python\n{src}\n```"
        rows.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": f"```python\n{dst}\n```"}]})
    return Dataset.from_list(rows)

class Log(TrainerCallback):
    def __init__(self): self.t0 = time.time(); self.last = self.t0
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            now = time.time()
            print(json.dumps({"step": state.global_step, "loss": logs["loss"], "lr": logs.get("learning_rate"),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "s_per_step": round((now - self.last) / max(1, args.logging_steps), 2)}), flush=True)
            self.last = now

def train(a):
    from trl import SFTConfig, SFTTrainer
    tok, model = load(a)
    print("loaded; vram alloc GB", round(torch.cuda.memory_allocated() / 2**30, 2), flush=True)
    if a.bits in (4, 8):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    tg = lora_targets(model); print("lora targets:", tg, flush=True)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=tg, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg); model.print_trainable_parameters()
    ds = build_dataset(tok, 20, a.seq)
    bf = torch.cuda.is_bf16_supported()
    args = SFTConfig(output_dir=a.out, max_steps=a.steps, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        learning_rate=2e-4, logging_steps=1, save_steps=a.save_steps, save_total_limit=2, bf16=bf, fp16=not bf,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if a.bits < 16 else "adamw_8bit", max_length=a.seq, dataset_num_proc=1,
        report_to=[], dataloader_num_workers=0, packing=False)
    tr = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, callbacks=[Log()])
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    tr.train(resume_from_checkpoint=a.resume or None)
    el = time.time() - t
    tr.save_model(a.out + "/adapter")
    print(json.dumps({"done": True, "seq": a.seq, "steps": a.steps, "total_s": round(el, 1),
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3), "smi_used_mb": smi_used_mb(),
        "global_step": tr.state.global_step}), flush=True)

def generate(a):
    tok, model = load(a)
    model = PeftModel.from_pretrained(model, a.out + "/adapter"); model.eval()
    msgs = [{"role": "user", "content": "Task: add type hints\n```python\ndef mul(a, b):\n    return a * b\n```"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=32, do_sample=False)
    print("GENERATION:", repr(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "generate"])
    p.add_argument("--model", default=NB[0]); p.add_argument("--rev", default=NB[1])
    p.add_argument("--bits", type=int, default=4); p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--steps", type=int, default=3); p.add_argument("--save-steps", type=int, default=3)
    p.add_argument("--resume", action="store_true"); p.add_argument("--out", default="outputs/run1")
    a = p.parse_args()
    if a.rev == "none": a.rev = None
    train(a) if a.cmd == "train" else generate(a)
