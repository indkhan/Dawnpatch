#!/usr/bin/env python3
"""Proxy evaluation: base (4-bit, no adapter) vs trained (base + LoRA) on held-out T3 examples.

Part A: assistant-token loss on ALL held-out examples, same examples and max_length for both.
        Reports mean loss, perplexity, and paired win rate (trained loss < base loss).
        A partial REPORT.md is written as soon as Part A finishes.
Part B: greedy generation (max_new_tokens 512) for as many held-out examples as time allows.
        Scores: parseable unified diff, edited-requested-file rate, difflib ratio of +/- lines.
Writes work/REPORT.md (work/smoke/REPORT.md in SMOKE mode).

Env:
  EVAL_DEADLINE=HH:MM    absolute local time; stop cleanly before every item after it
  MAX_EVAL_SECONDS=N     relative limit (smoke); overrides EVAL_DEADLINE
  SMOKE=1                5 examples, 2 generations, outputs in work/smoke/
"""
import datetime as dt
import difflib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # -I mode drops the script dir
import train as T  # shared encoding, paths, seeds

WORK = T.WORK
OUT = T.OUT
SMOKE = T.SMOKE
LOSS_SAMPLE = 5 if SMOKE else None
GEN_LIMIT = 2 if SMOKE else None
MAX_NEW = 512
PART_TAG = "[eval]"


def log(msg):
    print(f"{PART_TAG} [{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_deadline():
    if os.environ.get("MAX_EVAL_SECONDS"):
        return dt.datetime.now() + dt.timedelta(seconds=float(os.environ["MAX_EVAL_SECONDS"]))
    hhmm = os.environ.get("EVAL_DEADLINE", "07:40")
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    if d <= dt.datetime.now():
        raise SystemExit(f"EVAL_DEADLINE {hhmm} is already past")
    return d


def find_adapter():
    adapter = OUT / "adapter"
    if (adapter / "adapter_model.safetensors").exists():
        return adapter, "final adapter"
    ckpts = sorted((OUT / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts and (ckpts[-1] / "adapter_model.safetensors").exists():
        return ckpts[-1], f"latest checkpoint {ckpts[-1].name}"
    raise SystemExit("FAIL: no adapter or checkpoint found under " + str(OUT))


def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(str(T.MODEL_DIR))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(str(T.MODEL_DIR), quantization_config=bnb,
                                                dtype=dtype, device_map={"": 0})
    base.eval()
    adapter_dir, src = find_adapter()
    log(f"loading adapter from {adapter_dir} ({src})")
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    return tok, model, src


def assistant_nll(model, ids, labels):
    """Sum of NLL over assistant tokens and the token count (no grad)."""
    import torch
    import torch.nn.functional as F
    x = torch.tensor([ids], device="cuda:0")
    y = torch.tensor(labels[1:], device="cuda:0")
    # Memory fix (coordinator): never build full-vocabulary logits for the whole sequence
    # (130,560 vocab x 8k tokens would be ~4 GB). Project only labelled positions, in chunks.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.no_grad():
        hidden = base.model(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
        mask = y != -100
        h, t = hidden[mask], y[mask]
        w = base.lm_head.weight
        loss = 0.0
        for i in range(0, h.shape[0], 256):
            logits = F.linear(h[i:i + 256], w).float()
            loss += float(F.cross_entropy(logits, t[i:i + 256], reduction="sum").item())
    return loss, int(mask.sum().item())


def score_generation(text, real_diff, requested_file):
    """parseable: has ---/+++ or @@. edited_requested_file: 1 if the generated diff touches the
    requested file (basename match on its +++/--- headers, or the basename appears in the diff text)."""
    parseable = bool(re.search(r"^(?:---|\+\+\+) |^@@", text, re.M))
    base = (requested_file or "").replace("\\", "/").split("/")[-1].lower()
    gen_headers = set()
    for m in re.finditer(r"^(?:\+\+\+|---) (?:[ab]/)?(.+)$", text, re.M):
        name = m.group(1).strip()
        if name != "/dev/null":
            gen_headers.add(name.replace("\\", "/").split("/")[-1].lower())
    if base:
        edited = 1.0 if (base in gen_headers or base in text.lower()) else 0.0
    else:
        edited = None

    def changed(d):
        out = []
        for ln in d.splitlines():
            if ln.startswith(("+++", "---")):
                continue
            if ln.startswith(("+", "-")):
                out.append(ln)
        return "\n".join(out)

    ratio = difflib.SequenceMatcher(None, changed(real_diff), changed(text)).ratio()
    return {"parseable": parseable, "edited_requested_file": edited, "line_ratio": ratio}


def gen(tok, model, prompt):
    import torch
    msgs = [{"role": "system", "content": T.SYSTEM}, {"role": "user", "content": prompt}]
    p_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(p_text, add_special_tokens=False, return_tensors="pt").to("cuda:0")
    enc.pop("token_type_ids", None)  # Llama does not accept it in generate()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_report(state):
    summ = json.loads((OUT / "train_summary.json").read_text()) if (OUT / "train_summary.json").exists() else {}
    calib = json.loads((OUT / "calibration.json").read_text()) if (OUT / "calibration.json").exists() else {}
    stats = json.loads((WORK / "extract_stats.json").read_text()) if (WORK / "extract_stats.json").exists() else {}
    prog = []
    if (OUT / "train_progress.jsonl").exists():
        prog = [json.loads(l) for l in (OUT / "train_progress.jsonl").read_text().splitlines() if l.strip()]

    lines = ["# T3 -> MiniCPM5-2B QLoRA: proxy evaluation report", ""]
    lines += [f"Generated: {dt.datetime.now().isoformat(timespec='minutes')}"
              + (" (SMOKE run)" if SMOKE else ""), ""]
    lines += ["## Dataset", ""]
    if stats:
        lines += [f"- Eligible top-level threads: {stats.get('threads_eligible_top_level')}; excluded: {stats.get('threads_excluded')}",
                  f"- Threads with examples: {stats.get('threads_with_examples')} (train {stats.get('threads_train')}, heldout {stats.get('threads_heldout')})",
                  f"- Examples: train {stats.get('examples_train')}, heldout {stats.get('examples_heldout')}",
                  "- Drops (counts only):", ""]
        lines += [f"  - `{k}`: {v}" for k, v in stats.get("drops_and_counts", {}).items()]
    lines += ["", "## Context and training", ""]
    lines += [f"- Calibration trials: {calib.get('trials')}", f"- Chosen max_length: {calib.get('chosen_max_length')}"]
    if summ:
        lines += [f"- Examples trained: {summ.get('examples')} (truncated to context: {summ.get('truncated_examples')})",
                  f"- Steps: {summ.get('global_step')}; epochs completed: {summ.get('epochs_completed')}",
                  f"- Train loss (average): {fmt(summ.get('train_loss'))}",
                  f"- Train wall time (s): {summ.get('train_seconds')}; peak reserved GB: {summ.get('peak_reserved_gb')}",
                  f"- Stopped by deadline: {summ.get('stopped_by_deadline')}"]
    if prog:
        lines += ["", "Loss curve (about every 10% of logged steps):", "", "| step | epoch | loss | lr |", "|---|---|---|---|"]
        step_stride = max(1, len(prog) // 10)
        for i, r in enumerate(prog):
            if i % step_stride == 0 or i == len(prog) - 1:
                lines.append(f"| {r['step']} | {r['epoch']:.2f} | {r['loss']} | {r['lr']} |")

    lines += ["", "## Loss on held-out assistant tokens", ""]
    if state.get("loss"):
        L = state["loss"]
        lines += [f"n = {L['n']} held-out examples, max_length {L['max_length']}", "",
                  "| metric | base (4-bit) | trained | delta |", "|---|---|---|---|",
                  f"| mean per-example loss | {fmt(L['base_mean'])} | {fmt(L['trained_mean'])} | {fmt(L['trained_mean'] - L['base_mean'])} |",
                  f"| perplexity (exp of mean loss) | {fmt(L['base_ppl'], 3)} | {fmt(L['trained_ppl'], 3)} | |",
                  f"| paired win rate (trained < base) | | {fmt(L['win_rate'], 3)} | |", ""]
    else:
        lines += ["Not run (deadline reached before loss evaluation)."]

    lines += ["", "## Generation (greedy, max_new_tokens 512)", ""]
    g = state.get("gen") or []
    if g:
        def agg(key, side):
            vals = [r[side][key] for r in g if r[side].get(key) is not None]
            return mean([float(v) for v in vals]) if vals else None
        lines += [f"n = {len(g)} held-out examples", "",
                  "| metric | base | trained |", "|---|---|---|",
                  f"| parseable unified diff (fraction) | {fmt(agg('parseable', 'base'), 3)} | {fmt(agg('parseable', 'trained'), 3)} |",
                  f"| edited the requested file (fraction) | {fmt(agg('edited_requested_file', 'base'), 3)} | {fmt(agg('edited_requested_file', 'trained'), 3)} |",
                  f"| +/- line difflib ratio (mean) | {fmt(agg('line_ratio', 'base'), 3)} | {fmt(agg('line_ratio', 'trained'), 3)} |", ""]
        lines += ["### Samples (first 5)", ""]
        for r in g[:5]:
            lines += [f"**Prompt:** {r['prompt'][:1500]}", "",
                      f"**Real diff:**", "```diff", r["real"][:1500], "```",
                      "**Base:**", "```diff", r["base_text"][:1500], "```",
                      "**Trained:**", "```diff", r["trained_text"][:1500], "```", ""]
    else:
        lines += ["Not run (time or deadline)."]

    lines += ["", "## Verdict", "", state.get("verdict", "Pending.")]
    lines += ["", "_Proxy test only: held-out loss and a difflib/parse score are not the full replay + test gate in plan.txt._", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT / 'REPORT.md'}")


def verdict_from(L):
    if not L:
        return "No loss evaluation completed, so no verdict."
    wr = L["win_rate"]
    delta = L["trained_mean"] - L["base_mean"]
    if wr >= 0.6 and delta < 0:
        v = "BETTER (proxy): trained has lower held-out loss on most examples."
    elif wr <= 0.4 and delta > 0:
        v = "WORSE (proxy): trained has higher held-out loss on most examples."
    else:
        v = "NO MEASURABLE CHANGE (proxy): win rate and mean loss delta do not clearly favor either side."
    return v + f" (win rate {wr:.3f}, mean delta {delta:+.4f}, n={L['n']})."


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = resolve_deadline()
    T.set_seeds()
    log(f"eval deadline {deadline}, smoke={SMOKE}")

    rows = T.load_jsonl(WORK / "heldout.jsonl")
    rng = random.Random(42)
    rng.shuffle(rows)
    if LOSS_SAMPLE:
        rows = rows[:LOSS_SAMPLE]
    max_len = json.loads((OUT / "calibration.json").read_text())["chosen_max_length"]

    tok, model, src = load_model()
    state = {"loss": None, "gen": [], "verdict": None}

    # Part A: loss on all held-out examples, base and trained, same encoding.
    base_losses, trained_losses, ns = [], [], []
    for r in rows:
        if dt.datetime.now() >= deadline:
            log("deadline reached during loss evaluation: stopping")
            break
        ids, labels = T.encode(tok, r["prompt"], r["diff"], max_len)
        with model.disable_adapter():
            bl, bn = assistant_nll(model, ids, labels)
        tl, tn = assistant_nll(model, ids, labels)
        if bn == 0:
            continue
        base_losses.append(bl / bn)
        trained_losses.append(tl / tn)
        ns.append(bn)
    if base_losses:
        wins = sum(1 for b, t in zip(base_losses, trained_losses) if t < b) / len(base_losses)
        state["loss"] = {
            "n": len(base_losses), "max_length": max_len,
            "base_mean": mean(base_losses), "trained_mean": mean(trained_losses),
            "base_ppl": math.exp(mean(base_losses)), "trained_ppl": math.exp(mean(trained_losses)),
            "win_rate": wins,
        }
        log(f"loss: {state['loss']}")
    state["verdict"] = verdict_from(state["loss"])
    write_report(state)

    # Part B: generation, time permitting.
    if state["loss"]:
        gen_rows = rows[:GEN_LIMIT] if GEN_LIMIT else rows
        for r in gen_rows:
            if dt.datetime.now() >= deadline:
                log("deadline reached during generation: stopping")
                break
            with model.disable_adapter():
                base_text = gen(tok, model, r["prompt"])
            trained_text = gen(tok, model, r["prompt"])
            entry = {
                "prompt": r["prompt"], "real": r["diff"],
                "base_text": base_text, "trained_text": trained_text,
                "base": score_generation(base_text, r["diff"], r.get("file")),
                "trained": score_generation(trained_text, r["diff"], r.get("file")),
            }
            state["gen"].append(entry)
            if len(state["gen"]) % 5 == 0:
                write_report(state)
        write_report(state)
    log(f"done; generations scored: {len(state['gen'])}")


if __name__ == "__main__":
    sys.exit(main())
