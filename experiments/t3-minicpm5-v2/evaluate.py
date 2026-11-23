#!/usr/bin/env python3
"""Held-out evaluation: does the edit apply, and does it reproduce the kept file?

Contestants (same held-out examples, same prompt, greedy, thinking off, same token cap):
  base + sr instructions, base + udiff instructions, sr adapter, udiff adapter

Per generation:
  applies   the edit applies to the start-state file (sr: every SEARCH found; udiff: every hunk placed)
  exact     the edited file equals the file at the end checkpoint
  change    similarity of the changed lines (edited vs real), 0 if it does not apply
  runaway   hit the token cap without stopping (loops)

Differences adapter - base are reported with a 95% bootstrap CI, resampling whole threads.
No tests are run (no sandbox yet), so this is not the plan's pass@1 gate.
Env: MAX_EVAL_SECONDS (default 3600), SMOKE=1 (3 examples)
"""
import datetime as dt
import difflib
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "t3-minicpm5"))
os.environ.setdefault("FORMAT", "sr")
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2train", HERE / "train.py")
V2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V2)
T, E = V2.T, V2.E
WORK = HERE / "work"
SMOKE = os.environ.get("SMOKE") == "1"
MAX_EVAL_SECONDS = float(os.environ.get("MAX_EVAL_SECONDS", "3600"))
CAP = 1024


def log(m):
    print(f"[eval {dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def changed_lines(before, after):
    return "\n".join(l for l in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
                     if l[:1] in "+-" and not l.startswith(("+++", "---")))


def score(fmt, row, text, hit_cap):
    new = E.FORMATS[fmt]["apply"](row["before"], text)
    applies = new is not None
    exact = applies and new.rstrip() == row["after"].rstrip()
    change = 0.0
    if applies:
        change = difflib.SequenceMatcher(None, changed_lines(row["before"], row["after"]),
                                         changed_lines(row["before"], new), autojunk=False).ratio()
        if new == row["before"]:
            change = 0.0
    return {"applies": float(applies), "exact": float(exact), "change": change, "runaway": float(hit_cap)}


def generate(tok, model, fmt, row, cap):
    import torch
    enc = tok(V2.prompt_text(tok, fmt, row["prompt"]), add_special_tokens=False, return_tensors="pt").to("cuda:0")
    enc.pop("token_type_ids", None)
    end_id = tok.convert_tokens_to_ids(V2.END)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cap, do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=[end_id, tok.eos_token_id])
    new = out[0, enc["input_ids"].shape[1]:]
    hit_cap = len(new) >= cap and int(new[-1]) not in (end_id, tok.eos_token_id)
    return tok.decode(new, skip_special_tokens=True), hit_cap


def boot_ci(results, a, b, key, n=4000, seed=0):
    """Mean of (a - b) on `key` with a 95% CI, resampling threads."""
    by_t = {}
    for r in results:
        by_t.setdefault(r["thread_id"], []).append(r[a][key] - r[b][key])
    threads = list(by_t)
    allv = [v for t in threads for v in by_t[t]]
    point = sum(allv) / len(allv)
    rng, means = random.Random(seed), []
    for _ in range(n):
        vs = [v for t in (rng.choice(threads) for _ in threads) for v in by_t[t]]
        means.append(sum(vs) / len(vs))
    means.sort()
    return point, means[int(0.025 * n)], means[int(0.975 * n)]


CONTESTANTS = [("base_sr", "sr", None), ("base_udiff", "udiff", None), ("sr", "sr", "sr"), ("udiff", "udiff", "udiff")]


def main():
    import torch
    from peft import PeftModel
    T.set_seeds()
    rows = T.load_jsonl(WORK / "heldout.jsonl")
    if SMOKE:
        rows = rows[:3]
    tok, _, _ = None, None, None
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(str(T.MODEL_DIR))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(T.MODEL_DIR), dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = PeftModel.from_pretrained(base, str(WORK / "sr" / "adapter"), adapter_name="sr", is_trainable=False)
    model.load_adapter(str(WORK / "udiff" / "adapter"), adapter_name="udiff", is_trainable=False)
    model.eval()

    deadline = time.time() + MAX_EVAL_SECONDS
    results = []
    for i, row in enumerate(rows):
        if time.time() > deadline:
            log("time limit reached; stopping")
            break
        tgt_tokens = max(len(tok(row[f"target_{f}"])["input_ids"]) for f in ("sr", "udiff"))
        cap = min(CAP, 2 * tgt_tokens + 64)  # same cap for every contestant on this example
        rec = {"thread_id": row["thread_id"], "file": row["file"], "request": row["request"], "cap": cap, "texts": {}}
        for name, fmt, adapter in CONTESTANTS:
            if adapter is None:
                with model.disable_adapter():
                    text, hit = generate(tok, model, fmt, row, cap)
            else:
                model.set_adapter(adapter)
                text, hit = generate(tok, model, fmt, row, cap)
            rec[name] = score(fmt, row, text, hit)
            rec["texts"][name] = text
        rec["real_sr"] = row["target_sr"]
        results.append(rec)
        log(f"{i + 1}/{len(rows)} " + " ".join(f"{n}:a{rec[n]['applies']:.0f}e{rec[n]['exact']:.0f}c{rec[n]['change']:.2f}"
                                               for n, _, _ in CONTESTANTS))
        (WORK / "eval_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    report(results)


def report(results):
    stats = json.loads((WORK / "extract_stats.json").read_text())
    summ = {f: json.loads((WORK / f / "train_summary.json").read_text()) for f in ("sr", "udiff")}
    n = len(results)
    L = ["# v2: MiniCPM5-2B, request + file -> edit (search/replace vs unified diff)", "",
         f"Generated {dt.datetime.now().isoformat(timespec='minutes')}. Held-out: {n} edits from "
         f"{len({r['thread_id'] for r in results})} threads (newest 20% of threads per project).", "",
         "## Data", "",
         f"- Train {stats['examples_train']} / held-out {stats['examples_heldout']} examples from "
         f"{stats['threads_with_examples']} threads, {stats['projects_with_examples']} projects.",
         "- Input: developer's request + the file at the T3 checkpoint before the turn. Target: the change to that "
         "file at the last checkpoint before the developer's next message, rebuilt from git and checked by round trip.",
         "- Drops: " + ", ".join(f"{k.replace('drop_', '')} {v}" for k, v in stats["drops_and_counts"].items() if k.startswith("drop_")),
         "", "## Training", "", "| arm | examples | epochs | steps | train loss | minutes | peak GB |", "|---|---|---|---|---|---|---|"]
    for f, s in summ.items():
        L.append(f"| {f} | {s['examples']} | {s['epochs_completed']:.2f} | {s['global_step']} | {s['train_loss']:.3f} | "
                 f"{s['train_seconds'] / 60:.0f} | {s['peak_reserved_gb']} |")
    L += ["", "## Results (held-out)", "", "| contestant | applies | exact match | change similarity | runaway |", "|---|---|---|---|---|"]
    for name, _, _ in CONTESTANTS:
        m = {k: sum(r[name][k] for r in results) / n for k in ("applies", "exact", "change", "runaway")}
        L.append(f"| {name} | {m['applies']:.2f} | {m['exact']:.2f} | {m['change']:.2f} | {m['runaway']:.2f} |")
    L += ["", "Adapter minus base (95% CI, threads resampled):", "", "| comparison | metric | delta | CI |", "|---|---|---|---|"]
    for a, b in (("sr", "base_sr"), ("udiff", "base_udiff"), ("sr", "udiff")):
        for k in ("applies", "exact", "change"):
            p, lo, hi = boot_ci(results, a, b, k)
            L.append(f"| {a} vs {b} | {k} | {p:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    L += ["", "## Samples (first 3)", ""]
    for r in results[:3]:
        L += [f"**Request:** {r['request'][:400]}", f"**File:** `{r['file']}`", "", "Real change:", "```", r["real_sr"][:1200], "```"]
        for name, _, _ in CONTESTANTS:
            L += [f"{name} ({'applies' if r[name]['applies'] else 'does not apply'}"
                  f"{', exact' if r[name]['exact'] else ''}):", "```", r["texts"][name][:1200], "```"]
        L.append("")
    L += ["## Limits", "",
          "- The touched file is given to the model (oracle file choice), not found by the plan's retrieval.",
          "- No tests are run; 'exact' is a strict stand-in for 'passes'. Different correct edits score as misses.",
          "- No rules in the prompt (no rule extraction yet), so 'base' here is base + format instructions only.",
          f"- Small held-out set ({n} edits): read the CIs, not the point estimates.", ""]
    (WORK / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {WORK / 'REPORT.md'}")


if __name__ == "__main__":
    sys.exit(main())
