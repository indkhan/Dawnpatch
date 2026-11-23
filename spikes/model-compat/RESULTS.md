# Spike: Nanbeige4.2-3B QLoRA on a 6 GB laptop GPU

Date: 2026-11-21/22. Machine: Windows 11, RTX 3050 Laptop 6 GB (driver 581.86), 15.7 GB RAM.
Goal: prove the plan's "custom-code fallback" path (plain transformers + TRL + PEFT + bitsandbytes 4-bit, `trust_remote_code`, pinned commit) on the pilot laptop.

**Verdict:** the fallback path works, since the model loads, trains and resumes. But this model is too big for 6 GB: memory is maxed at 1024 tokens of context, and 2048 stalled. Use a smaller model (~2B) for the first run on this laptop.

## Setup

| Item | Value |
|---|---|
| Model | `Nanbeige/Nanbeige4.2-3B` @ `b82e54bd609793562a75cbf9337970a93369eab5` (custom `nanbeige` architecture, 4.19B params) |
| Python | 3.11.9 (uv venv in `.venv/`) |
| Packages | torch 2.11.0+cu128, transformers 4.57.6, peft 0.22.0, trl 1.18.2, bitsandbytes 0.50.2, accelerate 1.16.0, datasets 5.2.0 (`requirements.txt`) |
| Quantization | bnb NF4, double quant, fp16/bf16 compute |
| LoRA | r=16, alpha=32, targets found automatically: q/k/v/o/gate/up/down_proj; 23.97M trainable (0.57%) |
| Data | 20 synthetic code-edit chat examples, batch 1, grad accumulation 2, paged AdamW 8-bit, gradient checkpointing |
| Custom code review | `modeling_nanbeige.py` / `configuration_nanbeige.py` contain no network, subprocess, shell or file-write calls |

Scripts: `spike.py` (train / resume / generate), `run_all.sh` (download → train → resume → generate → context sweep). Raw log: `outputs_log.txt` (git-ignored).

## Results

| Check | Result |
|---|---|
| Load in 4-bit | ✅ 3.43 GB allocated after load |
| Train 3 steps, 1024 tokens | ✅ loss 5.97 → 4.06 → 3.32; about 3.1 s/step after warm-up (first step ~100 s including compile and caching) |
| Peak memory at 1024 | ⚠️ 5.90 GB allocated, 6.61 GB reserved; nvidia-smi 5.4 GB. Reserved is above the 6 GB card, so the Windows driver was spilling into shared system memory |
| Resume from checkpoint | ✅ continued from step 3 to step 5 (loss 3.04, 3.08); peak 6.03 GB allocated / 7.70 GB reserved |
| Generate with adapter | ❌ `AttributeError: 'DynamicCache' object has no attribute 'get_max_length'`. The model's code was written for an older transformers cache API |
| 2048 tokens | ❌ stalled at step 0 for over 3 minutes with the GPU full (spilling to shared memory); stopped by hand |
| 4096+ tokens | not attempted |

## Problems found and what the plan should do

1. **Model too big for the card.** 4.19B params plus a 166k vocabulary leaves no room for activations on 6 GB. → The first-run model on the pilot laptop should be about 2B. The compatibility check must treat "reserved > physical VRAM" as *does not fit*, because Windows silently spills to shared memory and steps become 30–100× slower instead of failing.
2. **Custom code is tied to a transformers version.** Training worked on 4.57.6; generation failed because the model's code calls an API that transformers has removed. PEFT also warns that transformers < 5.6 support is ending. → Each custom-code model needs its own pinned Python environment (transformers version chosen per model). The compatibility check must include generation, which the review already added.
3. **Hugging Face cache symlinks on Windows.** `snapshot_download` failed with `WinError 1314`, because creating symlinks needs Developer Mode or admin rights. → The downloader must not rely on cache symlinks on Windows: download with `local_dir` (plain files) or `allow_patterns` limited to weights, config, tokenizer and code.
4. **Load time dominates short runs.** Each process start (load + quantize + dataset prep) took 2–3 minutes for a few seconds of training. → Keep the trainer process alive across stages within a night, and cache tokenized datasets.
5. **Optimizer note:** `paged_adamw_8bit` + gradient checkpointing worked through the PEFT fallback with no Unsloth.

## Not done

The Qwen3.5-2B follow-up (Unsloth, 16-bit and 4-bit, context sweep up to 64k) was not run. The owner chose to finish the docs and hand off implementation instead. It is the first task in the plan's open questions.
