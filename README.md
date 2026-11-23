# Dawnpatch

Trains your own small open-weights coding model every night from your T3 Code, Claude Code and Codex history, and only uses the new version if it beats the old one on tasks it never saw. Everything runs on your machine. Use it from a small desktop app, or entirely from the terminal with no app installed.

Status: planning. Nothing is implemented yet.

## Two ways to use it (planned)

- **Desktop app:** install it, pick a model, and it runs at night from the tray.
- **Terminal only:** clone this repo and run the CLI from source. The app and the CLI share the same core and settings, and every app action has a CLI command.

  ```sh
  git clone <this repo> && cd Dawnpatch
  cargo run -p dawnpatch-cli -- init          # find history, GPU; pick a model
  cargo run -p dawnpatch-cli -- model search coder
  cargo run -p dawnpatch-cli -- config set schedule.idle_minutes 15
  cargo run -p dawnpatch-cli -- sleep         # run a night now
  cargo run -p dawnpatch-cli -- watch         # live loss, GPU, stage
  cargo run -p dawnpatch-cli -- daemon install  # schedule nightly runs without the app
  ```

## Where things are

| Path | What it is |
|---|---|
| [`plan.txt`](plan.txt) | The technical plan: data rules, episode format, training stages, evaluation gate, desktop app, epics (E0–E12), open questions, decision log |
| [`ui/DESIGN.md`](ui/DESIGN.md) | UI spec: design tokens (dark + light), components, the nine screens and the setting each control maps to, behaviour rules |
| [`ui/mockup/index.html`](ui/mockup/index.html) | Clickable mockup of the app. Open it in a browser; keys `1`–`8` switch screens (`,` opens Settings), the moon button toggles the theme. Data is fake |

## Stack (planned)

- **Core + CLI:** Rust crates; the CLI builds without Tauri, so it runs on servers and over SSH.
- **App:** Tauri 2 (Rust) with a Svelte UI and uPlot charts. v1 targets Windows and Linux; macOS comes later. Settings live in an internal SQLite store, so there is no config file.
- **Trainer:** a Python child process (Unsloth / TRL / PEFT; mlx-lm on Apple Silicon) that the app starts, pauses and stops. It streams progress as JSON lines.
- **Hardware:** anyone can run it: CPU only, a 6 GB laptop (small models), 24–80 GB GPUs, multi-GPU boxes and DGX Spark. It is designed mainly for big machines. Every tier gets memory (rules from corrections, checks, past fixes); training runs every night where the hardware allows, and you choose how often a new version may go live (nightly, weekly, monthly, or when it wins by X points). The promotion gate always applies.
- **Pilot experiments:** [`experiments/`](experiments/) has two 2B runs on the 6 GB laptop; their lessons are in `plan.txt` under "Pilot results".
- **First-run model on the 6 GB pilot laptop:** `Qwen/Qwen3.5-2B` via Unsloth. Nanbeige4.2-3B was tested and is too big for 6 GB (see [`spikes/model-compat/RESULTS.md`](spikes/model-compat/RESULTS.md)). Users can pick any Hugging Face or local model; there is no default.
- **Sources:** T3 Code, Claude Code and Codex, opened read-only. Agent chat replies, thinking and subagent calls are dropped at import and never stored.

## Build order (start here)

Follow the epics in `plan.txt` in this order. Each has deliverables and acceptance criteria.

1. **E0 Foundations:** Rust workspace (`core`, `cli`, `app` crates), Python trainer package, SQLite settings store, CI on Windows and Linux.
2. **E1 T3 connector + episode store:** read-only import, authorship evidence, hunk-level provenance, scrub, audit.
3. **E2 Evaluation harness:** checkpoint restore, Docker-in-WSL2 sandbox with offline dependency preparation, qualified verifiers, train/tune/promotion cohorts, clustered bootstrap. Build this before any training.
4. **E3 Scan + rules + correction scorer.**
5. **E4 Trainer:** compatibility check (first task: Qwen3.5-2B context sweep on the 6 GB laptop), FIM + SFT, JSON-line progress, cooperative pause.
6. **E5 Gate + versions + endpoint**, **E9 Claude Code / Codex connectors**, **E6 scheduler + report**, **E12 desktop app**. These complete milestone M2.
7. **M3** (preference training, practice gym) and **M4** (live learning, release).

## Contributing

Read `plan.txt` first. The data rules, the held-out split and the promotion gate's guardrails are fixed. Everything else (models, methods, thresholds, module boundaries) can change if it scores better on the evaluation harness. Record any such change in the plan's decision log.

For UI work, follow `ui/DESIGN.md` and keep the mockup in step with it.
