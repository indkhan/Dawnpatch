# Dawnpatch UI design

The desktop app is the easiest way to configure Dawnpatch: no settings file to edit. It is optional: the CLI offers every action and setting from the terminal (`dawnpatch config set <key> <value>` uses the same keys as the table below), so any control added here needs a CLI command too. This document is the source of truth for how it looks and behaves. The clickable reference is [`mockup/index.html`](mockup/index.html). Open it in any browser; keys `1`–`8` switch screens and the moon button toggles the theme. Its data is fake.

Visual language is adapted from Linear's design system: near-black canvas, one lavender accent, hairline borders, monospace numbers, no shadows or gradients except the faint area fill under the loss line.

## Principles

1. **Light.** The machine is training a model. The UI must cost almost nothing: Svelte, no component library, one small chart library (uPlot) or hand-drawn canvas. No animation loops while the window is hidden.
2. **For power users.** Dense, keyboard-first, real numbers with units, raw logs one click away. Every setting is visible; advanced ones sit in a drawer, never hidden behind a wizard.
3. **Honest.** Show what is being trained on, why a candidate was rejected, and what is locked. Never show a number without its source (CI, task count).
4. **One accent.** Lavender `--primary` only for: the brand mark, the primary button, focus rings, the live/training state, chart series of the served model. Status colours (ok/warn/bad) only inside pills and deltas.

## Tokens

Defined as CSS custom properties on `:root`. Dark is the default; light applies when the OS prefers light (and the user chose *System*) or when `data-theme="light"` is set on `<html>`. Code must read colours from tokens (charts included: `getComputedStyle(root).getPropertyValue('--primary')`), never hard-code hex.

| Token | Dark | Light | Use |
|---|---|---|---|
| `--canvas` | `#010102` | `#fbfbfc` | Window background |
| `--s1` | `#0f1011` | `#ffffff` | Cards, inputs |
| `--s2` | `#141516` | `#f5f6f6` | Hover, selected row, pills |
| `--s3` | `#18191a` | `#eef0f1` | Active nav item, segmented-control selection, empty bar track |
| `--s4` | `#191a1b` | `#e3e5e8` | Switch track (off) |
| `--line` | `#23252a` | `#e6e7ea` | 1px borders, dividers, chart gridlines |
| `--line-strong` | `#34343a` | `#d3d5da` | Hover / focus borders |
| `--line-3` | `#3e3e44` | `#c4c7cd` | Neutral chart series (e.g. replay) |
| `--ink` | `#f7f8f8` | `#0d0e10` | Primary text |
| `--ink-muted` | `#d0d6e0` | `#3c4049` | Values, emphasised secondary text |
| `--ink-subtle` | `#8a8f98` | `#6b7079` | Labels, secondary text, comparison series |
| `--ink-tertiary` | `#62666d` | `#9a9ea6` | Captions, axis labels, disabled |
| `--primary` | `#5e6ad2` | `#5e6ad2` | Accent (see principle 4) |
| `--primary-hover` | `#828fff` | `#4c58c4` | Primary hover, latest data point |
| `--primary-focus` | `#5e69d1` | `#5e69d1` | Focus ring (2px, 50% alpha) |
| `--ok` | `#27a644` | `#1f8a39` | Promoted, positive delta, clean audit |
| `--warn` | `#d9a03f` | `#b07a12` | Waiting, suggested |
| `--bad` | `#e5484d` | `#d1343a` | Rejected, negative delta, destructive |

**Type.** `--sans`: Inter (bundle it; fallback Segoe UI / system-ui). `--mono`: JetBrains Mono (bundle; fallback Cascadia Code / Consolas). Base size 13px. Page title 20px/600/-0.4px tracking. Card title 13px/500. Captions 11–12px. All numbers, ids, paths, model names and logs use `--mono` with tabular figures.

**Spacing** on a 4px grid: 4, 8, 12, 16, 24. Card padding 14–16px. Gaps between cards 12px.

**Radius.** 4px tags · 6px nav items, small controls · 8px buttons, inputs · 12px cards and window · 999px pills, switches.

**Depth.** No drop shadows. Hierarchy comes from the surface ladder (`canvas → s1 → s2 → s3`) and borders. Dark cards get a 1px inset top highlight (`--hi`).

## Layout

```
┌ titlebar 36px: mark · name · state pill · ⌘K search · theme · window buttons ┐
├ sidebar 196px ┬ main (scrolls) ──────────────────────────────────────────────┤
│ Run           │ page header: title, one-line explanation, actions (right)    │
│ Configure     │ cards in a 12px grid (4 / 3 / 2 / 2:1 columns)               │
│ Results       │                                                              │
│ GPU/VRAM meter│                                                              │
├ statusbar 26px (mono): state · step · loss · GPU · VRAM · temp · idle time ─┤
```

- Custom titlebar (Tauri `decorations: false`) with drag region.
- Below 880px window width the sidebar collapses to 52px icons and grids drop to 2 or 1 columns.
- Minimum window size 720×560.

## Components

| Component | Spec |
|---|---|
| Button | 28px high, 8px radius, 12.5px/500. Variants: default (`s1` + border), `primary` (accent fill, white text), `ghost` (no border), `danger` (red text). |
| Card | `s1`, 1px `line`, 12px radius. Optional header row (title, sub-caption, right-aligned actions) separated by a hairline. |
| KPI | Label 11.5px subtle · value 20px mono 600 · caption 11.5px tertiary. Positive/negative deltas use `--ok` / `--bad`. |
| Pill | 20px, `s2` + border, 6px dot. States: `run` (accent), `ok`, `warn`, `bad`, neutral. |
| Switch | 28×16, accent when on. Every boolean setting is a switch, never a checkbox. |
| Segmented control | For 2–5 exclusive options (source, method, theme, rebuild cadence). |
| Number input | 72px, right-aligned mono, unit label after it. |
| Settings row | Label + one-line explanation left, control right, hairline between rows. |
| Locked row | Same as settings row with a lock icon + "Locked" instead of a control. Used for data that can never be trained on. |
| Stepper | 8 segments (Import, Scrub & audit, Memory, Datasets, Practice, Train, Evaluate, Promote), wrapping to two rows of 4 below 880 px; done = grey, current = accent fill showing progress, todo = track. |
| Table | 11.5px tertiary headers, 9px row padding, hover `s2`, numbers right-aligned. |
| Log | Mono 11.5px, timestamp column tertiary, auto-scroll unless the user scrolled up. |
| Charts | Line 1.5px accent for the served/training series, 1.2px dashed `ink-subtle` for comparison (eval loss, baseline). Gridlines `line`, axis labels 10px mono tertiary. No legends inside the plot: legend sits in the card header. |

## Screens

Each control maps to a key in the app's settings store (SQLite, see `plan.txt` → Desktop app). Keys are a starting point.

| # | Screen | Shows | Controls → setting key |
|---|---|---|---|
| 1 | **Overview** | Serving version, pass@1 Δ vs base + memory with its interval, episode count, next window; tonight's stepper; mini loss chart; last 7 nights | Pause / Resume (`runtime.paused`), Open run |
| 2 | **Training** | Stepper; live train/eval loss; run stats (step, epoch, LR, batch, context, tok/s, ETA); data mix; GPU/VRAM/temp sparklines; episodes in the current batch; log | Pause, Stop, Run now |
| 3 | **Model** | Base model source and download progress; fit and compatibility badges; the picked model's quantized variants; compatibility-check result; method; stages; hyperparameters | `model.source` (hf \| local), `model.id` or `model.path`, `model.revision` (pinned commit), `model.variant` (repo of the chosen quantized variant), `train.precision` (4bit \| 8bit \| 16bit), `model.trust_remote_code` (asked once per pinned commit), `train.enabled` (on by default on GPU tiers, off on CPU-only, where nights are memory-only), `train.method` (auto \| lora \| qlora \| dora \| full), `train.stages[1..4]` (1 FIM off by default; 4 runs GRPO only on 80 GB+), `train.lora_rank`, `train.lora_alpha`, `train.lr`, `train.epochs_per_night`, `train.batch`, `train.context`, `train.replay_ratio`, `train.grad_clip` |
| 4 | **Data** | Sources with episode counts; projects with episodes, corrections, replayable tasks, verifier; what is used; what is locked | `sources.t3`, `sources.claude_code`, `sources.codex`, `projects.<id>.enabled`, `data.agent_diffs` (on by default; the row states that the user needs their providers' permission), `data.fim`. Locked, no key: agent chat, thinking, subagent calls, secrets |
| 5 | **Schedule** | Week grid (allowed hours, hours it ran) | Start: `schedule.idle_minutes`, `schedule.window_start`, `schedule.window_end`, `schedule.require_ac`, `schedule.require_gpu_free`, `schedule.require_no_agent`. Stop: `stop.on_return`, `stop.at_time`, `stop.gpu_temp_c`, `stop.ram_percent`, `stop.max_hours`. Test: `eval.after_each_run`, `eval.min_heldout` (can be raised, never below the gate's 50), `eval.general_benchmark`, `eval.rebuild` (weekly \| never), `eval.time_budget_percent`. Use a new version: `promotion.cadence` (nightly \| weekly \| monthly \| margin; training still runs every night), `promotion.margin_points` (for margin: the lower bound of the pass@1 gain over the served version). The app suggests weekly while the promotion split is small, because nightly checks need a stricter level. These only control when and how much testing runs; they can never promote a candidate without the gate's evidence. A candidate whose evaluation was skipped, cut short or underpowered stays *Pending* with the missing evidence listed |
| 6 | **Evaluation** | Pass@1 by version vs baseline; contestants table; promotion-gate checklist with values | Run eval |
| 7 | **Versions** | Stable endpoint URL + model name; version list with status | Use, Undo, Export (Ollama, LM Studio, llama.cpp, vLLM, mlx-lm) |
| 8 | **Memory** | Learned rules with evidence count, scope and their check (command or judge rubric); size of the past-fixes index (train split only) | Approve, Edit, Remove |
| , | **Settings** | Hardware, data folder | `ui.theme` (system \| dark \| light), `app.launch_at_login`, `app.morning_notification`, `storage.encrypt`, `remote_gpu.enabled`, export/import settings, Forget |

**Model screen rules.** There is no default model: on first run the Model screen is empty apart from search, local-folder picker and a few suggestions for the detected GPU, and nothing trains until the user picks one.

Each model row carries a compatibility badge: **Unsloth** (fastest), **Standard** (transformers + PEFT), **Custom code** (ships its own modelling code; asks once, pins the commit) or **Not trainable** (with the reason, only after every fallback failed). Selecting a model opens *Variants & quantization*: the original plus every quantized repo on Hugging Face that names it as its base (bnb 4-bit, GPTQ, AWQ, FP8, GGUF), each with format, size and how it trains. GGUF rows say "serve only, trains from the original". A *Train in* control (4-bit / 8-bit / 16-bit) sets training precision independently. After any change the compatibility check runs (load, two steps, save/reload, generate) and shows backend, precision, peak memory, the largest context that fits, and which serving targets will work.

Hugging Face search lists coder models with parameter count, download size and a *fits / may not fit* badge estimated from the calibrated VRAM and the chosen method. The badge is advice, never a block: the user can pick any model, method, context or hyperparameter. Picking something that may not fit shows an inline warning with the estimate ("needs ~31 GB, you have 24 GB") and a "Use anyway" button. Auto values from the hardware tier are just the starting defaults. Local source accepts a Hugging Face-format folder (`config.json` + `.safetensors`); the app reads `config.json` to show architecture and size. `.gguf` files are serve-only and cannot be trained from (show that inline). Changing the base model asks for confirmation because it triggers a full rebuild.

## Behaviour

- **Live updates.** Rust pushes events (`progress`, `stage`, `metrics`, `log`) to the window. The UI redraws charts at most once a second and not at all while the window is hidden or minimised. Closing the window hides it to the tray; training continues.
- **Keyboard.** `1`–`8` and `,` switch screens; `Ctrl/⌘ K` opens search over screens, settings and runs; `Space` on Overview/Training toggles pause.
- **Saving.** Settings save on change (no Save button) and take effect at the next safe point. Changes that need a rebuild or restart say so inline.
- **Empty states.** Before the first scan: Overview shows a single "Scan your history" card. With fewer than 50 held-out tasks, Evaluation shows "Waiting for data: 48/50 tasks".
- **Accessibility.** Text contrast ≥ 4.5:1 in both themes; visible focus ring on every control; controls reachable by keyboard; status never conveyed by colour alone (pills carry words).
