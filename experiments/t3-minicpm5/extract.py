#!/usr/bin/env python3
"""T3 Code history -> prompt/diff examples (train + held-out).

Reads %USERPROFILE%\\.t3\\userdata\\statev2.sqlite read-only and writes:
  work/train.jsonl, work/heldout.jsonl, work/extract_stats.json

Data rules (plan.txt, "Principles and data rules"):
  keep    developer-typed user messages on top-level threads, and file_change diffs
  drop    assistant text, reasoning, subagent turns, dynamic tool args, handoff/compaction text
  Only counts of drops are kept. Matched secret values are never printed or stored.
"""
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
import collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
SRC = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".t3" / "userdata" / "statev2.sqlite"

PROMPT_MAX = 4000
DIFF_MAX = 20000
TOTAL_DIFF_MAX = 30000
HELDOUT_FRACTION = 0.2
SEED = 42

# ---------------------------------------------------------------- secret scrub
SECRET_RULES = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key_header", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")),
    ("openai_style_key", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    ("slack_token", re.compile(r"xox[bpa]-[A-Za-z0-9\-]{10,}")),
    ("password_assignment", re.compile(r"password\s*[=:]\s*['\"][^'\"]{4,}['\"]", re.I)),
    ("generic_secret_assignment", re.compile(
        r"(?:api[_-]?key|secret|access[_-]?token)\s*[=:]\s*['\"][A-Za-z0-9/+_\-]{16,}['\"]", re.I)),
]
B64_ASSIGN = re.compile(r"[=:]\s*['\"]?([A-Za-z0-9+/]{41,}={0,2})['\"]?")
ENV_LINE = re.compile(r"^\+?[A-Z][A-Z0-9_]{2,}=\S{6,}$", re.M)
ENV_FILE = re.compile(r"(^|[\\/])\.env(\.[^\\/]+)?$", re.I)

# Messages that look like forwarded or injected agent output, not typed by the developer.
FORWARD_RULES = [re.compile(p, re.I | re.M) for p in [
    r"<task-notification>",
    r"<system-reminder>",
    r"\[Artifact comment",
    r"Continue where you left off",
    r"^\s*You are working in the git worktree",
    r"orchestrator, thread",
    r"^\s*Forwarded (message|from)\b",
    r"<subagent",
    r"^\s*Handoff from\b",
]]


def shannon(s):
    if not s:
        return 0.0
    counts = collections.Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def secret_hit(text):
    """Return a rule name if text matches a secret pattern, else None. Never returns the value."""
    for name, rx in SECRET_RULES:
        if rx.search(text):
            return name
    for m in B64_ASSIGN.finditer(text):
        if shannon(m.group(1)) > 4.0:
            return "long_base64_assignment"
    return None


def forwarded_hit(text):
    return any(rx.search(text) for rx in FORWARD_RULES)


# ---------------------------------------------------------------- audit helpers
TOKEN = re.compile(r"[a-z0-9_']+")
ID_LIKE = re.compile(r"^[\w:%.\-/]{30,}$")
SKIP_KEYS = re.compile(r"(id|Id|ref|Ref|At|ordinal|status|type|kind|Name|name|cwd|path)$")


def ngrams8(text):
    toks = TOKEN.findall(text.lower())
    return {hash(tuple(toks[i:i + 8])) for i in range(len(toks) - 7)}


def walk_strings(obj, out):
    if isinstance(obj, str):
        if len(obj) >= 20 and not ID_LIKE.match(obj):
            out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if SKIP_KEYS.search(k):
                continue
            walk_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_strings(v, out)


# ---------------------------------------------------------------- source access
REQUIRED = {
    "orchestration_v2_projection_turn_items": {"thread_id", "ordinal", "type", "payload_json", "parent_item_id", "turn_item_id"},
    "orchestration_v2_projection_threads": {"thread_id", "project_id", "created_at", "deleted_at", "payload_json"},
    "projection_projects": {"project_id", "workspace_root"},
    "orchestration_v2_projection_context_transfers": {"target_thread_id", "type", "status"},
    "orchestration_v2_projection_subagents": {"child_thread_id"},
    "orchestration_v2_projection_messages": {"thread_id", "role", "payload_json"},
    "projection_thread_messages": {"thread_id", "role", "text"},
}


def check_schema(con):
    for table, cols in REQUIRED.items():
        found = {r[1] for r in con.execute(f'pragma table_info("{table}")')}
        if not found:
            raise SystemExit(f"FAIL: table {table} missing. Unknown T3 schema, refusing to continue.")
        missing = cols - found
        if missing:
            raise SystemExit(f"FAIL: {table} missing columns {sorted(missing)}. Unknown T3 schema.")
    try:
        ver = con.execute("select schema_version from orchestration_v2_projection_metadata").fetchall()
        print("projection schema_version rows:", ver)
    except sqlite3.Error:
        pass


def open_source():
    """Read-only open. Fall back to a backup-API snapshot if WAL/locks block the read-only open."""
    try:
        con = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
        con.execute("select count(*) from sqlite_master").fetchone()
        print("opened source read-only:", SRC)
        return con
    except sqlite3.Error as e:
        print(f"read-only open failed ({e}); taking a backup-API snapshot into work/")
    snap = WORK / "t3-snapshot.sqlite"
    if not snap.exists():
        raw = WORK / "t3-raw"
        raw.mkdir(parents=True, exist_ok=True)
        for suffix in ["", "-wal", "-shm"]:
            src_file = Path(str(SRC) + suffix)
            if src_file.exists():
                shutil.copy2(src_file, raw / ("statev2.sqlite" + suffix))
        copy = sqlite3.connect(str(raw / "statev2.sqlite"))
        dst = sqlite3.connect(str(snap))
        copy.backup(dst)
        dst.close()
        copy.close()
        shutil.rmtree(raw, ignore_errors=True)
    con = sqlite3.connect(f"file:{snap}?mode=ro", uri=True)
    print("opened snapshot:", snap)
    return con


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    con = open_source()
    check_schema(con)
    cur = con.cursor()

    # Thread metadata and exclusions (child, fork, subagent and handoff threads).
    workspace = dict(cur.execute("select project_id, workspace_root from projection_projects"))
    handoff_targets = {r[0] for r in cur.execute(
        "select target_thread_id from orchestration_v2_projection_context_transfers where target_thread_id is not null")}
    subagent_children = {r[0] for r in cur.execute(
        "select child_thread_id from orchestration_v2_projection_subagents where child_thread_id is not null")}

    threads = {}
    excluded = collections.Counter()
    for tid, pid, created, deleted, payload in cur.execute(
            "select thread_id, project_id, created_at, deleted_at, payload_json from orchestration_v2_projection_threads"):
        d = json.loads(payload) if payload else {}
        lineage = d.get("lineage") or {}
        reason = None
        if lineage.get("parentThreadId") is not None:
            reason = "has_parent_thread"
        elif d.get("forkedFrom") is not None:
            reason = "forked"
        elif tid in handoff_targets:
            reason = "handoff_or_transfer_target"
        elif tid in subagent_children:
            reason = "subagent_child"
        if reason:
            excluded[reason] += 1
            continue
        threads[tid] = {"project_id": pid, "created": created, "deleted": deleted is not None}
    print("eligible top-level threads:", len(threads), "excluded:", dict(excluded))

    stats = collections.Counter()
    examples = []          # kept examples (in memory until split)
    thread_first_time = {}  # thread_id -> earliest user message time

    # Stream turn items grouped by thread (index: thread_id, ordinal).
    rows = cur.execute(
        "select thread_id, ordinal, type, payload_json from orchestration_v2_projection_turn_items "
        "where type in ('user_message','file_change','assistant_message','reasoning') "
        "order by thread_id, ordinal")

    def flush(tid, items):
        if tid not in threads:
            return
        # Agent-side text for the n-gram audit (assistant, reasoning, non-developer user text).
        # Held in memory for this thread only; the text string is discarded after hashing.
        agent_grams = set()
        for (_, _, typ, pj) in items:
            d = json.loads(pj) if pj else {}
            if typ == "assistant_message" or typ == "reasoning":
                buf = []
                walk_strings(d, buf)
                for s in buf:
                    agent_grams |= ngrams8(s)
            elif typ == "user_message" and d.get("createdBy") != "user":
                agent_grams |= ngrams8(d.get("text") or "")
        for (txt,) in cur2.execute("select text from projection_thread_messages where thread_id=? and role='assistant'", (tid,)):
            agent_grams |= ngrams8(txt or "")
        for (pj,) in cur2.execute("select payload_json from orchestration_v2_projection_messages where thread_id=? and role='assistant'", (tid,)):
            buf = []
            walk_strings(json.loads(pj) if pj else {}, buf)
            for s in buf:
                agent_grams |= ngrams8(s)

        # Earliest message time (any author) for date splitting.
        times = [json.loads(pj).get("startedAt") for (typ, pj) in
                 ((t, p) for (_, _, t, p) in items) if typ == "user_message"]
        times = [t for t in times if t]
        if times:
            thread_first_time[tid] = min(times)
        else:
            thread_first_time[tid] = threads[tid]["created"]

        # Order by payload startedAt (ordinal restarts per provider turn, so it cannot be used alone).
        parsed = []
        for (_, ordinal, typ, pj) in items:
            d = json.loads(pj) if pj else {}
            ts = d.get("startedAt") or d.get("createdAt") or ""
            parsed.append((ts, ordinal, typ, d))
        parsed.sort(key=lambda x: (x[0], x[1]))

        # Each file_change with a non-empty diffStr attaches to the most recent preceding
        # developer-typed user_message (createdBy == "user") in this thread, by time.
        turn = None  # dict(msg=dict, created=str, diffs=[], files=[])
        turns = []
        for ts, _, typ, d in parsed:
            if typ == "user_message":
                if d.get("createdBy") != "user":
                    continue  # agent/system-typed messages are never prompts
                if turn is not None:
                    turns.append(turn)
                turn = {"msg": d, "created": d.get("startedAt") or d.get("createdAt"), "diffs": [], "files": []}
            elif typ == "file_change" and turn is not None:
                diff = d.get("diffStr")
                if diff:
                    turn["diffs"].append((d.get("fileName") or "", diff))
        if turn is not None:
            turns.append(turn)

        ws_root = (workspace.get(threads[tid]["project_id"]) or "").replace("\\", "/").rstrip("/")

        def rel_name(name):
            n = name.replace("\\", "/")
            if ws_root and n.lower().startswith(ws_root.lower() + "/"):
                return n[len(ws_root) + 1:]
            return n

        # One example per (developer turn, file): diffs to the same file are concatenated in time order.
        for t in turns:
            if not t["diffs"]:
                stats["turns_without_diff"] += 1
                continue
            stats["turns_with_diff"] += 1
            by_file = collections.OrderedDict()
            for fname, diff in t["diffs"]:
                by_file.setdefault(fname, []).append(diff)
            msg = t["msg"]
            prompt_msg = (msg.get("text") or "").strip()
            if not prompt_msg:
                stats["drop_empty_prompt"] += 1
                continue
            if forwarded_hit(prompt_msg):
                stats["drop_forwarded_or_injected_prompt"] += 1
                continue
            hit = secret_hit(prompt_msg)
            if hit:
                stats[f"drop_secret_prompt:{hit}"] += 1
                continue
            prompt_msg = prompt_msg[:PROMPT_MAX]
            for fname, diff_list in by_file.items():
                stats["file_groups"] += 1
                rel = rel_name(fname)
                prompt = f"{prompt_msg}\n\nFile to edit: {rel}"
                if ENV_FILE.search(fname or ""):
                    stats["drop_env_file"] += 1
                    continue
                diffs = [dd if dd.endswith("\n") else dd + "\n" for dd in diff_list]
                full_diff = "".join(diffs)
                if len(ENV_LINE.findall(full_diff)) >= 3:
                    stats["drop_env_file_contents"] += 1
                    continue
                hit = secret_hit(full_diff) or secret_hit(prompt)
                if hit:
                    stats[f"drop_secret:{hit}"] += 1
                    continue
                # Per-file cap: truncate each diff at DIFF_MAX; drop the example if the total is still too large.
                target = "".join(dd[:DIFF_MAX] for dd in diffs)
                if len(target) > TOTAL_DIFF_MAX:
                    stats["drop_diff_too_large"] += 1
                    continue
                if ngrams8(prompt) & agent_grams:
                    stats["drop_8gram_overlap_with_agent_text"] += 1
                    continue
                examples.append({
                    "thread_id": tid,
                    "project": workspace.get(threads[tid]["project_id"]),
                    "created_at": t["created"],
                    "prompt": prompt,
                    "diff": target,
                    "file": rel,
                    "files": [rel],
                })
                stats["kept_examples"] += 1
        # agent_grams and the thread's text go out of scope here.

    cur2 = con.cursor()
    current, bucket = None, []
    n_rows = 0
    for tid, ordinal, typ, pj in rows:
        n_rows += 1
        if tid != current:
            if current is not None:
                flush(current, bucket)
            current, bucket = tid, []
        bucket.append((tid, ordinal, typ, pj))
    if current is not None:
        flush(current, bucket)
    print(f"streamed {n_rows} turn-item rows in {time.time() - t0:.0f}s")

    # Split: per workspace, newest 20% of threads (by first message time) are heldout.
    by_thread = collections.defaultdict(list)
    for ex in examples:
        by_thread[ex["thread_id"]].append(ex)
    per_ws = collections.defaultdict(list)
    for tid in by_thread:
        ws = workspace.get(threads[tid]["project_id"], "unknown")
        per_ws[ws].append(tid)
    heldout_threads, train_threads = set(), set()
    for ws, tids in per_ws.items():
        tids.sort(key=lambda t: thread_first_time.get(t) or "")
        k = int(round(HELDOUT_FRACTION * len(tids)))
        heldout_threads |= set(tids[len(tids) - k:]) if k else set()
        train_threads |= set(tids[:len(tids) - k])

    def write(path, tids):
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for tid in sorted(tids, key=lambda t: thread_first_time.get(t) or ""):
                for ex in sorted(by_thread[tid], key=lambda e: e["created_at"] or ""):
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n += 1
        return n

    n_train = write(WORK / "train.jsonl", train_threads)
    n_held = write(WORK / "heldout.jsonl", heldout_threads)

    out = {
        "source": str(SRC),
        "threads_eligible_top_level": len(threads),
        "threads_excluded": dict(excluded),
        "threads_with_examples": len(by_thread),
        "threads_train": len(train_threads),
        "threads_heldout": len(heldout_threads),
        "examples_train": n_train,
        "examples_heldout": n_held,
        "drops_and_counts": dict(sorted(stats.items())),
        "projects_with_examples": len(per_ws),
    }
    (WORK / "extract_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if n_train < 200:
        print("WARNING: fewer than 200 train examples; continuing as instructed.")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
