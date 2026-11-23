#!/usr/bin/env python3
"""T3 Code history -> (request, file at start state, file at end state) examples.

Differences from v1 (experiments/t3-minicpm5/extract.py):
  - the model sees the file it must edit, read from the T3 checkpoint ref before the turn
  - the target is rebuilt from git (start checkpoint -> last checkpoint before the developer's next
    message), not taken from provider diffStr payloads, so every target is one clean format
  - every target is checked: applying it to the start file must give the end file exactly
  - requests that cannot stand on their own ("sure sounds good", "do option 2") are dropped
  - one example per (developer turn, file); diffs are never concatenated across turns

Known shortcut (not the plan's retrieval): the touched file is given to the model. At serving time
the model would choose and read files itself. This isolates "can it make the edit" from "can it
find the file".

Writes work/train.jsonl, work/heldout.jsonl, work/extract_stats.json.
"""
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "t3-minicpm5"))
import extract as V1  # noqa: E402  (secret scrub, forwarded check, n-gram audit, read-only open)
import edits as E  # noqa: E402

V1.WORK = HERE / "work"
WORK = V1.WORK
PROMPT_MAX = 3000
FILE_MAX = 7000          # chars of the start-state file; bigger files are dropped
CHANGE_MAX = 4000        # chars of the sr target; bigger edits are dropped
HELDOUT_FRACTION = 0.2

REFERENTIAL = re.compile(
    r"^\s*(sure|yes|yep|yeah|ok|okay|k|go|go ahead|do it|do that|continue|proceed|sounds good|lgtm|great|"
    r"perfect|nice|cool|thanks|thank you|fine|alright|right)\b", re.I)
POINTER = re.compile(r"\b(option|approach|choice|number)\s*\d\b|\b(the above|what you said|you suggested|"
                     r"as you said|that one|this one|those)\b", re.I)


def self_contained(text):
    words = re.findall(r"\w+", text)
    if len(words) < 6:
        return False, "too_short"
    if REFERENTIAL.match(text) and len(words) < 25:
        return False, "starts_with_ack"
    if POINTER.search(text) and len(words) < 25:
        return False, "points_at_agent_reply"
    return True, None


def git_show(root, ref, rel, cache):
    key = (root, ref, rel)
    if key in cache:
        return cache[key]
    r = subprocess.run(["git", "-C", root, "show", f"{ref}:{rel}"], capture_output=True)
    if r.returncode != 0:
        val = None
    else:
        try:
            val = r.stdout.decode("utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            val = "\0binary"
    cache[key] = val
    return val


def ref_ok(root, ref, cache):
    key = ("ok", root, ref)
    if key not in cache:
        r = subprocess.run(["git", "-C", root, "rev-parse", "--verify", "-q", ref + "^{commit}"], capture_output=True)
        cache[key] = r.returncode == 0
    return cache[key]


def norm(p):
    return (p or "").replace("\\", "/").rstrip("/")


def main():
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    con = V1.open_source()
    V1.check_schema(con)
    cur = con.cursor()
    cur2 = con.cursor()
    workspace = dict(cur.execute("select project_id, workspace_root from projection_projects"))
    handoff_targets = {r[0] for r in cur.execute(
        "select target_thread_id from orchestration_v2_projection_context_transfers where target_thread_id is not null")}
    subagent_children = {r[0] for r in cur.execute(
        "select child_thread_id from orchestration_v2_projection_subagents where child_thread_id is not null")}

    threads, excluded = {}, collections.Counter()
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
        threads[tid] = {"project_id": pid, "created": created, "worktree": d.get("worktreePath")}

    # Checkpoints per thread, oldest first.
    ckpts = collections.defaultdict(list)
    for tid, captured, status, pj in cur.execute(
            "select thread_id, captured_at, status, payload_json from orchestration_v2_projection_checkpoints"):
        if tid in threads and status == "ready":
            ref = (json.loads(pj) if pj else {}).get("ref")
            if ref:
                ckpts[tid].append((captured, ref))
    for v in ckpts.values():
        v.sort()

    stats = collections.Counter()
    examples, thread_first_time = [], {}
    git_cache = {}

    rows = cur.execute(
        "select thread_id, ordinal, type, payload_json from orchestration_v2_projection_turn_items "
        "where type in ('user_message','file_change','assistant_message','reasoning') order by thread_id, ordinal")

    def flush(tid, items):
        if tid not in threads:
            return
        info = threads[tid]
        root = workspace.get(info["project_id"])
        # n-gram audit set (agent text, held in memory for this thread only)
        agent_grams = set()
        for (_, _, typ, pj) in items:
            d = json.loads(pj) if pj else {}
            if typ in ("assistant_message", "reasoning"):
                buf = []
                V1.walk_strings(d, buf)
                for s in buf:
                    agent_grams |= V1.ngrams8(s)
            elif typ == "user_message" and d.get("createdBy") != "user":
                agent_grams |= V1.ngrams8(d.get("text") or "")
        for (txt,) in cur2.execute("select text from projection_thread_messages where thread_id=? and role='assistant'", (tid,)):
            agent_grams |= V1.ngrams8(txt or "")

        parsed = []
        for (_, ordinal, typ, pj) in items:
            d = json.loads(pj) if pj else {}
            parsed.append((d.get("startedAt") or d.get("createdAt") or "", ordinal, typ, d))
        parsed.sort(key=lambda x: (x[0], x[1]))
        times = [ts for ts, _, typ, _ in parsed if typ == "user_message" and ts]
        thread_first_time[tid] = min(times) if times else info["created"]

        turns, turn = [], None
        for ts, _, typ, d in parsed:
            if typ == "user_message" and d.get("createdBy") == "user":
                if turn:
                    turns.append(turn)
                turn = {"msg": d, "at": ts, "files": []}
            elif typ == "file_change" and turn is not None and d.get("fileName"):
                if d["fileName"] not in turn["files"]:
                    turn["files"].append(d["fileName"])
        if turn:
            turns.append(turn)

        prefixes = [p for p in (norm(info.get("worktree")), norm(root)) if p]
        cks = ckpts.get(tid, [])
        for k, t in enumerate(turns):
            if not t["files"]:
                continue
            stats["turns_with_file_change"] += 1
            nxt = turns[k + 1]["at"] if k + 1 < len(turns) else "9999"
            start = [ref for (c, ref) in cks if c <= t["at"]]
            end = [ref for (c, ref) in cks if t["at"] < c <= nxt]
            if not root or not os.path.isdir(root):
                stats["drop_repo_missing"] += 1
                continue
            if not start or not end:
                stats["drop_no_checkpoint_pair"] += 1
                continue
            s_ref, e_ref = start[-1], end[-1]
            if not (ref_ok(root, s_ref, git_cache) and ref_ok(root, e_ref, git_cache)):
                stats["drop_checkpoint_ref_missing"] += 1
                continue
            request = (t["msg"].get("text") or "").strip()
            if not request:
                stats["drop_empty_prompt"] += 1
                continue
            if V1.forwarded_hit(request):
                stats["drop_forwarded_or_injected_prompt"] += 1
                continue
            ok, why = self_contained(request)
            if not ok:
                stats[f"drop_not_self_contained:{why}"] += 1
                continue
            if V1.secret_hit(request):
                stats["drop_secret_prompt"] += 1
                continue
            if V1.ngrams8(request) & agent_grams:
                stats["drop_8gram_overlap_with_agent_text"] += 1
                continue
            request = request[:PROMPT_MAX]
            for fname in t["files"]:
                stats["file_candidates"] += 1
                f = norm(fname)
                rel = None
                for p in prefixes:
                    if f.lower().startswith(p.lower() + "/"):
                        rel = f[len(p) + 1:]
                        break
                if rel is None:
                    stats["drop_file_outside_repo"] += 1
                    continue
                if V1.ENV_FILE.search(rel):
                    stats["drop_env_file"] += 1
                    continue
                before = git_show(root, s_ref, rel, git_cache)
                after = git_show(root, e_ref, rel, git_cache)
                if after is None:
                    stats["drop_deleted_or_not_kept"] += 1
                    continue
                # Trailing-newline differences are normalised away (both appliers assume a final newline).
                before = (before or "")
                before = before if before.endswith("\n") or not before else before + "\n"
                after = after if after.endswith("\n") or not after else after + "\n"
                if "\0" in before or "\0" in after:
                    stats["drop_binary"] += 1
                    continue
                if before == after:
                    stats["drop_no_net_change_at_end_checkpoint"] += 1
                    continue
                if len(before) > FILE_MAX:
                    stats["drop_file_too_large"] += 1
                    continue
                if V1.secret_hit(before) or V1.secret_hit(after) or len(V1.ENV_LINE.findall(after)) >= 3:
                    stats["drop_secret_in_file"] += 1
                    continue
                sr = E.make_sr(before, after)
                ud = E.make_udiff(before, after, rel)
                if len(sr) > CHANGE_MAX:
                    stats["drop_change_too_large"] += 1
                    continue
                if E.apply_sr(before, sr) != after:
                    stats["drop_sr_roundtrip_failed"] += 1
                    continue
                if E.apply_udiff(before, ud) != after:
                    stats["drop_udiff_roundtrip_failed"] += 1
                    continue
                examples.append({
                    "thread_id": tid, "project": root, "created_at": t["at"], "file": rel,
                    "start_ref": s_ref, "end_ref": e_ref, "evidence": "exact_tree",
                    "retention": "next_checkpoint",
                    "request": request, "prompt": E.build_prompt(request, rel, before),
                    "before": before, "after": after, "target_sr": sr, "target_udiff": ud,
                })
                stats["kept_examples"] += 1

    current, bucket = None, []
    for tid, ordinal, typ, pj in rows:
        if tid != current:
            if current is not None:
                flush(current, bucket)
            current, bucket = tid, []
        bucket.append((tid, ordinal, typ, pj))
    if current is not None:
        flush(current, bucket)

    by_thread = collections.defaultdict(list)
    for ex in examples:
        by_thread[ex["thread_id"]].append(ex)
    per_ws = collections.defaultdict(list)
    for tid in by_thread:
        per_ws[workspace.get(threads[tid]["project_id"], "unknown")].append(tid)
    held, train = set(), set()
    for ws, tids in per_ws.items():
        tids.sort(key=lambda x: thread_first_time.get(x) or "")
        k = int(round(HELDOUT_FRACTION * len(tids)))
        held |= set(tids[len(tids) - k:]) if k else set()
        train |= set(tids[:len(tids) - k])

    def write(path, tids):
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for tid in sorted(tids, key=lambda x: thread_first_time.get(x) or ""):
                for ex in sorted(by_thread[tid], key=lambda e: e["created_at"] or ""):
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n += 1
        return n

    out = {
        "threads_eligible_top_level": len(threads), "threads_excluded": dict(excluded),
        "threads_with_examples": len(by_thread), "threads_train": len(train), "threads_heldout": len(held),
        "examples_train": write(WORK / "train.jsonl", train), "examples_heldout": write(WORK / "heldout.jsonl", held),
        "drops_and_counts": dict(sorted(stats.items())), "projects_with_examples": len(per_ws),
        "seconds": round(time.time() - t0),
    }
    (WORK / "extract_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
