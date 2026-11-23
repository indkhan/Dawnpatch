"""Edit formats shared by extract, train and evaluate.

Two target formats are built from the same (before, after) file pair:
  sr     SEARCH/REPLACE blocks (no line numbers)
  udiff  canonical unified diff (difflib, a/ b/ headers, 3 lines of context)

Both have an applier that turns (before, model output) into the edited file, or None.
"""
import difflib
import re

SR_OPEN, SR_MID, SR_CLOSE = "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"


# ---------------------------------------------------------------- search/replace
def make_sr(before, after, context=1):
    """SEARCH/REPLACE blocks that turn before into after. Each SEARCH is unique in before."""
    if before == "":
        return f"{SR_OPEN}\n{SR_MID}\n{after}{'' if after.endswith(chr(10)) else chr(10)}{SR_CLOSE}\n"
    a = before.splitlines(keepends=True)
    b = after.splitlines(keepends=True)
    blocks = []
    for group in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(context):
        i1, i2 = group[0][1], group[-1][2]
        j1, j2 = group[0][3], group[-1][4]
        # Grow context until the SEARCH text occurs once in the file (and is non-empty).
        while True:
            search = "".join(a[i1:i2])
            if search and before.count(search) == 1:
                break
            if i1 == 0 and i2 == len(a):
                break
            if i1 > 0:
                i1 -= 1
                j1 -= 1
            if i2 < len(a):
                i2 += 1
                j2 += 1
        if blocks and i1 < blocks[-1][1]:
            pi1, _, pj1, _ = blocks.pop()
            i1, j1 = min(i1, pi1), min(j1, pj1)
        blocks.append((i1, i2, j1, j2))
    blocks = [("".join(a[i1:i2]), "".join(b[j1:j2])) for i1, i2, j1, j2 in blocks]
    out = []
    for s, r in blocks:
        s = s if s.endswith("\n") or not s else s + "\n"
        r = r if r.endswith("\n") or not r else r + "\n"
        out.append(f"{SR_OPEN}\n{s}{SR_MID}\n{r}{SR_CLOSE}\n")
    return "".join(out)


SR_RX = re.compile(
    r"^<<<<<<< SEARCH\n(.*?)^=======\n(.*?)^>>>>>>> REPLACE[ \t]*$", re.M | re.S)


def apply_sr(before, text):
    """Apply SEARCH/REPLACE blocks in order. Returns the new file, or None if any block fails."""
    blocks = SR_RX.findall(text.replace("\r\n", "\n"))
    if not blocks:
        return None
    cur, pos = before, 0
    for search, replace in blocks:
        if search == "":
            if cur != "":
                return None
            cur, pos = replace, len(replace)
            continue
        at = cur.find(search, pos)
        if at < 0 and search.endswith("\n") and cur.endswith(search[:-1]):
            # last line of the file has no newline
            search = search[:-1]
            replace = replace[:-1] if replace.endswith("\n") else replace
            at = cur.find(search, pos)
        if at < 0:
            return None
        cur = cur[:at] + replace + cur[at + len(search):]
        pos = at + len(replace)
    return cur


# ---------------------------------------------------------------- unified diff
def make_udiff(before, after, path):
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="/dev/null" if before == "" else f"a/{path}", tofile=f"b/{path}", n=3)
    out = []
    for ln in lines:
        if not ln.endswith("\n"):
            ln += "\n\\ No newline at end of file\n"
        out.append(ln)
    return "".join(out)


HUNK_RX = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_udiff(before, text):
    """Lenient unified-diff applier: each hunk's old lines (context + '-') are located by content,
    nearest to the stated line number. Returns the new file, or None if a hunk cannot be placed."""
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    hunks, cur = [], None
    for ln in lines:
        m = HUNK_RX.match(ln)
        if m:
            cur = {"start": int(m.group(1)), "old": [], "new": []}
            hunks.append(cur)
            continue
        if cur is None or ln.startswith(("--- ", "+++ ")):
            continue
        if ln.startswith("\\"):
            cur["nonl"] = cur.get("last")
            continue
        if ln[:1] in (" ", "-", "+"):
            cur["last"] = ln[:1]
        if ln.startswith(" "):
            cur["old"].append(ln[1:])
            cur["new"].append(ln[1:])
        elif ln.startswith("-"):
            cur["old"].append(ln[1:])
        elif ln.startswith("+"):
            cur["new"].append(ln[1:])
        elif ln == "":
            cur["old"].append("")
            cur["new"].append("")
    if not hunks:
        return None
    if hunks[-1]["old"] and hunks[-1]["old"][-1] == "" and hunks[-1]["new"] and hunks[-1]["new"][-1] == "":
        hunks[-1]["old"].pop()
        hunks[-1]["new"].pop()
    had_nl = before.endswith("\n") or before == ""
    file_lines = before.split("\n")
    if had_nl and file_lines and file_lines[-1] == "":
        file_lines.pop()
    offset = 0
    pos_floor = 0
    for h in hunks:
        old, new = h["old"], h["new"]
        if not old:
            if file_lines and before != "":
                at = max(0, min(len(file_lines), h["start"] + offset))
            else:
                at = 0
            file_lines[at:at] = new
            offset += len(new)
            pos_floor = at + len(new)
            continue
        want = h["start"] - 1 + offset
        cands = [i for i in range(pos_floor, len(file_lines) - len(old) + 1)
                 if file_lines[i:i + len(old)] == old]
        if not cands:
            return None
        at = min(cands, key=lambda i: abs(i - want))
        file_lines[at:at + len(old)] = new
        offset += len(new) - len(old)
        pos_floor = at + len(new)
    out = "\n".join(file_lines)
    return out + "\n" if (had_nl and file_lines) else out


FORMATS = {
    "sr": {
        "system": ("You are a coding assistant. You get a request and the current contents of one file. "
                   "Reply only with edits to that file as SEARCH/REPLACE blocks:\n"
                   f"{SR_OPEN}\n<exact lines copied from the file>\n{SR_MID}\n<the new lines>\n{SR_CLOSE}\n"
                   "Use one block per change. SEARCH must match the file exactly. "
                   "Use an empty SEARCH only when the file is new and empty."),
        "make": lambda before, after, path: make_sr(before, after),
        "apply": apply_sr,
    },
    "udiff": {
        "system": ("You are a coding assistant. You get a request and the current contents of one file. "
                   "Reply only with a unified diff of that file (--- a/<path>, +++ b/<path>, then @@ hunks)."),
        "make": make_udiff,
        "apply": apply_udiff,
    },
}


def build_prompt(request, path, before):
    shown = before if before else "(new file, currently empty)\n"
    return f"{request}\n\nFile: {path}\n```\n{shown}{'' if shown.endswith(chr(10)) else chr(10)}```"
