"""Scorer for Merge-Bench Java hunks.

Mirrors Merge-Bench's `merged_conflict_reward` (src/evaluation_metrics.py,
src/utils.py) so numbers are comparable with the paper, but splits the outcome
into explicit buckets and has no heavy dependencies.

Buckets (mutually exclusive):
  exact      code block == developer resolution (after .strip())
  normalized code block equals developer resolution after dropping comments and
             collapsing whitespace, but not exact
  conflict   code block still contains conflict markers (model "declined")
  wrong      code block present, no markers, does not match
  no_block   no markdown code block found (includes truncated generations)
"""
import re

CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)
BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/")
LINE_COMMENT_RE = re.compile(r"//.*")
WHITESPACE_RE = re.compile(r"\s+")
CONFLICT_MARKERS = ("<<<<<<<", "=======", "|||||||", ">>>>>>>")
BUCKETS = ("exact", "normalized", "conflict", "wrong", "no_block")


def extract_code_block(text, which="first"):
    """First fenced block (Merge-Bench's rule) or, for models that think out loud, the last one."""
    ms = CODE_BLOCK_RE.findall(text)
    if not ms:
        return None
    return (ms[0] if which == "first" else ms[-1]).strip()


def normalize_java(code):
    code = BLOCK_COMMENT_RE.sub("", code)
    code = LINE_COMMENT_RE.sub("", code)
    return WHITESPACE_RE.sub(" ", code).strip()


def has_conflict_markers(code):
    return any(m in code for m in CONFLICT_MARKERS)


def score(answer_text, reference, which="first"):
    """Classify one model answer (reasoning already removed) against the reference."""
    code = extract_code_block(answer_text, which)
    if code is None:
        return "no_block"
    if code == reference.strip():
        return "exact"
    if normalize_java(code) == normalize_java(reference.strip()):
        return "normalized"
    if has_conflict_markers(code):
        return "conflict"
    return "wrong"


def summarize(buckets):
    n = len(buckets)
    out = {b: sum(x == b for x in buckets) for b in BUCKETS}
    out["n"] = n
    out["exact_pct"] = 100 * out["exact"] / n if n else 0.0
    out["exact_or_normalized_pct"] = 100 * (out["exact"] + out["normalized"]) / n if n else 0.0
    out["conflict_pct"] = 100 * out["conflict"] / n if n else 0.0
    return out


# --- trivial baselines (Level 0) -------------------------------------------

def _snippet_parts(question):
    """Return the fenced snippet from the prompt (text between the first ```java fence and the last fence)."""
    start = question.index("```java\n") + len("```java\n")
    end = question.rindex("\n```")
    return question[start:end]


def _resolve_lines(snippet, strategy):
    """Line-based resolver; handles empty sides and multiple conflicts."""
    out, side, left, right = [], None, [], []
    open_marker = None
    for line in snippet.split("\n"):
        if side is None and line.startswith("<<<<<<<"):
            side, left, right, open_marker = "L", [], [], line
        elif side == "L" and line.startswith("======="):
            side = "R"
        elif side == "R" and line.startswith(">>>>>>>"):
            if strategy == "left":
                out += left
            elif strategy == "right":
                out += right
            elif strategy == "both":
                out += left + right
            elif strategy == "both_reversed":
                out += right + left
            else:
                raise ValueError(strategy)
            side = None
        elif side == "L":
            left.append(line)
        elif side == "R":
            right.append(line)
        else:
            out.append(line)
    if side is not None:
        # Dangling half-conflict from the context window (the developer's answer keeps it): emit verbatim.
        out += [open_marker] + left + (["======="] + right if side == "R" else [])
    return "\n".join(out)


def declined_strict(code, reference):
    """Stricter 'left the conflict' test: more line-anchored `<<<<<<<` markers than the reference has.

    Merge-Bench's own bucket (`conflict`) fires on any marker substring, and ~12% of reference answers
    contain stray markers from neighbouring conflicts or `// =====` comment rulers, so it over-counts.
    """
    def n(t):
        return sum(1 for l in t.split("\n") if l.startswith("<<<<<<<"))
    return n(code) > n(reference)


def trivial_resolve(question, strategy):
    """Apply a trivial strategy to every conflict in the snippet. Returns a fenced answer."""
    snippet = _snippet_parts(question)
    resolved = snippet if strategy == "untouched" else _resolve_lines(snippet, strategy)
    return "```java\n" + resolved.rstrip("\n") + "\n```"


# --- side swap (Level 2/3) --------------------------------------------------

def swap_sides(question):
    """Swap the contents of the first complete conflict's two sides, keeping the marker lines in place.

    The developer's resolution does not depend on which side is printed first, so `answer` is unchanged.
    Used to (a) remove the left/right orientation prior from training data and (b) build an
    orientation-neutral dev set. Stray half-conflicts in the context are left alone.
    """
    start = question.index("```java\n") + len("```java\n")
    end = question.rindex("\n```")
    lines = question[start:end].split("\n")
    i_open = i_mid = i_close = None
    for i, l in enumerate(lines):
        if i_open is None and l.startswith("<<<<<<<"):
            i_open = i
        elif i_open is not None and i_mid is None and l.startswith("======="):
            i_mid = i
        elif i_mid is not None and l.startswith(">>>>>>>"):
            i_close = i
            break
        elif i_mid is None and i_open is not None and l.startswith("<<<<<<<"):
            i_open = i  # an earlier '<<<<<<<' was a stray half; restart at this one
    if i_close is None:
        return question
    left, right = lines[i_open + 1:i_mid], lines[i_mid + 1:i_close]
    new = lines[:i_open + 1] + right + [lines[i_mid]] + left + lines[i_close:]
    return question[:start] + "\n".join(new) + question[end:]


def can_swap(question):
    """True when swap_sides is safe: it changes the text, is its own inverse, and exchanges take-left/take-right."""
    s = swap_sides(question)
    return (s != question and swap_sides(s) == question
            and trivial_resolve(s, "left") == trivial_resolve(question, "right")
            and trivial_resolve(s, "right") == trivial_resolve(question, "left"))


def swap_example(question, answer):
    """Side-swapped copy of an example. Returns (question, answer) or None when the swap is unsafe.

    The developer's answer is order-independent except when it is exactly a concatenation of the two sides
    (left+right or right+left); then it is re-derived in the new display order so the target stays
    "concatenate in the order shown".
    """
    if not can_swap(question):
        return None
    s = swap_sides(question)
    for strat in ("both", "both_reversed"):
        if _resolve_lines(_snippet_parts(question), strat).strip() == answer.strip():
            # raw (unstripped) text: stripping would drop the first line's indentation
            return s, _resolve_lines(_snippet_parts(s), strat).rstrip("\n") + "\n"
    return s, answer


def merge_sha(question):
    """40-hex merge SHA from the conflict labels (the best available repo/merge id), or None."""
    import re
    for line in question.split("\n"):
        if line.startswith(("<<<<<<<", ">>>>>>>")):
            m = re.search(r"\b[0-9a-f]{40}\b", line)
            if m:
                return m.group(0)
    return None
