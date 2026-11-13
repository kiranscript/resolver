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
    for line in snippet.split("\n"):
        if side is None and line.startswith("<<<<<<<"):
            side, left, right = "L", [], []
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
    return "\n".join(out)


def trivial_resolve(question, strategy):
    """Apply a trivial strategy to every conflict in the snippet. Returns a fenced answer."""
    snippet = _snippet_parts(question)
    resolved = snippet if strategy == "untouched" else _resolve_lines(snippet, strategy)
    return "```java\n" + resolved.rstrip("\n") + "\n```"
