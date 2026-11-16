"""Prompt/target formats for fine-tuning and evaluation (Level 3/4).

full        : paper's format. The model rewrites the whole snippet (context + resolution) in a fenced block.
resolution  : the model writes only the lines that replace the conflict block; a script rebuilds the snippet
              as prefix + resolution + suffix. Shorter output, no context copying.

Both use the no-decline instruction (Level 1's prompt B), the best prompt for both baselines.
"""
from scoring import extract_code_block

DECLINE_SENTENCE = (
    "If you are not sure on how to resolve the conflict or if the intent is ambiguous, "
    "please return the same snippet with the conflict.\n"
)
FULL_INSTR = (
    "You must always resolve the conflict. Never return the snippet with conflict markers "
    "(<<<<<<<, =======, >>>>>>>); if the intent is ambiguous, choose the most plausible resolution.\n"
)
RES_HEAD = (
    "You are a semantic merge conflict resolution expert. Below is a snippet of code with surrounding context "
    "that includes one merge conflict (between the <<<<<<< and >>>>>>> markers).\n"
    "Return ONLY the lines that should replace the whole conflict block (from <<<<<<< to >>>>>>> inclusive), "
    "in a markdown code block. Do not repeat the surrounding context and never return conflict markers.\n"
    "Think in terms of intent and semantics that both sides of the merge are trying to achieve; "
    "if the intent is ambiguous, choose the most plausible resolution.\n"
    "Here is the code snippet:\n"
)


def _snippet_bounds(q):
    start = q.index("```java\n") + len("```java\n")
    return start, q.rindex("\n```")


def prompt_full(question):
    assert DECLINE_SENTENCE in question
    return question.replace(DECLINE_SENTENCE, FULL_INSTR)


def prompt_resolution(question):
    s, e = _snippet_bounds(question)
    return RES_HEAD + "```java\n" + question[s:e] + "\n```\n"


def _conflict_span(lines):
    """Indices (open, close) of the first complete conflict (same rule as scoring.swap_sides)."""
    i_open = i_mid = None
    for i, l in enumerate(lines):
        if i_open is None and l.startswith("<<<<<<<"):
            i_open = i
        elif i_open is not None and i_mid is None and l.startswith("======="):
            i_mid = i
        elif i_mid is not None and l.startswith(">>>>>>>"):
            return i_open, i
        elif i_mid is None and i_open is not None and l.startswith("<<<<<<<"):
            i_open = i
    return None


def _trim_blank(lines):
    i, j = 0, len(lines)
    while i < j and lines[i].strip() == "":
        i += 1
    while j > i and lines[j - 1].strip() == "":
        j -= 1
    return lines[i:j]


def split_answer(question, answer):
    """Return (prefix_lines, resolution_lines, suffix_lines) or None if the answer does not keep the context verbatim.

    Leading/trailing blank lines are ignored on both sides because the scorer strips the whole block anyway.
    """
    s, e = _snippet_bounds(question)
    lines = _trim_blank(question[s:e].split("\n"))
    span = _conflict_span(lines)
    if span is None:
        return None
    o, c = span
    prefix, suffix = lines[:o], lines[c + 1:]
    ref = _trim_blank(answer.split("\n"))
    if len(ref) < len(prefix) + len(suffix):
        return None
    if ref[:len(prefix)] != prefix or (suffix and ref[len(ref) - len(suffix):] != suffix):
        return None
    return prefix, ref[len(prefix):len(ref) - len(suffix)], suffix


def target_full(answer):
    return "```java\n" + answer + "\n```"


def target_resolution(question, answer):
    sp = split_answer(question, answer)
    if sp is None:
        return None
    if not sp[1]:
        return "```java\n```"  # empty (conflict block deleted); one blank line is "```java\n\n```"
    return "```java\n" + "\n".join(sp[1]) + "\n```"


def rebuild_from_resolution(question, model_text):
    """Model output (fenced resolution) -> fenced full snippet, for scoring with scoring.score. None if unparseable."""
    s, e = _snippet_bounds(question)
    lines = _trim_blank(question[s:e].split("\n"))
    span = _conflict_span(lines)
    if span is None:
        return None
    import re
    m = re.search(r"```[^\n]*\n(.*?)\n```", model_text, re.DOTALL)
    if m is not None:
        res = m.group(1).split("\n")  # NOT stripped: indentation matters; "```\n\n```" is ONE blank line
    elif re.search(r"```[^\n]*\n```", model_text):
        res = []  # "```java\n```": the conflict block is deleted
    else:
        return None
    o, c = span
    full = "\n".join(lines[:o] + res + lines[c + 1:])
    return "```java\n" + full + "\n```"
