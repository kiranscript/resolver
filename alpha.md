# Alpha: can a fine-tuned 2B model beat K2 Horizon 3.7B at merge conflicts?

Plan written 2026-11-13. Nothing here has been run yet.

## The claim we are trying to earn

A fine-tuned **Qwen3.5 2B** beats untuned **K2 Horizon 3.7B** at resolving Java merge conflicts, under these rules:

- Both models quantized to 4-bit.
- Qwen runs with thinking off and fits in under 2 GB of memory.
- K2 is given its best shot: we must beat its best row, including thinking on at its recommended high reasoning effort.
- The result states plainly that one model is fine-tuned and the other is not.

## Benchmark

**Merge-Bench** (Schesch & Ernst, 2026), Java subset.

| Item | Value |
|---|---|
| Java test set | 806 conflict hunks from 44 repositories |
| Full benchmark | 7,938 hunks, 1,439 repositories, 11 languages |
| Input | diff3-style conflict (left, base, right) plus 20 lines of context each side |
| Size limits | Each side at most 20 lines; conflict at most 512 tokens |
| Headline metric | Exact match with the developer's resolution |
| Secondary metric | Match after stripping comments and normalising whitespace |
| Code | [Merge-Bench](https://github.com/benedikt-schesch/Merge-Bench), [Merge-Bench-Builder](https://github.com/benedikt-schesch/Merge-Bench-Builder), [LLMerge](https://github.com/benedikt-schesch/LLMerge) |

Why not ConflictBench: it has only 180 cases, too few to tell a real win from noise.

Reference scores from the paper (Java, exact match):

| Model | Exact match | Left the conflict unresolved |
|---|---|---|
| Gemini 2.5 Pro | 54.7% | 3.4% |
| LLMergeJ 14B (reinforcement learning, 4 days on an H200) | 48.8% | 5.6% |
| Qwen3 14B, plain fine-tune on 3,424 hunks | 36.6% | 19.2% |
| Qwen3 14B, untuned | 12.9% | 74.7% |
| Qwen3 8B, untuned | 5.5% | 86.1% |
| R1 1.5B, untuned | 0.0% | 44.0% |

What this suggests: untuned small models mostly decline to resolve, and plain fine-tuning roughly triples the score. K2 Horizon's score on this benchmark is unpublished, so Level 1 decides whether the opening is real.

## Resources

- **Kaggle:** CLI 2.2.5 installed with credentials. Roughly 30 GPU-hours per week; session cap is 9 or 12 hours depending on the source. Read the real figures from the account before the first push.
- **Local:** RTX 3050 laptop GPU with 4 GB. Enough for smoke tests and the under-2 GB memory check, not for training.
- **Budget for this plan:** about 8–9 GPU-hours, one day of wall-clock time.

## Method: breadth-first with gates

Each level runs its branches side by side. A branch that fails its gate is dropped. Everything that passes carries forward. No level starts until the one before it has passed its check.

Every level has four parts:

- **Do:** the work.
- **Verified by:** how we know the step itself worked.
- **Criticism:** the strongest objection to trusting the result, and how we answer it.
- **Prune if:** the condition that kills the branch or the whole slice.

---

## Level 0: Is the benchmark sound?

Local, about 1 hour, no GPU.

- **Do:**
  - Clone the three repositories and locate the 806 Java test hunks.
  - Read the repository licenses (the paper states none for the data).
  - Score the trivial answers: take left, take right, take base, concatenate both, leave the conflict.
- **Verified by:** the scorer gives 100% when fed the reference answers, and 0% exact match when fed the untouched conflict.
- **Criticism:** a model could "win" by learning to always take one side. The trivial scores set the floor that any real result must clear.
- **Prune if:** the data or scorer is unusable, or the license forbids this use. Fallback: ConflictBench, or pivot to autocomplete.

**Output:** a table of trivial-baseline scores and a working local scorer.

## Level 1: Do both models run, and what is the gap?

Kaggle, about 2–3 hours.

- **Do:**
  - Smoke test: 5 examples per model at 4-bit, to confirm each loads and generates.
  - Baseline on a fixed 200-hunk Java sample, three rows:
    1. Qwen3.5 2B, thinking off
    2. K2 Horizon, thinking off
    3. K2 Horizon, high reasoning effort (its recommended setting)
  - A second prompt variant for K2 that forbids leaving the conflict unresolved.
- **Verified by:** reading 20 raw outputs per row by hand, not only the scores. Confirms the scorer is parsing answers correctly and the models are attempting the task.
- **Criticism:** beating a model that mostly declines to answer is hollow. K2's best row across both prompts and both thinking settings is the number to beat.
- **Prune if:**
  - Either model will not load in standard tooling. Both use new architectures, so this is a real risk.
  - K2 with thinking off already exceeds about 40% exact match. The gap is then too large for one day; switch slice.

**Output:** the target number, and the gap Qwen must close.

## Level 2: Training data

CPU only, runs alongside Level 1.

- **Do:**
  - Check whether the LLMerge repository ships the paper's 3,424 Java training hunks. If so, use them.
  - Otherwise build Java hunks with Merge-Bench-Builder from repositories outside the benchmark.
  - Hold out 300 hunks, split by repository, as a dev set.
- **Verified by:**
  - Zero repository overlap between training data and the 806 test hunks.
  - A hash check on conflict text to catch duplicates from forks and vendored code.
  - Trivial-baseline scores on the dev set are close to those on the test set, so the dev set is representative.
- **Criticism:** the paper picked its fine-tuning settings on the test set and admits this favours its result. We tune only on the dev set and touch the test set once, at Level 4.
- **Prune if:** fewer than 2,000 clean hunks survive filtering.

**Output:** training set, dev set, and a leakage report.

## Level 3: Small fine-tunes

Kaggle, about 4 hours.

- **Do:** three or four short QLoRA runs on Qwen3.5 2B, thinking off, one epoch each, differing in one thing at a time:
  - Output format: full snippet with context (the paper's format) vs resolution only, with the snippet rebuilt by script
  - Adapter size: small vs large rank
  - Learning rate: two values
- **Verified by:** dev-set score for each run, compared with untuned Qwen and the trivial floor from Level 0.
- **Criticism:** a gain could come from format compliance alone, not better merging. Every result is split into "resolved correctly", "resolved wrongly" and "declined", so we can see which bucket moved.
- **Prune if:**
  - A run does not beat the trivial floor: drop that run.
  - The best run closes less than a third of the gap to K2's best row: drop the slice.

**Output:** one winning configuration, or a decision to stop.

## Level 4: The verdict

Kaggle, about 2 hours.

- **Do:**
  - Run the winning adapter once on all 806 Java test hunks.
  - Run K2's best configuration on the same 806.
  - Measure Qwen's peak memory at 4-bit.
- **Verified by:** a paired significance test on the 806 per-hunk results, so a small difference is not reported as a win.
- **Criticism:** both models may have seen these repositories in pretraining. This cannot be removed, only stated alongside the result.
- **Outcome, one of:**
  - **Win or near-win:** scale up (more data, more seeds, other languages).
  - **Clear loss with a closing gap:** one more round at Level 3 with more data.
  - **Clear loss, flat:** pivot to autocomplete.

**Output:** a results table for both models, the significance test, the memory figure, and the decision.

---

## Time and quota

| Level | Where | Time | GPU-hours |
|---|---|---|---|
| 0 | Local | 1 h | 0 |
| 1 | Kaggle | 2–3 h | 2–3 |
| 2 | Local or Kaggle CPU | alongside Level 1 | 0 |
| 3 | Kaggle | 4 h | 4 |
| 4 | Kaggle | 2 h | 2 |
| **Total** | | **about 1 day** | **8–9** |

## Known risks

- **Tooling support.** Qwen3.5 uses a hybrid attention layout and K2 Horizon ships its own training code. Either may need a newer library version than Kaggle's default image. Level 1's smoke test catches this first.
- **K2's size.** Its name says 3.7B but its Hugging Face metadata lists 5B parameters. This affects how much memory it needs, not the plan.
- **Qwen's vision encoder.** Part of its 2B parameters is not used for this task.
- **Dataset license.** Not stated in the paper; checked at Level 0.
- **Kaggle limits.** Session cap and concurrent GPU sessions need confirming from the account.

## What "done" looks like after day one

A single table with the trivial floor, untuned Qwen, fine-tuned Qwen and K2's best row on the same test hunks, plus a go, pivot or stop decision backed by the gates above.

## Sources

- [Merge-Bench paper (arXiv 2605.25890)](https://arxiv.org/html/2605.25890)
- [ConflictBench (ASE 2024 journal-first)](https://conf.researchr.org/details/ase-2024/ase-2024-journal-first-papers/22/ConflictBench-A-Benchmark-to-Evaluate-Software-Merge-Tools)
- [K2 Horizon 3.7B model card](https://huggingface.co/IFM/K2-Horizon-3.7B)
- [Qwen3.5 2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Artificial Analysis tiny open-source models](https://artificialanalysis.ai/models/open-source/tiny)
