# Level 0: is the benchmark sound?

Date: 2026-11-13. Reproduce with `scripts/fetch_data.sh && .venv/bin/python scripts/prepare_data.py`.

## Data
- Merge-Bench release `main`, asset `merges.tar.gz` (sha256 in `scripts/fetch_data.sh`).
- Java test split: **806 hunks**. Java train split: **3,424 hunks** (ships with the release).
- Hunks are **two-way** (`<<<<<<<` / `=======` / `>>>>>>>`), not diff3: 0 of 806 contain `|||||||`. There is no base side.
- Every hunk has exactly one complete conflict. **79 of 806** also carry unpaired markers (stray halves of neighbouring conflicts in the 20-line context window); the developer's answer keeps them. 12 test questions duplicate another test question.
- The dataset carries no repository id per hunk (only `question`, `answer`, `prompt`).

## Licenses
Merge-Bench, Merge-Bench-Builder, LLMerge: MIT. The dataset release has no separate license statement;
the hunks are excerpts of third-party open-source repositories whose own licenses apply. Treated as
fine for research use here; **this is an assumption, not a verified clearance.**

## Scorer checks (`scoring.py`)
- Fed the developer resolutions: **100.0%** exact (required: 100).
- Fed the untouched conflict: **0.0%** exact (required: 0).
- Mirrors Merge-Bench's `merged_conflict_reward` (first fenced block, exact after `.strip()`,
  else equal after dropping comments and collapsing whitespace, else conflict markers present).

## Trivial baselines (the floor)
| Strategy | Test 806 exact | + normalised | Left conflict* | Sample 200 exact |
|---|---|---|---|---|
| untouched conflict | 0.0% | 1.4% | 98.6% | 0.0% |
| take left (HEAD) | 33.7% | 42.2% | 7.6% | 33.0% |
| **take right** | **45.7%** | 53.0% | 4.7% | 44.0% |
| both, left first | 20.2% | 26.7% | 9.6% | 18.0% |
| both, right first | 18.1% | 24.2% | 9.6% | 18.5% |
| take base | not available (no base side in the data) | | | |

*Caveat on "left conflict": Merge-Bench's bucket fires on any marker substring. 93 of 806 *reference* answers contain
markers themselves (85 stray line-start markers from context, 8 `// =====` comment rulers), so this bucket over-counts
"declined" by up to ~11% of hunks. It is kept for paper comparability. `scoring.declined_strict` (more `<<<<<<<` lines than the
reference) is the stricter test; use it when analysing "declined" in Levels 1 and 3.

Fixed after review (Fable, commit 15dee37): the first resolver dropped a dangling `<<<<<<<` at the end of the snippet, which
understated take-right at 42.6% (now 45.7%). Scorer verified against Merge-Bench's real `merged_conflict_reward` on 12,090 cases
(five trivial strategies plus ten perturbations per hunk): 0 mismatches.

## Gate result
Data and scorer usable, licenses permissive: **passed**.
**Finding:** "always take the right side" scores 42.6%. That is the floor any real result must clear, and it is
above untuned Qwen3 14B (12.9%) and close to LLMergeJ 14B (48.8%) in the paper's table.
