# Level 0: is the benchmark sound?

Date: 2026-11-13. Reproduce with `scripts/fetch_data.sh && .venv/bin/python scripts/prepare_data.py`.

## Data
- Merge-Bench release `main`, asset `merges.tar.gz` (sha256 in `scripts/fetch_data.sh`).
- Java test split: **806 hunks**. Java train split: **3,424 hunks** (ships with the release).
- Hunks are **two-way** (`<<<<<<<` / `=======` / `>>>>>>>`), not diff3: 0 of 806 contain `|||||||`. There is no base side.
- **50 of 806** test hunks contain more than one conflict. 12 test questions are duplicates of another test question.
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
| Strategy | Test 806 exact | + normalised | Left conflict | Sample 200 exact |
|---|---|---|---|---|
| untouched conflict | 0.0% | 1.4% | 98.6% | 0.0% |
| take left (HEAD) | 32.0% | 40.2% | 4.5% | 31.5% |
| **take right** | **42.6%** | 49.8% | 2.9% | 41.0% |
| both, left first | 19.4% | 25.8% | 5.3% | 17.0% |
| both, right first | 17.2% | 23.3% | 5.3% | 17.0% |
| take base | not available (no base side in the data) | | | |

## Gate result
Data and scorer usable, licenses permissive: **passed**.
**Finding:** "always take the right side" scores 42.6%. That is the floor any real result must clear, and it is
above untuned Qwen3 14B (12.9%) and close to LLMergeJ 14B (48.8%) in the paper's table.
