# Level 3: small fine-tunes

Code: `kaggle/level3/` (run.py trains + evaluates, build.py makes the kernel; data and code reach Kaggle via the private dataset
`kiranscript/resolver-data`), formats in `formats.py`, report `scripts/level3_report.py` (writes `results/level3.json`).
Raw generations and adapters stay local in `kaggle/level3/out/grid/` (gitignored).

## Setup
- Qwen3.5-2B, thinking off (`enable_thinking=False`), 4-bit NF4 base (fp16 compute), LoRA on attention, linear-attention and MLP
  projections (`q/k/v/o_proj, in_proj_qkv, in_proj_z, out_proj, gate/up/down_proj`), alpha = 2 x rank, dropout 0.05.
- Data: `train_clean` (2,431 hunks; Level 2), **one random orientation per hunk** (side-swapped with p = 0.5; pure-concatenation answers
  re-ordered), **one epoch**, effective batch 16, cosine schedule with 5% warm-up, AdamW (cosine to 0.1 x lr), fp16 autocast, loss on the answer tokens only.
  The "resolution" format drops 22 hunks whose answer does not keep the context verbatim (2,409 examples).
- Prompt: **format `full`** uses Level 1's prompt B verbatim (same wording the baselines got); **format `resolution`** uses its own
  instruction text (`formats.RES_HEAD`, also no-decline). **Format `full`** = rewrite the whole snippet (paper's
  format). **Format `resolution`** = write only the lines replacing the conflict block; a script rebuilds the snippet (round-trip
  verified: 806/806 test answers, 2,409/2,431 train answers).
- Scored on `dev_sym` (607 items = 304 dev hunks + side-swapped copies), greedy decoding, same scorer as Levels 0-1.
  One factor changed at a time from run r1 (full, rank 16, lr 2e-4).

## Results (exact match, dev_sym, n = 607; 172 merge clusters)
| Row | Exact | + normalised | Declined (strict) | Wrong | No code block | Train time |
|---|---|---|---|---|---|---|
| trivial take-left = take-right | 27.2% | 36.4% | | | | |
| untuned Qwen3.5-2B (prompt B) | 13.5% | 22.4% | 15.7%* | 346 | 30 | |
| K2 low effort (prompt B) | 17.0% | 27.2% | 0.3%* | 413 | 27 | |
| r1 full, rank 16, lr 2e-4 | 33.4% | 41.7% | 0.0% | 339 | 2 | 31 min |
| **r2 resolution, rank 16, lr 2e-4** | **34.9%** | **45.6%** | 0.2% | 315 | 0 | 26 min |
| r3 full, rank 64, lr 2e-4 | 33.8% | 44.0% | 0.0% | 327 | 0 | 26 min |
| r4 full, rank 16, lr 5e-4 | 32.1% | 40.2% | 0.0% | 349 | 0 | 26 min |

*Merge-Bench's marker-substring "conflict" bucket; the fine-tunes' strict-declined column is from `declined_strict`.
K2 high effort was not run on dev_sym (about 8 GPU-hours); on the Level 1 test sample high B was 1.19x low B (31.5 vs 26.5).

Paired differences, 95% bootstrap CI resampling **merges** (not items):
| Comparison | Difference (points) | 95% CI |
|---|---|---|
| r2 - trivial | +7.7 | [+2.4, +12.8] |
| r2 - K2 low effort | +18.0 | [+13.5, +22.5] |
| r2 - untuned Qwen | +21.4 | [+16.5, +26.5] |
| r2 - r1 (format) | +1.5 | [-3.2, +5.8] |
| r3 - r1 (rank 64 vs 16) | +0.3 | [-3.5, +4.2] |
| r4 - r1 (lr 5e-4 vs 2e-4) | -1.3 | [-4.9, +2.3] |

## Gates
- Every run beats the dev trivial floor (27.2%): **passed** for r1-r4.
- Best run closes >= 1/3 of the gap to K2's best row: it exceeds K2's measured dev score outright (34.9 vs 17.0 low effort; even with
  a +19% allowance for high effort, ~20%): **passed**.

## Winner
**r2 (resolution format, rank 16, lr 2e-4).** Highest point estimate on both exact and normalised, shortest outputs (median 29 tokens vs
116), fastest eval (372 s vs 987 s), lowest memory. Honest caveat: the three one-factor variants are **not statistically
distinguishable** (all CIs include 0), so the choice among them is a point-estimate call; what the data supports is that
fine-tuning is the effect (+19-21 points over untuned Qwen), not the format, rank or learning rate. Rank and learning rate were
varied on the full format only.

## What the model actually learned (Claude Fable 5.2's analysis of the dev_sym generations)
- **Side selection is at chance.** Where exactly one side is the developer's answer (274 items), r2 picks the right side 117 times and the
  wrong side 121 (r1: 125/112, r3: 137/103, r4: 136/106). 84% of r2's outputs are literally take-left, take-right or concatenate.
- The +7.7 over trivial comes from hunks where one side is empty and also matches concatenation (93% exact), pure concatenations (58%),
  and a few novel resolutions (8% of the 237 hunks no trivial rule solves).
- **Gate nuance:** the 27.2% floor is the mean of take-left 32.2 and take-right 22.0 on the original-orientation dev (by construction of
  dev_sym). On the 304 original-orientation hunks alone r2 scores 36.2 vs take-left 32.2: +3.9, cluster CI [-3.1, +11.3], not
  significant. The "beats the trivial floor" gate passes as defined (against the orientation-neutral floor), not against the best
  single-orientation rule.
- **Level 4 expectation.** Re-weighting dev per-type accuracy to the test mix (62% of test hunks are exclusive-side) predicts roughly
  46% for r2 on test: far above K2-high (31.5% on the Level 1 sample) and about tied with take-right (45.7%). The same method
  under-predicts K2-low (22.4 predicted vs 26.5 measured), so this is a rough guide only. **Pre-registered for Level 4:** the take-right
  rule is reported next to the result and the claim is only about K2.
- r2 is weak at deleting a conflict block (8 of 56 exact when the gold resolution is empty vs 22 of 56 for r1); test has 77 such hunks.

## Caveats
- dev_sym is harder than test and orientation-neutral; absolute numbers will differ on test, where "take right" scores 45.7%. The
  test comparison is Level 4's job; the dev set was used to pick r2 and nothing here touched the test set.
- Dev has 304 distinct hunks (the swapped copies are not independent), hence the merge-clustered CIs.
- One run per config (no seeds): differences of a couple of points are within run-to-run noise.
- Selection among 4 runs on dev inflates the best score slightly (bootstrap max-of-4 35.7 vs mean 33.6); expect r2's true dev score
  around 33.5-35. Immaterial next to the +18 over K2-low.
- Dev vs train near-duplicates were only filtered train-vs-test in Level 2: 30 dev_sym items (15 hunks) have a conflict region with
  8-gram Jaccard >= 0.6 to some train hunk, and some repos span both sides (e.g. BuildCraft: 45 train hunks, 4 dev items; the
  package-root rule misses sibling packages). Dropping the union (130 items): r2 35.4, r1 33.3, trivial 27.0 on the remaining 477;
  conclusions unchanged.
- CIs resample merges, not repositories, so they are slightly optimistic. The K2 comparison uses K2 low effort on dev; r2 beats
  1.19 x K2-low by +14.7 [9.9, 19.5] and 1.5 x K2-low by +9.5 [4.3, 14.6] (Claude Fable 5.2's check), so the gate is robust to the unmeasured high-effort row.
- 3 of the 607 dev items are not representable in the resolution format (counted as failures for r2, about -0.5 points).
- Training peak memory 4.2-8.2 GB on a T4 is for training; the memory claim concerns inference (Level 4).
