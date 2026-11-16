# Level 4: the verdict

Pre-registered rule: `results/level4_preregistration.md` (committed before any test run of the fine-tuned model).
Numbers: `results/level4.json`, `results/level4_memory.json`; script: `scripts/level4_report.py`. The test set was scored once per model.

## The claim, as tested
Fine-tuned **Qwen3.5-2B** (adapter r2, 4-bit NF4 base, thinking off) vs untuned **K2-Horizon-3.7B** (4-bit NF4; its best row: high
reasoning effort, no-decline prompt B, card sampling, most generous answer reading; within Merge-Bench's output format, see "Format advantage" below) on all **806 Merge-Bench Java test hunks**.
**One model is fine-tuned on 2,409 training hunks of this task (2,431 clean hunks, 22 not representable in the resolution format), the other is not.** Both may have seen these repositories in pretraining.

## Results (exact match, n = 806, 196 merge clusters)
| Row | Exact | + normalised | Left conflict* | Wrong | No code block |
|---|---|---|---|---|---|
| **Qwen3.5-2B fine-tuned (r2)** | **45.0%** | 53.2% | 6.8% | 322 | 0 |
| K2 high effort, prompt B (strict first block) | 34.0% | 45.8% | 0.7% | 418 | 13 |
| **K2 high effort, prompt B (lenient)** = K2's best | **34.5%** | 46.3% | 0.7% | 418 | 9 |
| K2 low effort, prompt B (lenient) | 28.5% | 39.8% | 1.2% | 430 | 45 |
| Qwen3.5-2B untuned, prompt B | 17.0% | 23.2% | 18.6% | 444 | 25 |
| trivial: take right | 45.7% | 53.0% | 4.7% | 341 | 0 |
| trivial: take left | 33.7% | 42.2% | 7.6% | 405 | 0 |

*Merge-Bench's marker-substring bucket. It over-counts: ~12% of reference answers themselves contain marker text (see Level 0); for the fine-tuned model 48 of its 55 "conflict" outputs are of that kind and `declined_strict` is 0.
K2 high effort: the 200 Level 1 sample hunks come from that run; the other 606 were run with identical settings in two kernels.
K2 scores 30.0%/31.5% on the Level 1 sample, 32.3% on the first half and 38.6% on the second: hunk difficulty varies a lot between subsets, and all models rise together (Qwen fine-tuned 39.5 / 45.2 / 48.5, K2 low effort 24.5 / 28.1 / 31.7; reviewer check, hunk-level permutation p = 0.07 for the 38.6% half).
K2 high effort hit the 4,096-token cap on 159 hunks (re-run at 16,384); 3 are still cut off at 16,384.

## Paired comparisons (95% bootstrap CI resampling merge SHAs; exact McNemar on hunks)
| Comparison | Difference | 95% CI | Discordant (a only / b only) |
|---|---|---|---|
| **Qwen fine-tuned - K2 best (lenient)** | **+10.5 pts** | **[+5.4, +15.2]** | 185 / 100, McNemar p < 0.0001 |
| Qwen fine-tuned - K2 best (strict) | +11.0 | [+6.0, +15.8] | 187 / 98 |
| Qwen fine-tuned - K2 low effort | +16.5 | [+11.4, +20.8] | |
| Qwen fine-tuned - untuned Qwen | +28.0 | [+23.1, +32.3] | 256 / 30 |
| **Qwen fine-tuned - take right** | **-0.6** | **[-17.8, +17.6]** | 185 / 190, p = 0.84 (tie) |
| Qwen fine-tuned - take left | +11.3 | [-4.9, +25.6] | |
| take right - K2 best | +11.2 | [-4.1, +25.5] | |

## Memory (pre-registered definition: batch-1 peak allocated, longest test prompt, 527 tokens)
| | GB (1e9 B) |
|---|---|
| 4-bit weights (text-only) | 1.79 |
| + LoRA adapter | 1.85 |
| Repeated-prompt generation peak, RTX 3050 and Kaggle T4 | **1.93-1.94** (passes < 2 GB, 3% margin) |
| First generate call in a fresh process, Kaggle T4 (the scored run) | **2.17** (misses by 8.5%) |
| Chunked-prefill pass later in the same T4 process | 2.16 |
| Whole-process GPU memory incl. CUDA context (`nvidia-smi`, MiB converted) | 2.09 GB local (1,997 MiB), 2.66 GB Kaggle (2,539 MiB) |

The tied fp16 embedding/lm_head is ~1.02 GB of the weights, so the 2 GB limit is mostly a vocabulary-size problem.
Staged Kaggle diagnostic (`kaggle/level4/out/l4mem`): the 2.170 GB first-call figure reproduces to the byte, its cause is **unidentified**
(a ~0.3 GB transient above the load), repeating the same prompt peaks at 1.942 GB, but a chunked-prefill pass with new shapes late in the
process reaches 2.162 GB again, so the transient may recur for new input shapes. Steady state was verified only for a repeated prompt
(locally, varied 201-527-token prompts all peak at 1.92-1.94 GB). Explicit `logits_to_keep=1` changed nothing.
"Under 2 GB" therefore holds for PyTorch-allocated memory where measured in steady state, not for the cold first call on a T4, and not for
the whole process (over 2 GB in decimal units even locally).

## Verdict (pre-registered rule, conservative reading: every batch-1 measurement counts)
**NEAR-WIN.** Accuracy leg: clear win, +10.5 points over K2's best row, CI excluding zero, p < 0.0001. Memory leg: 1.94 GB steady
state where verified but 2.17 GB on a cold first call on the T4, a miss of 8.5% (the rule's near-win band is < 10%). Relabelling this as a win on steady state would be post hoc.
The plan's decision for a win or near-win is to **scale up** (more data, more seeds, other languages).

## What the win does and does not mean
- **It is a real, significant advantage over K2** under Merge-Bench's protocol: K2's best of 6 rows, lenient reading, 4-bit, prompt B. K2 was *not* given the resolution-only format (see below), so "K2 got its best shot" holds for the benchmark's own format only.
- **It is not a win over the trivial rule.** Always taking the right side scores 45.7% on these hunks, a statistical tie with the
  fine-tuned model (-0.6 pts, CI [-17.8, +17.6]), and K2 does not beat it either.
- **By gold hunk type (exact %):** right-only 368 hunks: Qwen 48.4, K2 44.0, take-right 100; left-only 272: Qwen 58.1, K2 36.0,
  take-right 0; concatenation 29: Qwen 55.2, K2 34.5; novel (no trivial rule solves it) 137: Qwen 8.0, K2 5.8, take-right 0.
  On the 640 one-sided hunks Qwen is exactly right 52.5% of the time; 82 of its outputs there are neither side, and among the 558 actual
  side picks 336 (60.2%) are the developer's side (66% on left-only, 56% on right-only): better than chance but weak side selection
  (Level 3 found near-chance on dev). It ties take-right only because 57% of the one-sided test hunks are right-only. Its edge over K2 is
  mostly producing a clean one-side resolution reliably (no off-script output, no context errors), not deeper merge understanding.
  Novel resolutions are rare for both models (8.0% vs 5.8% of 137).
- **Why train/dev/test differ.** Test is dominated by right-leaning merges; training on both orientations deliberately removed that prior, so
  the model cannot collect the "take right" bonus. A model that learned the orientation of each repo's merges would score higher here but
  would not generalise; we did not do that.

## Format advantage (found in review; the pre-registered metric was followed and the verdict stands)
Qwen's output is only the replacement lines; a script re-inserts the ~40 context lines verbatim. K2 must reproduce all of them and is marked
wrong for any context edit. 85 of the 806 references keep stray conflict markers in the context: Qwen gets 27 of them, K2 gets 1 (K2 on
prompt A scored 0/23 on the sample's marker hunks too). On the 721 marker-free references the gap is **+8.2 points, CI [+2.6, +13.4]**: still
significant. Scoring K2 on its resolution region only (reviewer heuristic; imperfect) gives a gap of +1.9 to +5.1 (CI including 0) with
1-3 anchor lines and +7.6 to +8.3 (CI excluding 0) with 5, so **part of the headline gap is the output format, not merging skill**.
On exact-plus-normalised the gap is +6.9, CI [+2.7, +11.2]. Counting every remaining K2 no-block answer as correct would still leave
+9.4 [+4.6, +14.1]. One merge cluster of 100 hunks contributes 20 of the 85-hunk margin. K2 with the resolution-only format was not run.

## Caveats (fixed in advance or found since)
- Fine-tuned vs untuned (stated in the claim). Single run, single seed per model; K2 sampled once at T = 1.0.
- Both models 4-bit NF4 with fp16 compute; K2 was trained in bf16 and its 4-bit score is probably a little below full precision (not measured).
- K2's thinking budget: 4,096 tokens first pass, 16,384 for truncated hunks; the card recommends >= 32,768 (3 hunks remain cut off at 16,384).
- Merge-cluster CIs resample merges, not repositories (slightly optimistic).
- The test set is a fixed 806-hunk sample from 44 repositories; results on other languages or repositories are untested.

## Next (scale-up, per the plan)
More training data (build more Java hunks with Merge-Bench-Builder), 2-3 epochs and seeds, and a cheaper memory path:
trimming the 248k-token vocabulary is the one memory fix worth trying (an ASCII-only vocabulary of 127,733 tokens would save ~0.49 GB, putting the
steady peak near 1.45 GB and the T4 first call near 1.7 GB; a train-derived vocabulary does not work, 795 of 806 test hunks contain an unseen token;
14 test hunks contain non-ASCII tokens and need a fallback). Validate by byte-identical greedy outputs on dev_sym, then re-scoring only the
differences on test; report it as a post-hoc addendum, not a change to this verdict. A varied-prompt warm-up diagnostic on the T4 should come first.
