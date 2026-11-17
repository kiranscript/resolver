# Level 1: do both models run, and what is the gap?

Date: 2026-11-13/14. Fixed 200-hunk Java sample (`results/sample200_ids.json`, seed 20261113), 4-bit NF4 (bitsandbytes 0.50.2,
fp16 compute) on Kaggle T4 x2, Transformers 5.23.0 for the three full runs (the 6-hunk smoke runs used 5.22.2; later runs pin 5.23.0 and the K2 repo revision). Code: `kaggle/level1/` (build.py, main.py). Table: `scripts/level1_report.py`
(re-scores saved raw outputs; writes `results/level1.json`). Raw outputs stay local in `kaggle/level1/out/` (gitignored).

## Setup actually used
- **Prompt A** = Merge-Bench's own prompt (allows leaving the conflict). **Prompt B** = same, with the "if unsure, return the
  snippet with the conflict" sentence replaced by "you must always resolve; never return markers".
- **Qwen3.5-2B**: chat template with `enable_thinking=False`, greedy. Loads text-only (`Qwen3_5ForCausalLM`, 1.2B params),
  1.79 GB at load. Peak during generation grows with batch size: 2.1-2.3 GB at batch 3-5, **3.23 GB at batch 16** (the full runs). The "under 2 GB" claim therefore needs a stated definition and a batch-1 measurement in Level 4.
- **K2-Horizon-3.7B** (`K2HorizonForCausalLM`, 5.06B params with untied embeddings, 4.7 GB loaded): card's sampling
  (temperature 1.0, top_p 0.95). K2 has **no "thinking off"**; its template only takes `reasoning_effort` in {high, medium, low}.
  Rows: `emptythink` (generation prompt + an empty `<ifm|think></ifm|think>` prefill), `low`, `high`.
- Max new tokens: 1,280 for non-thinking rows; high effort 4,096 in pass 1, then only the truncated hunks re-run at 16,384.
  Card says >= 32,768; observed high-effort length is median ~1.7-1.9k, p90 ~3.6-3.9k tokens, 2 of 400 hunks still unfinished at 16,384.

## Results (exact match, n = 200)
| Row | Exact (strict) | Exact (lenient)** | + normalised | Declined* | Wrong | No code block |
|---|---|---|---|---|---|---|
| trivial: take left | 33.0% | | 43.5% | | | |
| **trivial: take right** | **44.0%** | | 55.5% | | | |
| Qwen3.5 2B, A | 3.0% | 3.0% | 4.0% | 86.0% | 14 | 6 |
| Qwen3.5 2B, B | 14.5% | 14.5% | 20.5% | 15.5% | 106 | 7 |
| K2 empty-think, A | 3.0% | 5.5% | 3.5% | 63.0% | 48 | 12 |
| K2 empty-think, B | 17.0% | 21.5% | 23.0% | 7.0% | 122 | 15 |
| K2 low, A | 5.5% | 5.5% | 7.0% | 78.0% | 24 | 5 |
| K2 low, B | 26.5% | 26.5% | 37.5% | 0.5% | 116 | 5 |
| K2 high, A | 23.5% | 23.5% | 33.5% | 31.5% | 60 | 8 |
| **K2 high, B** | 30.0% | **31.5%** | 43.5% | 0.0% | 107 | 6 |

*Declined = more `<<<<<<<` lines than the developer's answer (stricter than Merge-Bench's marker-substring bucket); "Wrong" and "No code block" are Merge-Bench's buckets, so a row does not sum to 200.
**Lenient (K2 only, Claude Fable 5.2's review): best of first block, last block, or, when the final answer has no code block, the last block inside its thinking. 3 of K2-high-B's 6 no-block answers had the exact developer answer in their thinking and replied in prose. We use the lenient figure as the target so K2 gets its best reading.
95% Wilson interval at n = 200: 30.0% -> [24, 37]; 26.5% -> [21, 33]; 14.5% -> [10, 20]. Differences of a few points between K2 rows are noise.

## Target and gap
- **Number to beat: K2 high effort, prompt B, 31.5% exact lenient (30.0% strict)** (best across both prompts and all effort settings).
- Untuned Qwen best: 14.5% (prompt B). **Gap: 17.0 points** (15.5 on the strict reading). The paper's Qwen3 14B plain fine-tune tripled its untuned score.
- **Trivial "take right" is 44.0% on this sample (45.7% on all 806).** It is above every model row. A fine-tuned Qwen that only
  matches K2 would still lose to this rule.

## Gates
- Both models load in standard tooling (Transformers 5.22, bitsandbytes): **passed** (K2 via `trust_remote_code`).
- K2 without thinking > ~40%? Best non-thinking K2 is 26.5% -> **keep**.

## Significance (paired, same 200 hunks; Claude Fable 5.2's review)
- K2 high B vs untuned Qwen B: 47 vs 16 discordant, p < 0.001. K2 is clearly ahead of untuned Qwen.
- **Take-right vs K2 high B: 54 vs 26 discordant, p = 0.002.** The trivial rule is significantly better than K2's best row.

## Limitations (recorded, not rerun)
- Both models are 4-bit NF4 with fp16 compute, as the claim specifies; K2 was trained in bf16 and its card recipes are unquantised, so
  K2's 4-bit score is probably a little below its full-precision score. The size of that loss was not measured.
- One sample per hunk at temperature 1.0 for K2 (card setting), best of 6 K2 rows: noise is +/- ~6 points (95% interval).
- Qwen runs greedy (not the card's sampling); greedy loops cause its 7 no-block outputs in B, worth at most ~+3 points.
- `emptythink` is not true "thinking off": K2 still reasons in prose in ~25% of outputs. Conclusion does not depend on it.

## Findings that change the plan
1. K2's decline rate drives its low scores under prompt A; prompt B is what makes both models competitive. The fine-tuned Qwen
   should be trained and scored with a no-decline format.
2. Untuned Qwen with thinking off still writes reasoning prose after its code block and hits the 1,280-token cap on 18/200 hunks.
3. Prompt B, not effort, is the big lever: at high effort A -> B is 23.5 -> 30.0 (p = 0.09). Effort itself is not significant (low B 26.5 vs high B 30.0, paired McNemar p = 0.43). High effort costs ~2.8 h of two T4s per 200 hunks under HF+bnb. Level 4 needs
   K2-high on 806 hunks: ~11 h at this speed, more than the plan's 2 h. Options: vLLM, or report K2-high on a subset (or use the
   200-hunk number plus a paired test on the 200).
4. Qwen at 4-bit is 1.79 GB at load but 2.1-2.3 GB at peak generation, so "under 2 GB" must be defined (weights vs. peak).
