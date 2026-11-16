# Level 4 pre-registration (written before any test-set run of the fine-tuned model)

Date: 2026-11-16. Commit this file before running Level 4.

## What will be run (once each)
1. **Fine-tuned Qwen3.5-2B = adapter r2** (resolution format, rank 16, lr 2e-4; chosen on dev_sym in Level 3, nothing re-tuned),
   greedy, thinking off, 4-bit NF4 base + unmerged LoRA, on **all 806** Java test hunks.
2. **K2-Horizon-3.7B best configuration = high reasoning effort with the no-decline prompt B** (best row in Level 1 on the 200-hunk
   sample: 31.5% lenient / 30.0% strict), 4-bit NF4, card sampling (T=1.0, top_p=0.95), on the same 806 hunks. The 200 sample hunks reuse
   the Level 1 run; the other 606 are run with identical settings.
3. Cheap reference rows on all 806: untuned Qwen (prompt B), K2 low effort (prompt B), trivial take-left / take-right.

## Primary comparison and decision rule
- Metric: exact match (Merge-Bench's headline metric), strict first-code-block rule for Qwen; for K2 the most lenient reading
  (first block, last block, or last block in its thinking) so K2 gets its best shot.
- Test: paired difference (Qwen - K2 best) in exact match over the 806 hunks, **95% bootstrap CI resampling merge SHAs** (the
  hunks are clustered in ~196 merges with near-constant orientation); also an exact McNemar test on hunks for reference.
- **Win:** point estimate > 0 and the 95% CI lower bound > 0, **and** Qwen's inference memory is under 2 GB under the definition below.
- **Near-win:** point estimate > 0 with CI including 0, or the memory limit missed by < 10%.
- **Clear loss with a closing gap / flat:** point estimate < 0 (closing vs flat judged against the Level 1 gap of 17 points and the
  Level 3 dev improvement).
- The claim is only about K2. **Take-right (45.7% on the 806) is reported next to the result and not part of the claim**; Fable's Level 3
  review predicts the fine-tuned model lands near it, and the write-up will say plainly if it does not beat it.

## Memory definition ("fits under 2 GB")
Peak `torch.cuda.max_memory_allocated` at **batch size 1** while generating the longest-prompt test hunk, with the 4-bit base,
text-only (vision encoder not loaded) and the unmerged adapter loaded; also report the load-time allocation, and process memory from
`nvidia-smi` including the CUDA context for the same run. Measured locally on the RTX 3050 and on Kaggle.

## Stated limitations, fixed in advance
Fine-tuned vs untuned (the result states this plainly); both models may have seen these repositories in pretraining; one run, no
seeds; both 4-bit NF4 with fp16 compute; K2's sampling is single-sample at T=1.0.
