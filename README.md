# Can a fine-tuned 2B model beat K2 Horizon 3.7B at merge conflicts?

Plan: `alpha.md`. Results per level in `results/` (each level was reviewed by a Fable subagent before its commit).

**Verdict: NEAR-WIN** (`results/level4.md`). Fine-tuned Qwen3.5-2B (4-bit, thinking off) beats untuned K2-Horizon-3.7B (4-bit, its best row) on
the 806 Merge-Bench Java test hunks: **45.0% vs 34.5% exact, +10.5 points, 95% merge-clustered CI [+5.4, +15.2]**. One model is fine-tuned, the other is not.
Caveats: it only ties the trivial "always take the right side" rule (45.7%); part of the gap is that Qwen emits only the resolution lines while
K2 copies the context (+8.2 on marker-free hunks); memory is 1.93-1.94 GB at steady state but 2.17 GB on a cold first call on a T4 (8.5% over 2 GB).

| Level | Result |
|---|---|
| 0 benchmark sound | Scorer verified (100% / 0%, 12,090-case agreement with Merge-Bench); take-right floor 45.7% on 806 hunks |
| 1 baselines | K2 best = high effort + no-decline prompt (31.5% on the 200-hunk sample); untuned Qwen 14.5%; both load at 4-bit |
| 2 data | 2,431 train / 304 dev hunks grouped by merge SHA and package root, zero overlap with test; orientation-neutral dev_sym |
| 3 fine-tunes | 4 QLoRA runs, 32-35% on dev_sym vs trivial 27.2%, K2 17.0%, untuned 13.5%; winner r2 (resolution format) |
| 4 verdict | Near-win, see above |

## Reproduce
`python -m venv`/`uv venv .venv`; `uv pip install -r requirements.txt peft flash-linear-attention`; `scripts/fetch_data.sh`;
`scripts/prepare_data.py` (Level 0); `scripts/build_level2.py`; Kaggle kernels are built by `kaggle/level1/build.py` and `kaggle/level3/build.py`
(private Kaggle datasets `resolver-data` and `resolver-adapter` hold the data and the r2 adapter; raw generations stay local and gitignored);
reports: `scripts/level1_report.py`, `scripts/level3_report.py`, `scripts/level4_report.py`.
