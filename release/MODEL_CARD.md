---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
tags:
  - lora
  - qlora
  - merge-conflicts
  - code
  - java
---

# resolver-r2: a LoRA adapter for Java merge conflicts (Qwen3.5-2B)

A rank-16 QLoRA adapter for `Qwen/Qwen3.5-2B` (4-bit NF4 base, thinking off, greedy decoding) that writes the lines replacing a
two-way Java merge conflict (Merge-Bench format). **Research artifact, not a production merge tool.**

## Result (Merge-Bench Java test, 806 hunks, exact match)

| Row | Exact | 
|---|---|
| **This adapter (Qwen3.5-2B + r2)** | **45.0%** |
| K2-Horizon-3.7B, untuned, its best of 6 settings (4-bit, high effort) | 34.5% |
| Qwen3.5-2B, untuned | 17.0% |
| Trivial rule: always take the right side | **45.7%** |
| Trivial rule: always take the left side | 33.7% |

- vs K2-Horizon-3.7B: **+10.5 points**, 95% CI [+5.4, +15.2] (bootstrap over 196 merge commits), McNemar p < 0.0001.
- vs "take right": **-0.6 points, CI [-17.8, +17.6]. It is a tie.** The model does not beat a one-line heuristic on this test set.
- Pre-registered verdict: **NEAR-WIN** (accuracy leg met; memory leg missed on one measurement, below).

### Read this before quoting the numbers
- **Fine-tuned vs untuned.** The adapter was trained on 2,409 hunks of this task; K2 and untuned Qwen were not.
- **Part of the gap is output format.** The adapter emits only the replacement lines and a script re-inserts the context; K2 must
  copy about 40 context lines verbatim and is marked wrong for any edit. On the 721 marker-free references the gap is still
  +8.2 [+2.6, +13.4], but scoring K2 on its resolution region only shrinks it to between +1.9 and +8.3 depending on the heuristic.
  K2 was never run with the resolution-only format.
- **Weak side selection.** On the 640 one-sided hunks it picks the developer's side 60.2% of the time. On the 137 hunks that need a
  novel resolution it scores 8.0% (K2: 5.8%).
- **Memory.** 1.93-1.94 GB peak allocated at batch size 1 in steady state; **2.17 GB on the first generate call on a Kaggle T4**;
  2.09 GB (local) to 2.66 GB (Kaggle) of whole-process GPU memory including the CUDA context.
- One seed, one run, Java only, 44 repositories.

Full write-up, pre-registration and per-hunk results: <https://github.com/kiranscript/resolver> (`results/level4.md`,
`results/level4_per_hunk.csv`).

## Use

```python
# from the repo root: python scripts/run_one.py --adapter <this-repo-id> --file my_conflict.txt
```

Input: a Java snippet containing one two-way conflict (`<<<<<<<`, `=======`, `>>>>>>>`, no base side) with surrounding context. The
prompt is `formats.prompt_resolution`; the output is the replacement lines in a fenced block, merged back with
`formats.rebuild_from_resolution`.

## Training
QLoRA on 2,409 Java conflict hunks (2,431 after cleaning, 22 not representable in the resolution format), grouped by merge commit
and package root, zero overlap with the test set; both side orientations used so the model cannot learn the benchmark's
right-leaning prior. Rank 16, alpha 32, lr 2e-4. Details: `results/level2.md`, `results/level3.md`.

## Licenses and attribution
- Adapter: Apache-2.0, inherited from the base model `Qwen/Qwen3.5-2B` (Apache-2.0, Copyright Alibaba Cloud). This adapter was
  modified from and must be used with that base model.
- Compared model: `IFM/K2-Horizon-3.7B` (Apache-2.0 per its model card; used only as a baseline, not redistributed).
- Training data: hunks from the Merge-Bench dataset (Schesch and Ernst; code MIT), which are excerpts of third-party open-source
  repositories under their own licenses. The data is not redistributed here.
- Cite Qwen3.5: Qwen Team, "Qwen3.5: Towards Native Multimodal Agents", 2026. Merge-Bench: Schesch and Ernst, 2026.
