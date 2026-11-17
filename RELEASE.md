# Release checklist

Status of the pre-launch blockers. "Owner" items need an account, a decision or an email that the repo cannot do by itself.

| # | Blocker | Status |
|---|---|---|
| 1 | LICENSE | Done: MIT (`LICENSE`). |
| 2 | Model licenses | Checked against the model cards: Qwen3.5-2B is Apache-2.0 (full license file read, no extra terms). K2-Horizon-3.7B is tagged Apache-2.0 on its card but its `LICENSE` file returned 404, so **owner: re-check that before launch**. Neither states any restriction on adapters or published comparisons. Recorded in `release/MODEL_CARD.md`. |
| 3 | Data licensing | Hunk files are not tracked (`data/` and `kaggle/dataset/` are gitignored); `scripts/fetch_data.sh` rebuilds them from pinned SHAs and a sha256. `results/level4_per_hunk.csv` has ids, merge SHAs and outcomes only, no code. **Do not re-host the hunk files.** Owner: email the Merge-Bench author before launch (also the place to share the take-right observation). |
| 4 | Private Kaggle datasets | Adapter is ready to publish (`kaggle/adapter_ds/`, 60 MB) with a card in `release/MODEL_CARD.md`. **Owner: upload it to Hugging Face** and either make the Kaggle kernels pull from there or make `resolver-adapter` public. The `resolver-data` dataset holds the hunk files and should stay private. |
| 5 | Install and predictions | `requirements.txt` now pins `peft` and `flash-linear-attention`. `scripts/level4_report.py` writes `results/level4_per_hunk.csv`; `scripts/verify_from_csv.py` re-derives the CIs and McNemar numbers from it with no data or GPU. Raw generations stay local (they contain third-party code). |
| 6 | Tidy | `__pycache__` removed; README reviewer wording now "Claude Fable 5.2". The untracked, gitignored `.venv523/` (5.8 GB) was left alone: it never reaches the repo. |

Also done: `scripts/run_one.py` + `examples/conflict.java.txt` (one-hunk quickstart, tested on an RTX 3050).

Tag after the first public push: `git tag v0.1-level4`.
