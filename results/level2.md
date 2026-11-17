# Level 2: training data

Reproduce: `scripts/fetch_data.sh && .venv/bin/python scripts/prepare_data.py && .venv/bin/python scripts/build_level2.py`
(every number below is in `results/level2.json`). Built with Claude Fable 5.2's review (see "Review history").

## Source
The Merge-Bench release already ships the paper's **3,424 Java training hunks** (`repos_reaper_java_train`), so no
Merge-Bench-Builder run was needed. The vendored builder takes train from `repos_reaper_Java_0_1000.csv` and test from `_1000_1200.csv`.

## Grouping without repository ids
The data has no repo id, but every conflict label ends in the **40-hex SHA of the merged commit** (`>>>>>>> <sha>`; 759 distinct in
train, 196 in test), and many hunks carry a `package` line or a path label. Groups are connected components of hunks that share a merge
SHA or a package root (first 3 components for `com/org/net/...`, else 2; `java`, `javax`, `android`, `junit` ignored). (A first attempt
grouped by shared rare identifiers; review showed it dropped exactly the identifiers that define big repos, so 253 of 300 dev hunks
shared a merge with train. It was replaced.)

## Cleaning and split
| Step | Hunks left |
|---|---|
| Shipped train split | 3,424 |
| Drop exact duplicate prompts (74) | 3,350 |
| Cap repeats of the same conflict region at 3 (one licence-header conflict appears 270 times) | 2,976 |
| Drop train hunks with a test-set merge SHA (0), package root (0) or near-duplicate conflict region, 8-gram Jaccard >= 0.6 (58) | 2,918 |
| Hold out dev: whole groups of <= 30 hunks, seeded | 304 dev |
| Cap any single merge at 120 train hunks (drops 183) | **2,431 train** |

Groups: 557 (largest 696 hunks, so big repos stay whole and are never split between train and dev). Dev spans 172 merges; its largest
merge is 4.9% of dev. Gate (>= 2,000 clean hunks): **passed (2,431)**.

## Leakage report (all independent of the grouping step)
Shared merge SHAs: dev-train 0, train-test 0, dev-test 0. Shared package roots: dev-train 0, train-test 0, dev-test 0. Identical
conflict region train-test: 0. Near-duplicate removal above: 58. Not controllable: both models may have seen these projects in pretraining.

## Is dev representative of test? Not in difficulty or orientation
| Exact % | train | dev | **dev_sym** | test |
|---|---|---|---|---|
| take left | 43.0 | 32.2 | 27.2 | 33.7 |
| take right | 20.4 | 22.0 | 27.2 | **45.7** |
| both | 10.9 | 14.8 | 14.8 | 20.2 |
| solved by none of left/right/both/both-reversed | 32.7 | 39.1 | | 17.0 |

Orientation is a property of the *merge*, not the dataset: the large merges are almost entirely one-sided. Train: merges of 120 hunks
at 98% left, 120 at 100% left, 75 at 84% right. Test: a 100-hunk merge at 98% left, a 78-hunk merge at 100% right, 34 at 91% right and
31 at 100% right. Test happens to hold more right-leaning merges. Train also has 35.4% path-style labels
(`<<<<<<< HEAD:src/...`) vs 2.0% in test, and is harder. Dev is harder than test (39.1% vs 17.0% unsolved by any trivial rule), so
**dev scores will read lower than test scores**; compare each model with the trivial floor on the same set.

## Decisions carried into the later levels
1. **Train side-agnostic.** `scoring.swap_example` swaps the two sides' contents (markers stay in place). If the developer answer is
   exactly left+right (or right+left), it is re-derived in the new order, so the target stays "concatenate in the order shown"; all other
   answers are unchanged. `can_swap` refuses hunks with nested or duplicated markers. Level 3 trains each hunk in one random orientation (p = 0.5 swapped).
2. **Orientation-neutral dev.** `data/dev_sym.jsonl` = 304 dev hunks + their swapped copies (607 items; 21 swapped answers re-ordered).
   Take-left = take-right = **27.2%** there; Level 3's "beat the trivial floor" gate uses this number.
3. **Level 4 needs a cluster-aware test.** Test hunks cluster in 196 merge SHAs with near-constant orientation; resample by merge SHA
   (cluster bootstrap), not by hunk.
4. Level 3's "close a third of the gap to K2" gate compares across distributions (K2 was measured on the test sample). Level 3 therefore
   also measures untuned Qwen and K2 low effort on `dev_sym` and states which comparison each gate uses.

## Review history
Claude Fable 5.2 review of the first version found: dev/train leakage through shared merges (fixed by SHA + package-root grouping), heavy
within-train duplication (region cap + per-merge cap), swapped copies of pure-concatenation answers being wrong targets (fixed in
`swap_example`), and unreproducible numbers in this write-up (replaced by script-produced ones). The identifier-overlap
"leakage guard" was dropped (mostly false positives such as `SOFTWARE`).
