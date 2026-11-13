"""Export Merge-Bench Java data to jsonl, pick the fixed Level 1 sample, and run Level 0 checks.

Usage: .venv/bin/python scripts/prepare_data.py
"""
import json, random, sys
from pathlib import Path
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scoring import score, summarize, trivial_resolve  # noqa: E402

SEED = 20261113
SAMPLE_N = 200
base = ROOT / "data/merges/merges"
test = load_from_disk(str(base / "repos_reaper_java/dataset"))["test"]
train = load_from_disk(str(base / "repos_reaper_java_train/dataset"))["train"]

def dump(ds, path):
    with open(path, "w") as f:
        for i, ex in enumerate(ds):
            f.write(json.dumps({"id": i, "question": ex["question"], "answer": ex["answer"]}) + "\n")

dump(test, ROOT / "data/test806.jsonl")
dump(train, ROOT / "data/train3424.jsonl")

ids = sorted(random.Random(SEED).sample(range(len(test)), SAMPLE_N))
(ROOT / "results/sample200_ids.json").write_text(json.dumps(ids))
print(f"test={len(test)} train={len(train)} sample={len(ids)} seed={SEED}")

# ---- Level 0 verification: reference -> 100%, untouched -> 0% ------------
ref = [score("```java\n" + ex["answer"] + "\n```", ex["answer"]) for ex in test]
untouched = [score(trivial_resolve(ex["question"], "untouched"), ex["answer"]) for ex in test]
print("reference-fed  :", summarize(ref)["exact_pct"], "% exact (must be 100)")
print("untouched-fed  :", summarize(untouched)["exact_pct"], "% exact (must be 0)")

# ---- trivial baselines on full test set and on the sample ---------------
rows = []
for name in ["untouched", "left", "right", "both", "both_reversed"]:
    for label, idxs in [("test806", range(len(test))), ("sample200", ids)]:
        b = [score(trivial_resolve(test[i]["question"], name), test[i]["answer"]) for i in idxs]
        s = summarize(b); rows.append((name, label, s))
print(f"\n{'strategy':14}{'set':10}{'exact%':>8}{'+norm%':>8}{'conflict%':>10}{'wrong':>7}{'no_block':>9}")
for name, label, s in rows:
    print(f"{name:14}{label:10}{s['exact_pct']:8.1f}{s['exact_or_normalized_pct']:8.1f}{s['conflict_pct']:10.1f}{s['wrong']:7}{s['no_block']:9}")
json.dump([{"strategy": n, "set": l, **s} for n, l, s in rows], open(ROOT / "results/level0_trivial.json", "w"), indent=1)
