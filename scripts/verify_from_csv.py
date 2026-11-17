"""Re-derive the Level 4 headline numbers from results/level4_per_hunk.csv alone (no data, no GPU, stdlib only).

  python scripts/verify_from_csv.py

Same procedure as scripts/level4_report.py: exact match, paired difference, 95% bootstrap CI resampling merge SHAs (5,000 draws,
seed 7) and an exact two-sided McNemar test on hunks. Prints the table that results/level4.md reports.
"""
import csv, math, random
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).resolve().parent.parent / "results/level4_per_hunk.csv")))
ids = [r["id"] for r in rows]
cols = [c[:-len("__bucket")] for c in rows[0] if c.endswith("__bucket")]
bucket = {c: {r["id"]: r[f"{c}__bucket"] for r in rows if r[f"{c}__bucket"]} for c in cols}
exact = {c: {i: int(b == "exact") for i, b in d.items()} for c, d in bucket.items()}
by_cluster = {}
for r in rows:
    by_cluster.setdefault(r["merge_sha"], []).append(r["id"])
clusters = list(by_cluster.values())

print(f"{len(ids)} hunks, {len(clusters)} merge clusters\n{'row':24}{'n':>5}{'exact%':>8}")
for c in cols:
    print(f"{c:24}{len(exact[c]):>5}{100 * sum(exact[c].values()) / len(exact[c]):8.1f}")


def paired(a, b, B=5000, seed=7):
    idx = set(exact[a]) & set(exact[b])
    cl = [c for c in ([i for i in c if i in idx] for c in clusters) if c]
    rng = random.Random(seed); diffs = []
    for _ in range(B):
        pick = [rng.choice(cl) for _ in cl]
        n = sum(len(c) for c in pick)
        diffs.append(100 * sum(exact[a][i] - exact[b][i] for c in pick for i in c) / n)
    diffs.sort()
    n10 = sum(exact[a][i] > exact[b][i] for i in idx); n01 = sum(exact[a][i] < exact[b][i] for i in idx)
    k, n = min(n10, n01), n10 + n01
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    d = 100 * (sum(exact[a][i] for i in idx) - sum(exact[b][i] for i in idx)) / len(idx)
    return d, diffs[int(B * .025)], diffs[int(B * .975)], n10, n01, p


print()
for a, b in [("qwen_finetuned_r2", "k2_high_B_lenient"), ("qwen_finetuned_r2", "trivial_right"),
             ("qwen_finetuned_r2", "qwen_untuned_B"), ("trivial_right", "k2_high_B_lenient")]:
    d, lo, hi, n10, n01, p = paired(a, b)
    print(f"{a} - {b}: {d:+.1f} pts, 95% CI [{lo:+.1f}, {hi:+.1f}], only-a {n10} / only-b {n01}, McNemar p = {p:.4f}")
