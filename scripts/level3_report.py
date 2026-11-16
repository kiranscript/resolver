"""Level 3 report: re-scores the saved dev_sym generations, adds baselines and a merge-clustered paired bootstrap.

  .venv/bin/python scripts/level3_report.py        # writes results/level3.json
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scoring import score, summarize, trivial_resolve, merge_sha, declined_strict, extract_code_block  # noqa: E402
from formats import rebuild_from_resolution  # noqa: E402

dev = [json.loads(l) for l in open(ROOT / "data/dev_sym.jsonl")]
ref = {r["id"]: r for r in dev}
cluster = {r["id"]: (merge_sha(r["question"]) or f"id:{r['id']}") for r in dev}
ids = [r["id"] for r in dev]


def load_run(path, fmt):
    out = {}
    for l in open(path):
        r = json.loads(l)
        q = ref[r["id"]]["question"]
        ans = r["raw"] if fmt == "full" else (rebuild_from_resolution(q, r["raw"]) or "")
        b = score(ans, ref[r["id"]]["answer"])
        assert b == r["bucket"], (path, r["id"], b, r["bucket"])  # stored bucket must equal an independent re-score
        out[r["id"]] = b
    return out


def load_l1(path, closer=None):
    out = {}
    for l in open(path):
        r = json.loads(l)
        raw = r["raw"]
        for t in ("<|ifm|im_end|>", "<|im_end|>"):
            raw = raw.replace(t, "")
        if closer and closer in raw:
            raw = raw.split(closer, 1)[1]
        elif "</think>" in raw:
            raw = raw.split("</think>", 1)[1]
        out[r["id"]] = score(raw, ref[r["id"]]["answer"])
    return out


runs = {}
for side in ("left", "right"):
    runs[f"trivial_{side}"] = {r["id"]: score(trivial_resolve(r["question"], side), r["answer"]) for r in dev}
base = ROOT / "kaggle/level1/out/devsym/out"
runs["qwen_untuned_B"] = load_l1(base / "qwen_nothink_B.jsonl")
runs["k2_low_B"] = load_l1(base / "k2_low_B.jsonl", "</ifm|think_faster>")
g = ROOT / "kaggle/level3/out/grid/out"
fmts = {"r1_full_r16_lr2e-4": "full", "r2_res_r16_lr2e-4": "resolution", "r3_full_r64_lr2e-4": "full", "r4_full_r16_lr5e-4": "full"}
for name, fmt in fmts.items():
    runs[name] = load_run(g / f"{name}_dev_sym.jsonl", fmt)

exact = {k: {i: int(v[i] == "exact") for i in ids} for k, v in runs.items()}
by_cluster = {}
for i in ids:
    by_cluster.setdefault(cluster[i], []).append(i)
clusters = list(by_cluster.values())


def boot_diff(a, b, B=4000, seed=1):
    """Cluster bootstrap of mean(exact_a) - mean(exact_b), resampling merges (not items)."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(B):
        pick = [c for _ in clusters for c in [rng.choice(clusters)]]
        n = sum(len(c) for c in pick)
        diffs.append(100 * sum(exact[a][i] - exact[b][i] for c in pick for i in c) / n)
    diffs.sort()
    return {"diff": 100 * sum(exact[a][i] - exact[b][i] for i in ids) / len(ids), "ci95": [diffs[int(B * .025)], diffs[int(B * .975)]]}


rows = []
for k, v in runs.items():
    s = summarize(list(v.values()))
    rows.append({"row": k, "n": s["n"], "exact_pct": s["exact_pct"], "exact_or_normalized_pct": s["exact_or_normalized_pct"],
                 "conflict_pct": s["conflict_pct"], "wrong": s["wrong"], "no_block": s["no_block"]})
print(f"{'row':22}{'n':>5}{'exact%':>8}{'+norm%':>8}{'conflict%':>10}{'wrong':>7}{'no_blk':>7}")
for r in rows:
    print(f"{r['row']:22}{r['n']:>5}{r['exact_pct']:8.1f}{r['exact_or_normalized_pct']:8.1f}{r['conflict_pct']:10.1f}{r['wrong']:7}{r['no_block']:7}")
print(f"\n{len(clusters)} merge clusters over {len(ids)} items (largest {max(map(len, clusters))})")
cmp = {}
for a, b in [("r2_res_r16_lr2e-4", "trivial_right"), ("r2_res_r16_lr2e-4", "k2_low_B"), ("r2_res_r16_lr2e-4", "qwen_untuned_B"),
             ("r2_res_r16_lr2e-4", "r1_full_r16_lr2e-4"), ("r3_full_r64_lr2e-4", "r1_full_r16_lr2e-4"),
             ("r4_full_r16_lr5e-4", "r1_full_r16_lr2e-4"), ("r1_full_r16_lr2e-4", "trivial_right")]:
    cmp[f"{a} - {b}"] = boot_diff(a, b)
    d = cmp[f"{a} - {b}"]
    print(f"{a} - {b}: {d['diff']:+.1f} pts, 95% cluster CI [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]")
(ROOT / "results/level3.json").write_text(json.dumps({"rows": rows, "n_clusters": len(clusters), "comparisons": cmp}, indent=1))
