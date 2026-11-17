"""Level 4 verdict report (see results/level4_preregistration.md for the decision rule).

  .venv/bin/python scripts/level4_report.py      # writes results/level4.json
Inputs (local, downloaded from Kaggle into kaggle/level4/out/<kernel>/):
  qwen/        qwen_r2_test.jsonl + .json      (fine-tuned Qwen r2 on 806)
  k2hp1/, k2hp2/  out/k2_high_B.jsonl           (K2 high+B on the 606 hunks outside the Level 1 sample)
  l4base/      out/qwen_nothink_B.jsonl, out/k2_low_B.jsonl (reference rows on 806)
  plus kaggle/level1/out/highb/out/k2_high_B.jsonl (K2 high+B on the 200-hunk sample, Level 1)
"""
import json, math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scoring import score, summarize, trivial_resolve, merge_sha  # noqa: E402
from formats import rebuild_from_resolution  # noqa: E402

test = [json.loads(l) for l in open(ROOT / "data/test806.jsonl")]
ref = {r["id"]: r for r in test}
ids = [r["id"] for r in test]
cluster = {r["id"]: (merge_sha(r["question"]) or f"id:{r['id']}") for r in test}
L4 = ROOT / "kaggle/level4/out"


def clean(raw):
    for t in ("<|ifm|im_end|>", "<|im_end|>", "<|endoftext|>"):
        raw = raw.replace(t, "")
    return raw


def k2_buckets(path, closer):
    """(strict first-block, lenient) buckets per id. Lenient = best of first/last block of the answer, or, if the answer has
    no code block, the last block inside the thinking (K2 gets its most generous reading)."""
    strict, lenient = {}, {}
    for l in open(path):
        r = json.loads(l); raw = clean(r["raw"]); a = raw.split(closer, 1)[1] if closer in raw else ""
        ans = ref[r["id"]]["answer"]
        s_first = score(a, ans, "first"); s_last = score(a, ans, "last")
        strict[r["id"]] = s_first
        best = [s_first, s_last]
        if s_first == "no_block":
            best.append(score(raw, ans, "last"))
        lenient[r["id"]] = "exact" if "exact" in best else s_first
    return strict, lenient


runs = {}
# --- fine-tuned Qwen (re-score from raw; stored bucket must match)
q = {}
for l in open(L4 / "qwen/out/qwen_r2_test.jsonl"):
    r = json.loads(l)
    ans = rebuild_from_resolution(ref[r["id"]]["question"], r["raw"]) or ""
    b = score(ans, ref[r["id"]]["answer"]); assert b == r["bucket"], (r["id"], b, r["bucket"])
    q[r["id"]] = b
runs["qwen_finetuned_r2"] = q
# --- K2 high+B on all 806
hs, hl = {}, {}
for p in (ROOT / "kaggle/level1/out/highb/out/k2_high_B.jsonl", L4 / "k2hp1/out/k2_high_B.jsonl", L4 / "k2hp2/out/k2_high_B.jsonl"):
    if p.exists():
        s_, l_ = k2_buckets(p, "</ifm|think>"); hs.update(s_); hl.update(l_)
have_k2 = len(hl) == 806
runs["k2_high_B_strict"] = hs; runs["k2_high_B_lenient"] = hl
# --- references
for side in ("left", "right"):
    runs[f"trivial_{side}"] = {r["id"]: score(trivial_resolve(r["question"], side), r["answer"]) for r in test}
bp = L4 / "l4base/out"
if (bp / "qwen_nothink_B.jsonl").exists():
    runs["qwen_untuned_B"] = {}
    for l in open(bp / "qwen_nothink_B.jsonl"):
        r = json.loads(l); raw = clean(r["raw"]); raw = raw.split("</think>", 1)[1] if "</think>" in raw else raw
        runs["qwen_untuned_B"][r["id"]] = score(raw, ref[r["id"]]["answer"])
    s_, l_ = k2_buckets(bp / "k2_low_B.jsonl", "</ifm|think_faster>")
    runs["k2_low_B_lenient"] = l_

exact = {k: {i: int(v[i] == "exact") for i in v} for k, v in runs.items()}
rows = []
print(f"{'row':24}{'n':>5}{'exact%':>8}{'+norm%':>8}{'conflict%':>10}{'wrong':>7}{'no_blk':>7}")
for k, v in runs.items():
    s = summarize(list(v.values()))
    rows.append({"row": k, **{x: s[x] for x in ("n", "exact_pct", "exact_or_normalized_pct", "conflict_pct", "wrong", "no_block")}})
    print(f"{k:24}{s['n']:>5}{s['exact_pct']:8.1f}{s['exact_or_normalized_pct']:8.1f}{s['conflict_pct']:10.1f}{s['wrong']:7}{s['no_block']:7}")

by_cluster = {}
for i in ids:
    by_cluster.setdefault(cluster[i], []).append(i)
clusters = list(by_cluster.values())
print(f"\n{len(clusters)} merge clusters over {len(ids)} hunks (largest {max(map(len, clusters))})")


def common(a, b):
    return [i for i in ids if i in exact[a] and i in exact[b]]


def paired(a, b, B=5000, seed=7):
    idx = set(common(a, b)); cl = [[i for i in c if i in idx] for c in clusters]; cl = [c for c in cl if c]
    rng = random.Random(seed); diffs = []
    for _ in range(B):
        pick = [rng.choice(cl) for _ in cl]
        n = sum(len(c) for c in pick)
        diffs.append(100 * sum(exact[a][i] - exact[b][i] for c in pick for i in c) / n)
    diffs.sort()
    n10 = sum(exact[a][i] > exact[b][i] for i in idx); n01 = sum(exact[a][i] < exact[b][i] for i in idx)
    k, n = min(n10, n01), n10 + n01  # exact two-sided McNemar (binomial)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n": len(idx), "diff_pts": 100 * (sum(exact[a][i] for i in idx) - sum(exact[b][i] for i in idx)) / len(idx),
            "ci95_cluster": [diffs[int(B * .025)], diffs[int(B * .975)]], "a_only": n10, "b_only": n01, "mcnemar_p": p}


cmp = {}
pairs = [("qwen_finetuned_r2", "k2_high_B_lenient"), ("qwen_finetuned_r2", "k2_high_B_strict"), ("qwen_finetuned_r2", "trivial_right"),
         ("qwen_finetuned_r2", "trivial_left"), ("qwen_finetuned_r2", "qwen_untuned_B"), ("qwen_finetuned_r2", "k2_low_B_lenient"),
         ("trivial_right", "k2_high_B_lenient")]
for a, b in pairs:
    if a in exact and b in exact:
        cmp[f"{a} - {b}"] = d = paired(a, b)
        print(f"{a} - {b}: {d['diff_pts']:+.1f} pts (n={d['n']}), 95% merge-cluster CI [{d['ci95_cluster'][0]:+.1f}, {d['ci95_cluster'][1]:+.1f}], "
              f"only-a {d['a_only']} / only-b {d['b_only']}, McNemar p = {d['mcnemar_p']:.4f}")

def gold_type(r):
    a = r["answer"].strip()
    from scoring import extract_code_block as ecb
    hit = {st: ecb(trivial_resolve(r["question"], st)) == a for st in ("left", "right", "both", "both_reversed")}
    if hit["left"] and hit["right"]: return "left=right (one side empty or equal)"
    if hit["left"]: return "left only"
    if hit["right"]: return "right only"
    if hit["both"] or hit["both_reversed"]: return "concatenation"
    return "novel (no trivial rule)"


types = {r["id"]: gold_type(r) for r in test}
breakdown = {}
print("\nBy gold hunk type (exact %):")
print(f"{'type':38}{'n':>5}{'Qwen-FT':>9}{'K2-high':>9}{'take-right':>11}")
for t in sorted(set(types.values()), key=lambda t: -sum(v == t for v in types.values())):
    idx = [i for i in ids if types[i] == t]
    f = lambda k: 100 * sum(exact[k][i] for i in idx) / len(idx) if k in exact else float("nan")
    breakdown[t] = {"n": len(idx), "qwen_ft": f("qwen_finetuned_r2"), "k2_high_lenient": f("k2_high_B_lenient"), "take_right": f("trivial_right")}
    print(f"{t:38}{len(idx):>5}{f('qwen_finetuned_r2'):9.1f}{f('k2_high_B_lenient'):9.1f}{f('trivial_right'):11.1f}")

mem = json.loads((L4 / "qwen/out/qwen_r2_test.json").read_text())["mem"] if (L4 / "qwen/out/qwen_r2_test.json").exists() else None
mem_all = json.loads((ROOT / "results/level4_memory.json").read_text())
mem_steady_ok = mem_all["local_rtx3050"]["steady_peak_gb"] < 2.0 and mem_all["kaggle_t4"]["steady_state_peak_gb"] < 2.0
verdict = None
if have_k2 and "qwen_finetuned_r2" in exact:
    d = cmp["qwen_finetuned_r2 - k2_high_B_lenient"]
    # Conservative reading of the pre-registered rule: every bs-1 measurement counts, including the cold first call on the T4.
    mem_ok = bool(mem and mem["bs1_peak_allocated_gb"] < 2.0 and mem_steady_ok and mem_all["kaggle_t4"]["first_call_peak_gb"] < 2.0)
    if d["diff_pts"] > 0 and d["ci95_cluster"][0] > 0 and mem_ok: verdict = "WIN"
    elif d["diff_pts"] > 0 and (d["ci95_cluster"][0] <= 0 or not mem_ok): verdict = "NEAR-WIN"
    else: verdict = "LOSS"
    print(f"\nVERDICT (pre-registered rule, conservative memory reading): {verdict}   memory ok under every measurement: {mem_ok}   steady-state ok: {mem_steady_ok}")
    print("  memory:", json.dumps({k: mem_all[k] for k in ("local_rtx3050", "kaggle_t4")}))
else:
    print("\nK2 high on all 806 not complete yet:", len(hl), "of 806")
(ROOT / "results/level4.json").write_text(json.dumps({"rows": rows, "n_clusters": len(clusters), "comparisons": cmp, "memory": mem, "memory_all": mem_all, "by_gold_type": breakdown, "verdict": verdict}, indent=1))

# --- per-hunk table: ids, merge-SHA cluster, gold hunk type and per-row outcome only (no code), so the CIs / McNemar numbers can be
# --- re-derived without the data or a GPU: scripts/verify_from_csv.py
import csv
cols = list(runs)
with open(ROOT / "results/level4_per_hunk.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "merge_sha", "gold_type"] + [f"{c}__bucket" for c in cols])
    for i in ids:
        w.writerow([i, cluster[i], types[i]] + [runs[c].get(i, "") for c in cols])
