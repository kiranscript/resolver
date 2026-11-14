"""Level 1 report: re-scores saved raw outputs with the current scoring.py and prints the table + gates.

  .venv/bin/python scripts/level1_report.py            # writes results/level1.json too
Reads kaggle/level1/out/{fast,higha,highb}/out/*.jsonl (whichever exist) and data/test806.jsonl.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scoring import score, summarize, declined_strict, extract_code_block  # noqa: E402

ref = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(ROOT / "data/test806.jsonl")}
K2_CLOSE = ("</ifm|think>", "</ifm|think_faster>")


def answer_of(row, raw):
    for t in ("<|ifm|im_end|>", "<|im_end|>", "<|endoftext|>"):
        raw = raw.replace(t, "")
    if row.startswith("qwen"):
        return raw.split("</think>", 1)[1] if "</think>" in raw else raw
    if "emptythink" in row:
        return raw
    for c in K2_CLOSE:
        if c in raw:
            return raw.split(c, 1)[1]
    return ""  # thinking never closed -> failed response


rows = []
for run in ("fast", "higha", "highb"):
    d = ROOT / f"kaggle/level1/out/{run}/out"
    for f in sorted(d.glob("*.jsonl")):
        row = f.stem
        recs = [json.loads(l) for l in open(f)]
        first, last, lenient, strict = [], [], [], 0
        for r in recs:
            a = answer_of(row, r["raw"])
            first.append(score(a, ref[r["id"]], "first")); last.append(score(a, ref[r["id"]], "last"))
            # Lenient (K2-friendly): if the final answer has no code block, fall back to the last block in its thinking.
            lb = first[-1]
            if lb == "no_block" and not row.startswith("qwen"):
                lb = score(r["raw"].replace("<|ifm|im_end|>", ""), ref[r["id"]], "last")
            lenient.append(lb)
            code = extract_code_block(a)
            strict += bool(code is not None and declined_strict(code, ref[r["id"]]))
        s = summarize(first)
        toks = sorted(r["gen_tokens"] for r in recs)
        s.update(row=row, exact_last_pct=summarize(last)["exact_pct"], exact_lenient_pct=summarize(lenient)["exact_pct"], declined_strict_pct=100 * strict / len(recs),
                 truncated=sum(r["truncated"] for r in recs), hit_limit=sum(r["hit_limit"] for r in recs),
                 tok_p50=toks[len(toks) // 2], tok_p90=toks[int(len(toks) * .9)],
                 # Target for K2 = its most lenient reading (first block, last block, or last block in thinking)
                 headline_exact_pct=s["exact_pct"] if row.startswith("qwen") else max(
                     s["exact_pct"], summarize(last)["exact_pct"], summarize(lenient)["exact_pct"]))
        rows.append(s)

triv = {(r["strategy"], r["set"]): r for r in json.loads((ROOT / "results/level0_trivial.json").read_text())}
hdr = f"{'row':22}{'n':>4}{'exact%':>8}{'lenient%':>9}{'+norm%':>8}{'declined*':>10}{'conflict':>9}{'wrong':>7}{'no_blk':>7}{'trunc':>6}{'tok p50/p90':>13}"
print(hdr); print("-" * len(hdr))
for t in ("left", "right"):
    r = triv[(t, "sample200")]
    print(f"{'[trivial] ' + t:22}{r['n']:>4}{r['exact_pct']:8.1f}{'':>9}{r['exact_or_normalized_pct']:8.1f}")
for r in rows:
    print(f"{r['row']:22}{r['n']:>4}{r['exact_pct']:8.1f}{r['headline_exact_pct']:9.1f}{r['exact_or_normalized_pct']:8.1f}{r['declined_strict_pct']:10.1f}"
          f"{r['conflict_pct']:9.1f}{r['wrong']:7}{r['no_block']:7}{r['truncated']:6}{str(r['tok_p50']) + '/' + str(r['tok_p90']):>13}")
print("* declined = more '<<<<<<<' lines than the developer's answer (stricter than Merge-Bench's 'conflict' bucket)")

k2 = [r for r in rows if r["row"].startswith("k2")]
q = [r for r in rows if r["row"].startswith("qwen")]
out = {"rows": rows}
if k2 and q:
    best = max(k2, key=lambda r: r["headline_exact_pct"])
    qb = max(q, key=lambda r: r["exact_pct"])
    nothink = [r for r in k2 if "high" not in r["row"]]
    nt = max(r["headline_exact_pct"] for r in nothink)
    out.update(k2_best_row=best["row"], k2_best_exact_pct=best["headline_exact_pct"], qwen_best_row=qb["row"],
               qwen_best_exact_pct=qb["exact_pct"], gap_points=best["headline_exact_pct"] - qb["exact_pct"],
               trivial_right_sample_pct=triv[("right", "sample200")]["exact_pct"], k2_nonthinking_best_pct=nt,
               prune_k2_nonthinking_over_40=nt > 40)
    print(f"\nK2 best row            : {best['row']} at {best['headline_exact_pct']:.1f}% exact  (the number to beat)")
    print(f"Qwen untuned best      : {qb['row']} at {qb['exact_pct']:.1f}% exact")
    print(f"Gap Qwen must close    : {out['gap_points']:.1f} points")
    print(f"Trivial 'right' floor  : {out['trivial_right_sample_pct']:.1f}% exact on this sample")
    print(f"Prune check (best non-thinking K2 > ~40%?): {nt:.1f}% -> {'PRUNE' if nt > 40 else 'keep'}")
(ROOT / "results/level1.json").write_text(json.dumps(out, indent=1))
