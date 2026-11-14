"""Level 1: run Qwen3.5-2B and K2-Horizon-3.7B (4-bit) on a fixed Merge-Bench Java sample.

Runs locally (`python main.py --models qwen --n 5`) or on Kaggle (build.py inlines the
scorer and a config, then pushes this file as a script kernel).

Outputs, under OUT_DIR:
  <row>.jsonl            one line per hunk: id, bucket, tokens, truncated, raw output
  summary.json           per-row bucket counts + token/memory/timing stats
  review/<row>.md        20 raw outputs per row for hand reading
"""
import argparse, gc, json, os, random, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

# ---- injected by build.py; None when run locally ---------------------------
EMBEDDED_CONFIG = None  # @@CONFIG@@
from scoring import score, summarize, BUCKETS  # @@INLINE_SCORING@@

RELEASE_URL = "https://github.com/benedikt-schesch/Merge-Bench/releases/download/main/merges.tar.gz"

# Prompt variants. "A" is Merge-Bench's own prompt (the dataset's `question` field, unchanged).
# "B" forbids leaving the conflict unresolved: it swaps only the sentence that permits it.
DECLINE_SENTENCE = (
    "If you are not sure on how to resolve the conflict or if the intent is ambiguous, "
    "please return the same snippet with the conflict.\n"
)
NO_DECLINE_SENTENCE = (
    "You must always resolve the conflict. Never return the snippet with conflict markers "
    "(<<<<<<<, =======, >>>>>>>); if the intent is ambiguous, choose the most plausible resolution.\n"
)

# model key -> HF id and chat/generation setup. Every K2 row uses the card's recommended
# sampling (temperature 1.0, top_p 0.95). Qwen is greedy: the fine-tuned model will be too.
MODELS = {
    "qwen": {"hf_id": "Qwen/Qwen3.5-2B"},
    "k2": {"hf_id": "IFM/K2-Horizon-3.7B", "revision": "ea9a45d0e46f25a9a1848621e0f5605d937040a2"},
}
# row name -> (model key, mode)
ROWS = {
    "qwen_nothink": ("qwen", "nothink"),
    "k2_emptythink": ("k2", "emptythink"),  # closest thing K2 has to "thinking off"
    "k2_low": ("k2", "low"),                # lightest official effort (think_faster)
    "k2_high": ("k2", "high"),              # the card's recommended setting
}
K2_CLOSE = {"high": "</ifm|think>", "emptythink": "</ifm|think>", "low": "</ifm|think_faster>"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_data(work, ids):
    work.mkdir(parents=True, exist_ok=True)
    tar = work / "merges.tar.gz"
    if not (work / "merges").exists():
        log("downloading Merge-Bench release")
        urllib.request.urlretrieve(RELEASE_URL, tar)
        with tarfile.open(tar) as t:
            t.extractall(work / "merges")
    from datasets import load_from_disk
    ds = load_from_disk(str(work / "merges/merges/repos_reaper_java/dataset"))["test"]
    assert len(ds) == 806, len(ds)
    return [{"id": i, "question": ds[i]["question"], "answer": ds[i]["answer"]} for i in ids]


def make_prompt_text(question, variant):
    if variant == "A":
        return question
    assert DECLINE_SENTENCE in question, "decline sentence not found; variant B cannot be built"
    return question.replace(DECLINE_SENTENCE, NO_DECLINE_SENTENCE)


def load_model(key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    hf_id = MODELS[key]["hf_id"]
    kw = dict(trust_remote_code=(key == "k2"))
    if MODELS[key].get("revision"):
        kw["revision"] = MODELS[key]["revision"]
    tok = AutoTokenizer.from_pretrained(hf_id, **kw)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16,
            low_cpu_mem_usage=True, **kw)
    except Exception as e:  # Qwen3.5 may only register as an image-text model on some versions
        if key != "qwen":
            raise
        log("AutoModelForCausalLM failed for qwen, trying ImageTextToText:", repr(e)[:300])
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16,
            low_cpu_mem_usage=True)
    model.eval()
    mem = {"load_allocated_gb": (torch.cuda.memory_allocated() - base) / 1e9,
           "load_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
           "n_params_b": sum(p.numel() for p in model.parameters()) / 1e9,
           "class": type(model).__name__}
    log(key, "loaded", mem)
    return tok, model, mem


def build_input(tok, key, mode, text):
    msgs = [{"role": "user", "content": text}]
    if key == "qwen":
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    effort = "low" if mode == "low" else "high"
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                reasoning_effort=effort)
    if mode == "emptythink":
        s += "</ifm|think>"
    return s


def extract_answer(key, mode, raw):
    """Return (answer_text, truncated). Thinking rows without a closing tag are truncated."""
    for t in ("<|ifm|im_end|>", "<|im_end|>", "<|endoftext|>"):
        raw = raw.replace(t, "")
    if key == "qwen":
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1]
        return raw.strip(), False
    if mode == "emptythink":
        return raw.strip(), False
    close = K2_CLOSE[mode]
    if close in raw:
        return raw.split(close, 1)[1].strip(), False
    return "", True


def run_batches(todo, texts, tok, model, key, max_new, bs, seed, mode, label):
    """Generate for `todo` (already sorted) and return a list of per-hunk records."""
    import torch
    sample = key == "k2"
    torch.manual_seed(seed)
    recs, t0, tot = [], time.time(), 0
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    for b in range(0, len(todo), bs):
        batch = todo[b:b + bs]
        enc = tok([texts[it["id"]] for it in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        enc.pop("token_type_ids", None)
        gk = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        gk.update(dict(do_sample=True, temperature=1.0, top_p=0.95) if sample else dict(do_sample=False))
        with torch.no_grad():
            out = model.generate(**enc, **gk)
        gen = out[:, enc["input_ids"].shape[1]:]
        for it, g in zip(batch, gen):
            ids = g.tolist()
            n_tok = next((i + 1 for i, t in enumerate(ids) if t in eos_ids), len(ids))
            hit_limit = n_tok == len(ids) and ids[-1] not in eos_ids and ids[-1] != tok.pad_token_id
            raw = tok.decode(ids[:n_tok], skip_special_tokens=False)
            if tok.pad_token:
                raw = raw.replace(tok.pad_token, "")
            ans, trunc = extract_answer(key, mode, raw)
            recs.append({"id": it["id"], "bucket": score(ans, it["answer"], "first"),
                         "bucket_last": score(ans, it["answer"], "last"),
                         "gen_tokens": n_tok, "truncated": trunc, "hit_limit": hit_limit, "raw": raw})
            tot += n_tok
        log(label, f"{min(b + bs, len(todo))}/{len(todo)}", f"{tot / (time.time() - t0):.1f} tok/s",
            f"max_new={max_new}")
    return recs


def generate_row(row, tok, model, key, mode, variant, items, out_dir, max_new, bs, seed,
                 retry_max_new=None, retry_bs=2):
    """Pass 1 at `max_new`; optionally pass 2 re-runs only truncated hunks at `retry_max_new`."""
    import torch
    path = out_dir / f"{row}.jsonl"
    done = {}
    if path.exists():
        for l in open(path):
            r = json.loads(l); done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    log(row, f"{len(done)} done, {len(todo)} to go, max_new={max_new}, bs={bs}")
    texts = {it["id"]: build_input(tok, key, mode, make_prompt_text(it["question"], variant)) for it in items}
    lens = {i: len(tok(t, add_special_tokens=False).input_ids) for i, t in texts.items()}
    todo.sort(key=lambda it: lens[it["id"]])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    recs = list(done.values())
    for b in range(0, len(todo), bs * 4):  # checkpoint every few batches
        recs += run_batches(todo[b:b + bs * 4], texts, tok, model, key, max_new, bs, seed + b, mode, row)
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    if retry_max_new:
        redo_ids = {r["id"] for r in recs if r["hit_limit"] or r["truncated"]}
        if redo_ids:
            redo = sorted((it for it in items if it["id"] in redo_ids), key=lambda it: lens[it["id"]])
            log(row, f"pass 2: {len(redo)} truncated hunks at max_new={retry_max_new}")
            new = {r["id"]: r for r in run_batches(redo, texts, tok, model, key, retry_max_new,
                                                    retry_bs, seed + 999, mode, row + "/pass2")}
            for r in new.values():
                r["pass2"] = True
            recs = [new.get(r["id"], r) for r in recs]
            with open(path, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
    return {"row": row, "wall_s": round(time.time() - t0, 1),
            "gen_peak_gb": torch.cuda.max_memory_allocated() / 1e9}


def summarize_row(out_dir, row, variant, max_new, retry_max_new):
    rows = [json.loads(l) for l in open(out_dir / f"{row}.jsonl")]
    toks = sorted(r["gen_tokens"] for r in rows)
    s = summarize([r["bucket"] for r in rows])
    last = summarize([r["bucket_last"] for r in rows])
    s.update(row=row, variant=variant, max_new_tokens=max_new, retry_max_new_tokens=retry_max_new,
             exact_pct_last_block=last["exact_pct"],
             exact_or_normalized_pct_last_block=last["exact_or_normalized_pct"],
             truncated_final=sum(r["truncated"] for r in rows),
             hit_limit_final=sum(r["hit_limit"] for r in rows),
             rerun_in_pass2=sum(1 for r in rows if r.get("pass2")),
             tokens_median=toks[len(toks) // 2], tokens_p90=toks[int(len(toks) * .9)], tokens_max=toks[-1])
    return s


def write_review(row, out_dir, items_by_id, k=20):
    rows = [json.loads(l) for l in open(out_dir / f"{row}.jsonl")]
    pick = random.Random(0).sample(rows, min(k, len(rows)))
    (out_dir / "review").mkdir(exist_ok=True)
    with open(out_dir / "review" / f"{row}.md", "w") as f:
        for r in pick:
            f.write(f"## hunk {r['id']}  bucket={r['bucket']} last={r['bucket_last']}  tokens={r['gen_tokens']}  truncated={r['truncated']}\n\n")
            f.write("### model output (raw)\n```\n" + r["raw"] + "\n```\n\n")
            f.write("### developer resolution\n```java\n" + items_by_id[r["id"]]["answer"] + "\n```\n\n")


def parse_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="qwen_nothink,k2_emptythink,k2_low,k2_high")
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--n", type=int, default=None, help="use only the first n sample ids")
    ap.add_argument("--ids_file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--shard", default=None, help="k/n: process only items[k::n] (child mode)")
    args = ap.parse_args()
    cfg = dict(EMBEDDED_CONFIG or {})
    on_kaggle = os.path.exists("/kaggle/working")
    cfg["out"] = args.out or ("/kaggle/working/out" if on_kaggle else "out")
    cfg["work"] = args.work or ("/kaggle/working/data" if on_kaggle else "work")
    cfg["ids"] = cfg.get("ids") or json.load(open(args.ids_file))
    n = cfg.get("n", args.n)
    if n:
        cfg["ids"] = cfg["ids"][:n]
    cfg["rows"] = cfg.get("rows") or args.rows.split(",")
    cfg["variants"] = cfg.get("variants") or args.variants.split(",")
    mn = {"qwen_nothink": 2048, "k2_emptythink": 2048, "k2_low": 2048, "k2_high": 4096}
    mn.update(cfg.get("max_new") or {})
    cfg["max_new"] = mn
    bs = {"qwen_nothink": 8, "k2_emptythink": 8, "k2_low": 8, "k2_high": 8}
    bs.update(cfg.get("bs") or {})
    cfg["bs"] = bs
    cfg["retry_max_new"] = cfg.get("retry_max_new") or {}
    cfg["shard"] = args.shard
    return cfg


def env_info():
    import torch, transformers
    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count(),
           "gpu_mem_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
           "bf16_supported": torch.cuda.is_bf16_supported()}
    try:
        import bitsandbytes; env["bitsandbytes"] = bitsandbytes.__version__
    except Exception as e:
        env["bitsandbytes"] = repr(e)
    return env


def run_worker(cfg, items, out_dir):
    """Load each model once and run its rows over `items`. Writes <row>.jsonl in out_dir."""
    import torch
    stats = {"load": {}, "errors": {}, "gen": []}
    for key in ("qwen", "k2"):
        key_rows = [r for r in cfg["rows"] if ROWS[r][0] == key]
        if not key_rows:
            continue
        try:
            tok, model, mem = load_model(key)
        except Exception as e:
            import traceback; traceback.print_exc()
            stats["errors"][key] = repr(e)[:2000]
            continue
        stats["load"][key] = mem
        for r in key_rows:
            for v in cfg["variants"]:
                name = f"{r}_{v}"
                try:
                    stats["gen"].append(generate_row(
                        name, tok, model, key, ROWS[r][1], v, items, out_dir, cfg["max_new"][r],
                        cfg["bs"][r], 1234, cfg["retry_max_new"].get(r), cfg.get("retry_bs", 2)))
                except Exception as e:
                    import traceback; traceback.print_exc()
                    stats["errors"][name] = repr(e)[:2000]
                    torch.cuda.empty_cache()
                (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
        del model, tok; gc.collect(); torch.cuda.empty_cache()
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))


def main():
    cfg = parse_cfg()
    out_dir = Path(cfg["out"]); out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(cfg["work"])
    items = load_data(work, cfg["ids"])

    if cfg["shard"]:  # child: one GPU, a slice of the hunks
        k, n = map(int, cfg["shard"].split("/"))
        run_worker(cfg, items[k::n], out_dir)
        return

    import torch
    env = env_info(); log("env", env)
    nshards = min(torch.cuda.device_count(), cfg.get("max_shards", 2))
    if nshards > 1:
        procs = []
        for k in range(nshards):
            d = out_dir / f"shard{k}"; d.mkdir(exist_ok=True)
            cmd = [sys.executable, os.path.abspath(__file__), "--shard", f"{k}/{nshards}",
                   "--out", str(d), "--work", str(work)]
            if cfg.get("n"): cmd += ["--n", str(cfg["n"])]
            procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}))
        for pr in procs:
            pr.wait()
        shard_dirs = [out_dir / f"shard{k}" for k in range(nshards)]
        for f in out_dir.glob("*.jsonl"):  # stale merged files from an earlier run in this dir
            f.unlink()
        for sd in shard_dirs:  # merge per-row jsonl
            for f in sd.glob("*.jsonl"):
                with open(out_dir / f.name, "a") as o:
                    o.write(f.read_text())
    else:
        shard_dirs = [out_dir]
        run_worker(cfg, items, out_dir)

    stats = [json.loads((sd / "stats.json").read_text()) for sd in shard_dirs if (sd / "stats.json").exists()]
    items_by_id = {it["id"]: it for it in items}
    summary = {"env": env, "n": len(items), "rows": [], "errors": {}, "load": {}, "shards": len(shard_dirs)}
    for st in stats:
        summary["errors"].update(st["errors"]); summary["load"].update(st["load"])
    for key in ("qwen", "k2"):
        for r in [r for r in cfg["rows"] if ROWS[r][0] == key]:
            for v in cfg["variants"]:
                name = f"{r}_{v}"
                if not (out_dir / f"{name}.jsonl").exists():
                    continue
                s = summarize_row(out_dir, name, v, cfg["max_new"][r], cfg["retry_max_new"].get(r))
                s["wall_s_per_shard"] = [g["wall_s"] for st in stats for g in st["gen"] if g["row"] == name]
                s["gen_peak_gb"] = max([g["gen_peak_gb"] for st in stats for g in st["gen"] if g["row"] == name] or [0])
                summary["rows"].append(s)
                write_review(name, out_dir, items_by_id)
                log("RESULT", json.dumps(s))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
