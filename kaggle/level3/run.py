"""Level 3: QLoRA fine-tune Qwen3.5-2B (thinking off) on merge conflicts, then score on dev_sym.

Local smoke:  PYTHONPATH=. python kaggle/level3/run.py --smoke
Kaggle     :  build.py inlines a config; data + scoring.py + formats.py come from the private dataset
              kiranscript/resolver-data mounted at /kaggle/input/resolver-data.

One process trains+evaluates the configs assigned to it, sequentially (fresh model per config). The parent
starts one child per GPU. Every config trains for exactly one epoch over train_clean (one random orientation per hunk).
"""
import argparse, collections, gc, json, math, os, random, subprocess, sys, time
from pathlib import Path

EMBEDDED_CONFIG = None  # @@CONFIG@@
LOCAL = not os.path.exists("/kaggle/working")
def _find_input():
    """Kaggle mounts datasets under /kaggle/input/... (the exact layout varies); locate our files by name."""
    import glob
    hits = glob.glob("/kaggle/input/**/scoring.py", recursive=True)
    if not hits:
        print("input tree:", [p for p in glob.glob("/kaggle/input/**", recursive=True)][:40], flush=True)
        raise FileNotFoundError("resolver-data dataset not mounted")
    return str(Path(hits[0]).parent)


_IN = None if LOCAL else _find_input()
DATA = Path(os.environ.get("RESOLVER_DATA", "data" if LOCAL else _IN))
CODE = Path(os.environ.get("RESOLVER_CODE", "." if LOCAL else _IN))
sys.path.insert(0, str(CODE))
from scoring import score, summarize, declined_strict, extract_code_block, swap_example  # noqa: E402
from formats import prompt_full, prompt_resolution, target_full, target_resolution, rebuild_from_resolution  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-2B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def make_prompt(fmt, q):
    return prompt_full(q) if fmt == "full" else prompt_resolution(q)


def make_target(fmt, q, a):
    return target_full(a) if fmt == "full" else target_resolution(q, a)


def chat_prompt(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def build_train(tok, rows, fmt, orient, max_len, limit=None, seed=0):
    """orient: "none" (as shipped), "both" (each hunk in both orientations) or "random" (one random orientation per hunk)."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    ex, dropped = [], collections.Counter()
    rng = random.Random(seed + 1)
    for r in rows:
        variants = [(r["question"], r["answer"])]
        sw = swap_example(r["question"], r["answer"]) if orient != "none" else None
        if sw:
            variants = variants + [sw] if orient == "both" else [sw] if rng.random() < 0.5 else variants
        for q, a in variants:
            tgt = make_target(fmt, q, a)
            if tgt is None:
                dropped["not_representable"] += 1; continue
            p = tok(chat_prompt(tok, make_prompt(fmt, q)), add_special_tokens=False).input_ids
            t = tok(tgt, add_special_tokens=False).input_ids + [end]
            if len(p) + len(t) > max_len:
                dropped["too_long"] += 1; continue
            ex.append((p, t))
    random.Random(seed).shuffle(ex)
    if limit:
        ex = ex[:limit]
    return ex, dict(dropped)


def load_4bit(adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0},
                                                 dtype=torch.float16, low_cpu_mem_usage=True)
    return tok, model


def train(model, tok, examples, cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lc = LoraConfig(r=cfg["rank"], lora_alpha=cfg["rank"] * cfg.get("alpha_mult", 2), lora_dropout=0.05,
                    target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lc)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    log("trainable params", n_train, "examples", len(examples))
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    mb = cfg.get("mb", 4)
    accum = max(1, cfg.get("accum", 16) // mb)  # optimizer step every `accum` micro-batches (~16 sequences)
    pad = tok.pad_token_id
    # group similar lengths into micro-batches (reduces padding), keep the shuffled order of the batches
    chunks = [sorted(examples[i:i + 64], key=lambda e: len(e[0]) + len(e[1])) for i in range(0, len(examples), 64)]
    batches = [c[j:j + mb] for c in chunks for j in range(0, len(c), mb)]
    random.Random(cfg.get("seed", 0)).shuffle(batches)
    total_steps = max(1, len(batches) // accum)
    warm = max(1, int(0.05 * total_steps))
    sched = lambda s_: (s_ + 1) / warm if s_ < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s_ - warm) / max(1, total_steps - warm)))
    t0, toks, losses, step, micro = time.time(), 0, [], 0, 0
    max_s = cfg.get("max_train_minutes", 1e9) * 60
    stopped_early = False
    for bt in batches[: total_steps * accum]:
        L = max(len(p) + len(t) for p, t in bt)
        K = max(len(t) for p, t in bt) + 1
        ids = torch.full((len(bt), L), pad, dtype=torch.long)
        att = torch.zeros((len(bt), L), dtype=torch.long)
        lab = torch.full((len(bt), L), -100, dtype=torch.long)
        for i, (p, t) in enumerate(bt):  # left padding: sequences end at the same position
            seq = p + t
            ids[i, L - len(seq):] = torch.tensor(seq); att[i, L - len(seq):] = 1
            lab[i, L - len(t):] = torch.tensor(t)
        ids, att, lab = ids.cuda(), att.cuda(), lab.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=ids, attention_mask=att, logits_to_keep=K).logits  # [B, K, V]
        tgt = lab[:, L - K + 1:]
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                                 tgt.reshape(-1), ignore_index=-100)
        scaler.scale(loss / accum).backward()
        losses.append(loss.item()); toks += int(att.sum()); micro += 1
        if micro % accum == 0:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * sched(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == total_steps:
                w = losses[-accum * 5:]
                log(f"step {step}/{total_steps} loss {sum(w) / len(w):.4f} {toks / (time.time() - t0):.0f} tok/s "
                    f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
            if time.time() - t0 > max_s:
                stopped_early = True; log("time cap reached, stopping early"); break
    stats = {"steps": step, "planned_steps": total_steps, "examples_seen": micro * mb, "train_s": round(time.time() - t0),
             "final_loss": sum(losses[-accum:]) / max(1, len(losses[-accum:])), "trainable_params": n_train,
             "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "stopped_early": stopped_early}
    return model, stats


def evaluate(model, tok, rows, fmt, max_new, bs, out_path):
    import torch
    model.eval()
    model.config.use_cache = True
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    texts = {r["id"]: chat_prompt(tok, make_prompt(fmt, r["question"])) for r in rows}
    order = sorted(rows, key=lambda r: len(texts[r["id"]]))
    recs, t0, gen_tokens = [], time.time(), 0
    for b in range(0, len(order), bs):
        batch = order[b:b + bs]
        enc = tok([texts[r["id"]] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for r, g in zip(batch, gen):
            ids = g.tolist()
            n = next((i + 1 for i, t in enumerate(ids) if t in eos), len(ids))
            hit = n == len(ids) and ids[-1] not in eos and ids[-1] != tok.pad_token_id
            raw = tok.decode(ids[:n], skip_special_tokens=True)
            if "</think>" in raw:
                raw = raw.split("</think>", 1)[1]
            ans = raw if fmt == "full" else (rebuild_from_resolution(r["question"], raw) or "")
            bucket = score(ans, r["answer"])
            code = extract_code_block(ans)
            recs.append({"id": r["id"], "bucket": bucket, "gen_tokens": n, "hit_limit": hit,
                         "declined_strict": bool(code is not None and declined_strict(code, r["answer"])), "raw": raw})
            gen_tokens += n
        log(f"eval {min(b + bs, len(order))}/{len(order)} {gen_tokens / (time.time() - t0):.0f} tok/s")
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    s = summarize([r["bucket"] for r in recs])
    s.update(declined_strict_pct=100 * sum(r["declined_strict"] for r in recs) / len(recs),
             hit_limit=sum(r["hit_limit"] for r in recs), eval_s=round(time.time() - t0),
             tokens_p50=sorted(r["gen_tokens"] for r in recs)[len(recs) // 2])
    return s


def run_config(cfg, out_dir, smoke):
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    log("=== config", cfg)
    tok, model = load_4bit()
    train_rows = read_jsonl(DATA / "train_clean.jsonl")
    dev = read_jsonl(DATA / "dev_sym.jsonl")
    if cfg.get("eval_limit"):
        dev = dev[: cfg["eval_limit"]]
    ex, dropped = build_train(tok, train_rows, cfg["format"], cfg.get("orient", "random"), cfg.get("max_len", 2048),
                              limit=24 if smoke else cfg.get("limit"))
    lens = sorted(len(p) + len(t) for p, t in ex)
    log("train examples", len(ex), "dropped", dropped, "median len", lens[len(lens) // 2], "max", lens[-1])
    torch.cuda.reset_peak_memory_stats()
    model, tstats = train(model, tok, ex, dict(cfg, accum=8 if smoke else cfg.get("accum", 16)))
    model.save_pretrained(out_dir / f"{name}_adapter")
    ev = evaluate(model, tok, dev[:12] if smoke else dev, cfg["format"], cfg.get("max_new", 1280 if cfg["format"] == "full" else 768),
                  cfg.get("eval_bs", 16), out_dir / f"{name}_dev_sym.jsonl")
    res = {"config": cfg, "train": tstats, "train_examples": len(ex), "dropped": dropped, "dev_sym": ev}
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    log("RESULT", json.dumps(res))
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--configs", default=None, help="json list of config dicts (child mode)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out or ("out_l3" if LOCAL else "/kaggle/working/out"))
    if args.smoke:
        run_config({"name": "smoke", "format": os.environ.get("FMT", "resolution"), "rank": 8, "lr": 2e-4}, out_dir, True)
        return
    if args.configs:  # child: run its configs sequentially on CUDA_VISIBLE_DEVICES's GPU
        for c in json.loads(args.configs):
            try:
                run_config(c, out_dir, False)
            except Exception:
                import traceback; traceback.print_exc()
                (out_dir / f"{c['name']}.error").write_text(traceback.format_exc())
        return
    cfgs = EMBEDDED_CONFIG["configs"]
    import torch
    n = max(1, min(torch.cuda.device_count(), EMBEDDED_CONFIG.get("max_gpus", 2)))
    procs = []
    for k in range(n):
        mine = cfgs[k::n]
        if mine:
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--configs", json.dumps(mine),
                                           "--out", str(out_dir)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}))
    for p in procs:
        p.wait()
    summary = [json.loads(f.read_text()) for f in sorted(out_dir.glob("*.json"))]
    (out_dir / "summary_all.json").write_text(json.dumps(summary, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
