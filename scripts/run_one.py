"""Resolve ONE merge conflict with the fine-tuned adapter (Qwen3.5-2B, 4-bit NF4, thinking off, greedy), as evaluated in Level 4.

  python scripts/run_one.py --adapter kaggle/adapter_ds                 # built-in example, examples/conflict.java.txt
  python scripts/run_one.py --adapter <hf-repo-or-dir> --file my_conflict.txt

--file holds a Java snippet with one two-way conflict (<<<<<<< / ======= / >>>>>>>, the Merge-Bench format, no base side) plus context lines.
Needs a CUDA GPU (about 2 GB). Research use: it ties the trivial "take the right side" rule on the benchmark, see README.md.
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from formats import prompt_resolution, rebuild_from_resolution  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True, help="local adapter directory or Hugging Face repo id")
ap.add_argument("--file", default=str(ROOT / "examples/conflict.java.txt"))
ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
a = ap.parse_args()

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

snippet = Path(a.file).read_text().rstrip("\n")
question = "```java\n" + snippet + "\n```"
tok = AutoTokenizer.from_pretrained(a.base)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16)
model = PeftModel.from_pretrained(model, a.adapter).eval()

text = tok.apply_chat_template([{"role": "user", "content": prompt_resolution(question)}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=768, do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0])
raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
merged = rebuild_from_resolution(question, raw)
print("--- model output (replacement lines only) ---\n" + raw.strip())
print("\n--- merged snippet ---\n" + (merged.removeprefix("```java\n").removesuffix("\n```") if merged else "(could not parse the model output)"))
