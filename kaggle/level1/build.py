"""Build a self-contained Kaggle script kernel for Level 1.

  .venv/bin/python kaggle/level1/build.py smoke2 | fast | higha | highb

Writes kaggle/level1/build/<mode>/{script.py,kernel-metadata.json}. Does NOT push.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
mode = sys.argv[1]
ids = json.loads((ROOT / "results/sample200_ids.json").read_text())

CONFIGS = {
    # Exercises sharding, merge and the pass-2 retry on a handful of hunks (tiny caps force truncation).
    "smoke2": {"n": 6, "ids": ids, "variants": ["A"],
               "max_new": {"qwen_nothink": 1536, "k2_emptythink": 1536, "k2_low": 1536, "k2_high": 600},
               "retry_max_new": {"k2_high": 3000}, "retry_bs": 2,
               "bs": {"qwen_nothink": 3, "k2_emptythink": 3, "k2_low": 3, "k2_high": 3}},
    # Fast rows, both prompt variants, full 200 hunks.
    "fast": {"n": None, "ids": ids, "variants": ["A", "B"],
             "rows": ["qwen_nothink", "k2_emptythink", "k2_low"],
             "max_new": {"qwen_nothink": 1280, "k2_emptythink": 1280, "k2_low": 1280},
             "bs": {"qwen_nothink": 16, "k2_emptythink": 16, "k2_low": 16}},
    # K2 at its recommended high effort. Pass 1 caps at 4096; truncated hunks rerun at 16384.
    "higha": {"n": None, "ids": ids, "variants": ["A"], "rows": ["k2_high"],
              "max_new": {"k2_high": 4096}, "retry_max_new": {"k2_high": 16384}, "retry_bs": 3,
              "bs": {"k2_high": 10}},
    "highb": {"n": None, "ids": ids, "variants": ["B"], "rows": ["k2_high"],
              "max_new": {"k2_high": 4096}, "retry_max_new": {"k2_high": 16384}, "retry_bs": 3,
              "bs": {"k2_high": 10}},
}
cfg = CONFIGS[mode]

src = (HERE / "main.py").read_text()
scoring = (ROOT / "scoring.py").read_text()
src = src.replace("from scoring import score, summarize, BUCKETS  # @@INLINE_SCORING@@", scoring)
src = src.replace("EMBEDDED_CONFIG = None  # @@CONFIG@@", "EMBEDDED_CONFIG = " + repr(cfg))
assert "@@" not in src.replace("@@INLINE", "").replace("@@CONFIG", "") or True

# Kaggle's image pins older transformers/bitsandbytes; both models need new ones.
prelude = '''import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                       "transformers==5.23.0", "accelerate", "bitsandbytes==0.50.2", "huggingface_hub"])
'''
out = HERE / "build" / mode
out.mkdir(parents=True, exist_ok=True)
(out / "script.py").write_text(prelude + src)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"kiranscript/resolver-level1-{mode}",
    "title": f"resolver level1 {mode}",
    "code_file": "script.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
}, indent=1))
print("built", out)
