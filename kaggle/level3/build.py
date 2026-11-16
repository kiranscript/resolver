"""Build a Level 3 Kaggle script kernel.   .venv/bin/python kaggle/level3/build.py <mode>   (does not push)"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
mode = sys.argv[1]
BASE = {"format": "full", "rank": 16, "lr": 2e-4}
CONFIGS = {
    # Throughput check on a T4: a few steps + a short eval, both formats, one per GPU.
    "timing": {"max_gpus": 2, "configs": [
        {"name": "t_full", "format": "full", "rank": 16, "lr": 2e-4, "limit": 96, "eval_limit": 32},
        {"name": "t_res", "format": "resolution", "rank": 16, "lr": 2e-4, "limit": 96, "eval_limit": 32}]},
}
CONFIGS.update(json.loads((HERE / "grids.json").read_text()) if (HERE / "grids.json").exists() else {})
cfg = CONFIGS[mode]
src = (HERE / "run.py").read_text().replace("EMBEDDED_CONFIG = None  # @@CONFIG@@", "EMBEDDED_CONFIG = " + repr(cfg))
prelude = '''import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers==5.23.0", "peft", "accelerate",
                       "bitsandbytes==0.50.2", "flash-linear-attention", "huggingface_hub"])
'''
out = HERE / "build" / mode
out.mkdir(parents=True, exist_ok=True)
(out / "script.py").write_text(prelude + src)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"kiranscript/resolver-level3-{mode}", "title": f"resolver level3 {mode}", "code_file": "script.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["kiranscript/resolver-data"] + cfg.get("extra_datasets", []), "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1))
print("built", out)
