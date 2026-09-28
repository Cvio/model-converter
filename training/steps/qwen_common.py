"""What the Qwen (translator) job's stages share.

The translator is trained on the exact text volis sends it. That text comes
from volis itself (`volis --print-prompt <source> <target>`, M7.8), never
copied into Python, so a change to volis's prompt can't silently drift from
what a translator was trained on. Each pair's training text is:

    <volis's prompt, with the source sentence in place of {text}><target><|im_end|>

and only "<target><|im_end|>" is learned (the prompt part is labelled -100).
"""

import json
import platform
import subprocess
from pathlib import Path

from common import REPO, Stop, load_machine, run_dir

QWEN_DEFAULTS = {
    "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
    "training": {"learning_rate": 2.0e-4, "epochs": 2, "batch_size": 16, "gradient_accumulation": 1},
    "general_share": 0.2,
}
END = "<|im_end|>"
MAX_NEW_TOKENS = 256
LENGTH_RATIO = 3.0


def setting(job: dict, section: str, key: str = None):
    if key is None:
        return job.get(section, QWEN_DEFAULTS.get(section))
    return job.get(section, {}).get(key, QWEN_DEFAULTS.get(section, {}).get(key))


def volis_dir() -> Path:
    machine = load_machine()
    folder = machine.get("volis_path") or machine.get("cnverc_path")
    if not folder:
        raise Stop("machine.yaml has no volis_path (the folder volis.exe is in)")
    return Path(folder).expanduser().resolve()


def volis_exe() -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    for name in ("volis", "cnverc"):
        exe = volis_dir() / f"{name}{suffix}"
        if exe.is_file():
            return exe
    raise Stop(f"volis is not in {volis_dir()} (volis_path in machine.yaml)")


def prompt_template(job: dict, source: str, target: str) -> str:
    """volis's exact prompt for a pair of tags, cached in the run folder."""
    cache_path = run_dir(job) / "prompts.json"
    cache = json.loads(cache_path.read_text("utf-8")) if cache_path.is_file() else {}
    key = f"{source}>{target}"
    if key not in cache:
        done = subprocess.run([str(volis_exe()), "--print-prompt", source, target],
                              capture_output=True)
        if done.returncode != 0:
            raise Stop(f"volis --print-prompt {source} {target} failed: "
                       f"{done.stderr.decode('utf-8', 'replace').strip()[-500:]}\n"
                       "(it needs volis from the M7.8 build or later)")
        cache[key] = done.stdout.decode("utf-8")
        if "{text}" not in cache[key]:
            raise Stop(f"volis --print-prompt {source} {target} printed no {{text}} placeholder")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return cache[key]


def prompt_for(job: dict, p: dict) -> str:
    return prompt_template(job, p["source_lang"], p["target_lang"]).replace("{text}", p["source"])


def read_jsonl(path: Path) -> list:
    if not path.is_file():
        raise Stop(f"{path} is missing")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_model(folder: Path, lora: Path = None, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=dtype or torch.bfloat16)
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora)
    return model.to("cuda").eval()


def load_tokenizer(folder: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(folder)
    tokenizer.padding_side = "left"   # for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate(model, tokenizer, prompts: list, batch_size: int = 8) -> list:
    """Greedy continuations of volis's prompts, up to <|im_end|>, as volis
    decodes (greedy, thinking already closed by the prompt's empty block)."""
    import torch

    end_id = tokenizer.convert_tokens_to_ids(END)
    out = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 eos_token_id=end_id, pad_token_id=tokenizer.pad_token_id)
        for row in ids[:, enc["input_ids"].shape[1]:]:
            out.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return out


def chrf_by_direction(pairs: list, hyps: list) -> dict:
    """chrF per direction (e.g. 'es-MX>en'), and the counts."""
    import sacrebleu

    groups = {}
    for p, h in zip(pairs, hyps):
        groups.setdefault(f"{p['source_lang']}>{p['target_lang']}", ([], []))
        groups[f"{p['source_lang']}>{p['target_lang']}"][0].append(h)
        groups[f"{p['source_lang']}>{p['target_lang']}"][1].append(p["target"])
    return {d: {"chrF": round(sacrebleu.CHRF().corpus_score(h, [r]).score, 2), "pairs": len(h)}
            for d, (h, r) in sorted(groups.items())}


def pairs_dir(job: dict) -> Path:
    return REPO / "runs" / job["from_pairs_job"] / "out"
