"""W1 - check: the base model and the data are what the job says.

The base model folder has weights, config.json, and the processor and
tokenizer files; its architecture is printed. Each data entry opens, the
mapped columns exist, and a few rows are printed. Nothing is written but the
stage marker.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, heading, resolve_input, run_stage  # noqa: E402
from whisper_common import decode, source_rows  # noqa: E402

PROCESSOR_FILES = ("preprocessor_config.json", "tokenizer_config.json")


def work(job: dict) -> dict:
    if job["kind"] != "whisper":
        raise Stop(f"{job['name']} is a '{job['kind']}' job, not a whisper job")
    for key in ("base_model", "language", "data", "engine"):
        if key not in job:
            raise Stop(f"the job file has no '{key}'")
    if "train" not in job["data"]:
        raise Stop("the job file has no data.train")

    heading("Base model")
    base = resolve_input(job["base_model"], "models", job)
    files = {p.name for p in base.iterdir()}
    if not ({"model.safetensors", "pytorch_model.bin"} & files):
        raise Stop(f"{base} has neither model.safetensors nor pytorch_model.bin")
    for name in ("config.json",) + PROCESSOR_FILES:
        if name not in files:
            raise Stop(f"{base} has no {name}")
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if "WhisperForConditionalGeneration" not in config.get("architectures", []):
        raise Stop(f"{base} is {config.get('architectures')}, not a Whisper model")
    arch = {k: config.get(k) for k in ("num_mel_bins", "encoder_layers", "decoder_layers",
                                       "d_model", "vocab_size")}
    print(f"  {base}")
    for k, v in arch.items():
        print(f"  {k:16} {v}")

    counts = {}
    for which in ("train", "test", "regression"):
        if not job["data"].get(which):
            continue
        heading(f"data.{which}")
        rows = source_rows(job, which)
        if not rows:
            raise Stop(f"data.{which} has no rows")
        counts[which] = len(rows)
        speakers = {r["speaker"] for r in rows if r["speaker"]}
        print(f"  {len(rows):,} rows, {len(speakers):,} speakers")
        for r in rows[:3]:
            seconds = len(decode(r["audio"])) / 16000
            print(f"  [{seconds:4.1f} s, speaker {r['speaker'] or '-'}] {r['text'][:90]}")
        if which != "regression" and not speakers:
            print("  NOTE: no speaker column. Test data will be held back by clip, which "
                  "can let a speaker's voice leak from training into testing.")
    return {"architecture": arch, "rows": counts}


if __name__ == "__main__":
    run_stage("W1", "check the base model and the data", work)
