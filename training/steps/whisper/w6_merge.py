"""W6 - merge: the kept LoRA added into the base model's weights.

1. The base model in full precision (float32; merging into compressed weights
   loses accuracy).
2. The LoRA loaded onto it, then merge_and_unload(): a plain model, no LoRA.
3. Saved as one model.safetensors (max_shard_size 20GB; the converter's step 2
   doesn't read sharded weights), with the processor and generation_config.
4. Checked: the merged model, reloaded from disk alone, is scored on the test
   set with W5's precision and decoding; its WER must be within 0.3 points of
   W5's tuned WER. A bad merge still loads and runs, just not the trained model.

The LoRA folder is kept too.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, heading, resolve_input, run_dir, run_stage, stage_result  # noqa: E402
from textnorm import error_rates  # noqa: E402
from whisper_common import from_pcm16, load_model, load_processor, read_split, transcribe  # noqa: E402

TOLERANCE = 0.3


def work(job: dict) -> dict:
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    base = resolve_input(job["base_model"], "models", job)
    lora = run_dir(job) / "lora"
    merged_dir = run_dir(job) / "merged"
    tuned_wer = stage_result(job, "W5")["test"]["wer"]

    heading("Merging")
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, lora).merge_and_unload()
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="20GB")
    load_processor(base).save_pretrained(merged_dir)
    shards = sorted(p.name for p in merged_dir.glob("*.safetensors"))
    if shards != ["model.safetensors"]:
        raise Stop(f"the merged model was saved as {shards}, not one model.safetensors")
    size = (merged_dir / "model.safetensors").stat().st_size
    print(f"  {merged_dir / 'model.safetensors'}  ({size / 1e6:,.0f} MB)")
    del model
    torch.cuda.empty_cache()

    heading("Checking the merge: the merged model alone, on the test set")
    rows = read_split(job, "test")
    merged = load_model(merged_dir)
    hyps = transcribe(merged, load_processor(merged_dir), [from_pcm16(r["audio"]) for r in rows],
                      job["language"])
    wer = error_rates([r["text"] for r in rows], hyps, job["language"])["wer"]
    print(f"  merged WER {wer:.2f}; tuned (base + LoRA) WER {tuned_wer:.2f}; "
          f"difference {abs(wer - tuned_wer):.2f} (limit {TOLERANCE})")
    if abs(wer - tuned_wer) > TOLERANCE:
        raise Stop("the merged model doesn't score like base + LoRA: the merge went wrong")
    return {"merged_test_wer": round(wer, 3), "size_mb": round(size / 1e6)}


if __name__ == "__main__":
    run_stage("W6", "merge the LoRA into the model", work)
