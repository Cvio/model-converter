"""Q6 - merge: the kept LoRA added into Qwen's weights.

1. The base model in full precision (float32, on the CPU: merging into
   compressed weights loses accuracy).
2. The LoRA loaded onto it, then merge_and_unload().
3. Saved as one model.safetensors (max_shard_size 20GB) with the tokenizer and
   generation_config, in bfloat16 (Qwen's own precision; the GGUF converter
   reads it).
4. Checked: the merged model, reloaded alone, and base + LoRA are given the
   same test prompts with their reference answers, and must predict the same
   next token at least 98% of the time. A correct merge agrees on nearly every
   token; a broken one (wrong weights, the LoRA missing) falls far below.

Why not compare chrF, as the Whisper job does: this model runs in bfloat16,
where merging rounds the LoRA into the weights while base + LoRA rounds the two
separately. The differences are tiny, but greedy decoding amplifies them (one
token flips and the rest of the sentence follows another path), so chrF on a
hundred sentences moves by a few tenths either way even for a correct merge.
The rehearsal measured exactly that: merged 60.48, base + LoRA 60.08. chrF is
still printed, for information.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, heading, resolve_input, run_dir, run_stage, stage_result  # noqa: E402
from qwen_common import chrf_by_direction, generate, load_model, load_tokenizer, read_jsonl  # noqa: E402

AGREEMENT = 0.98
CHECK_PAIRS = 60


def predictions(model, tokenizer, rows: list) -> list:
    """The model's top next-token choice at every answer position, given the
    prompt and the reference answer so far (teacher forcing)."""
    import torch

    out = []
    with torch.no_grad():
        for r in rows:
            prompt = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
            answer = tokenizer(r["answer"], add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prompt + answer], device="cuda")
            logits = model(input_ids=ids).logits[0]
            # Position i predicts token i+1: the answer's tokens are predicted
            # from positions len(prompt)-1 onward.
            out.append(logits[len(prompt) - 1:len(prompt) - 1 + len(answer)].argmax(-1).tolist())
    return out


def work(job: dict) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = resolve_input(job["base_model"], "models", job)
    merged_dir = run_dir(job) / "merged"
    tuned = {k: v for k, v in stage_result(job, "Q5").items() if isinstance(v, dict)}

    heading("Merging (float32, on the CPU)")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, run_dir(job) / "lora").merge_and_unload()
    model = model.to(torch.bfloat16)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="20GB")
    load_tokenizer(base).save_pretrained(merged_dir)
    shards = sorted(p.name for p in merged_dir.glob("*.safetensors"))
    if shards != ["model.safetensors"]:
        raise Stop(f"the merged model was saved as {shards}, not one model.safetensors")
    print(f"  {merged_dir / 'model.safetensors'} ({(merged_dir / 'model.safetensors').stat().st_size / 1e6:,.0f} MB)")
    del model

    heading(f"Checking the merge: next-token agreement with base + LoRA on {CHECK_PAIRS} test pairs")
    test = read_jsonl(run_dir(job) / "data" / "test.jsonl")
    sample = test[::max(1, len(test) // CHECK_PAIRS)][:CHECK_PAIRS]
    tokenizer = load_tokenizer(merged_dir)
    with_lora = load_model(base, lora=run_dir(job) / "lora")
    expected = predictions(with_lora, tokenizer, sample)
    del with_lora
    torch.cuda.empty_cache()
    merged_model = load_model(merged_dir)
    got = predictions(merged_model, tokenizer, sample)
    same = sum(a == b for e, g in zip(expected, got) for a, b in zip(e, g))
    total = sum(len(e) for e in expected)
    agreement = same / max(total, 1)
    print(f"  the merged model picks the same next token as base + LoRA {agreement:.2%} of the time "
          f"({same:,} of {total:,} tokens; {AGREEMENT:.0%} needed)")
    if agreement < AGREEMENT:
        raise Stop("the merged model doesn't behave like base + LoRA: the merge went wrong")

    heading("For information: chrF of the merged model (greedy, the full test set)")
    hyps = generate(merged_model, tokenizer, [t["prompt"] for t in test])
    merged = chrf_by_direction(test, hyps)
    for d in merged:
        print(f"  {d:10} merged {merged[d]['chrF']:6.2f}   base + LoRA {tuned[d]['chrF']:6.2f}")
    return {"token_agreement": round(agreement, 4), "merged": merged}


if __name__ == "__main__":
    run_stage("Q6", "merge the LoRA into the model", work)
