"""W4 - train: a LoRA on Whisper's attention layers, encoder and decoder.

LoRA on q_proj, k_proj, v_proj and out_proj in both the encoder (where accent
is heard) and the decoder. Validation WER every training.eval_every_steps;
the LoRA with the lowest validation WER is kept, not the last one.

Quiet failure points, all handled here:
- With gradient checkpointing, enable_input_require_grads() is needed or the
  decoder learns nothing. It only hooks the decoder's token embeddings, though:
  the encoder's input comes from a frozen convolution, so its checkpointed
  layers would get no gradients either, and the encoder LoRA would silently
  stay at zero. A hook on the encoder's first convolution fixes that, and the
  stage checks after the first optimizer step that encoder LoRA weights moved.
- forced_decoder_ids = None and suppress_tokens = [] before training.
- Labels are padded with -100, so padding isn't learned as text.

Memory: a few real steps are measured first; if the GPU is over 90% full, the
batch is halved and accumulation doubled (the effective batch stays the same).
"""

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, gb, heading, resolve_input, run_dir, run_stage  # noqa: E402
from textnorm import error_rates  # noqa: E402
from whisper_common import (RATE, from_pcm16, job_setting, load_processor, read_split,  # noqa: E402
                            transcribe, write_json)

TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]
SEED = 0


def build_model(base: Path, job: dict):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float32)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # The encoder's input: make the first convolution's output require grad.
    model.model.encoder.conv1.register_forward_hook(lambda m, i, out: out.requires_grad_(True))
    lora = LoraConfig(r=job_setting(job, "lora", "rank"), lora_alpha=job_setting(job, "lora", "alpha"),
                      lora_dropout=job_setting(job, "lora", "dropout"), target_modules=TARGETS)
    model = get_peft_model(model, lora)
    return model.to("cuda")


def encoder_gets_gradients(model) -> bool:
    """Whether every encoder LoRA weight has a nonzero gradient right now.
    (Checked on gradients, not on weights: the learning rate warms up from 0,
    so the first optimizer step moves nothing anywhere.)"""
    grads = [p.grad for n, p in model.named_parameters() if "encoder" in n and "lora_B" in n]
    return bool(grads) and all(g is not None and g.abs().sum().item() > 0 for g in grads)


def batches(rows: list, size: int, rng: random.Random) -> list:
    order = list(range(len(rows)))
    rng.shuffle(order)
    return [[rows[i] for i in order[s:s + size]] for s in range(0, len(order), size)]


def collate(batch: list, processor, language: str):
    import torch

    features = processor.feature_extractor([from_pcm16(r["audio"]) for r in batch],
                                           sampling_rate=RATE, return_tensors="pt").input_features
    processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")
    ids = [processor.tokenizer(r["text"]).input_ids for r in batch]
    start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    # The model builds decoder inputs by shifting labels right and adding the
    # start token itself, so labels must not begin with it.
    ids = [i[1:] if i and i[0] == start else i for i in ids]
    longest = max(len(i) for i in ids)
    labels = torch.full((len(ids), longest), -100, dtype=torch.long)
    for n, i in enumerate(ids):
        labels[n, :len(i)] = torch.tensor(i)
    return features.to("cuda"), labels.to("cuda")


def validation_wer(model, processor, rows: list, language: str) -> float:
    model.eval()
    model.config.use_cache = True
    hyps = transcribe(model, processor, [from_pcm16(r["audio"]) for r in rows], language)
    model.config.use_cache = False
    model.train()
    return error_rates([r["text"] for r in rows], hyps, language)["wer"]


def fit_batch(model, processor, rows: list, job: dict, batch_size: int, accumulation: int) -> tuple:
    """Measure a few real steps; halve the batch while over 90% of the GPU."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    while True:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        longest = sorted(rows, key=lambda r: -r["seconds"])[:batch_size]
        try:
            for _ in range(2):
                features, labels = collate(longest, processor, job["language"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_features=features, labels=labels).loss
                loss.backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError:
            peak = total
            model.zero_grad(set_to_none=True)
        share = peak / total
        print(f"  batch {batch_size}: peak {gb(peak)} of {gb(total)} ({share:.0%})")
        if share <= 0.9:
            return batch_size, accumulation
        if batch_size == 1:
            raise Stop("even a batch of one fills over 90% of the GPU; this model needs a bigger card")
        batch_size, accumulation = batch_size // 2, accumulation * 2
        print(f"  over 90%: halving the batch to {batch_size}, accumulation to {accumulation}")


def work(job: dict) -> dict:
    import torch
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    base = resolve_input(job["base_model"], "models", job)
    processor = load_processor(base)
    train, validation = read_split(job, "train"), read_split(job, "validation")
    lora_dir = run_dir(job) / "lora"

    heading("Model")
    model = build_model(base, job)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA on {', '.join(TARGETS)}, encoder and decoder: {trainable:,} trainable parameters")

    heading("Memory")
    batch_size, accumulation = fit_batch(model, processor, train, job,
                                         job_setting(job, "training", "batch_size"),
                                         job_setting(job, "training", "gradient_accumulation"))
    epochs = job_setting(job, "training", "epochs")
    steps_per_epoch = math.ceil(len(train) / batch_size / accumulation)
    total_steps = steps_per_epoch * epochs
    eval_every = job_setting(job, "training", "eval_every_steps")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=job_setting(job, "training", "learning_rate"))
    schedule = get_linear_schedule_with_warmup(optimizer, job_setting(job, "training", "warmup_steps"),
                                               total_steps)

    heading(f"Training: {epochs} epochs, {total_steps} steps (batch {batch_size} x {accumulation})")
    best = {"wer": validation_wer(model, processor, validation, job["language"]), "step": 0}
    print(f"  step     0: validation WER {best['wer']:.2f} (before training)", flush=True)
    model.save_pretrained(lora_dir)
    history = [{"step": 0, "validation_wer": round(best["wer"], 3)}]
    rng = random.Random(SEED)
    step, started, checked_encoder = 0, time.time(), False
    model.train()
    for epoch in range(epochs):
        running = []
        groups = batches(train, batch_size, rng)
        for n, batch in enumerate(groups, 1):
            features, labels = collate(batch, processor, job["language"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=features, labels=labels).loss / accumulation
            loss.backward()
            running.append(loss.item() * accumulation)
            if n % accumulation and n != len(groups):
                continue
            if not checked_encoder:
                checked_encoder = True
                if not encoder_gets_gradients(model):
                    raise Stop("some of the encoder's LoRA weights got no gradient: training "
                               "would leave the encoder unchanged (see this stage's notes)")
                print("  checked: every encoder LoRA weight receives gradients", flush=True)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % eval_every == 0 or step == total_steps:
                wer = validation_wer(model, processor, validation, job["language"])
                mean_loss = sum(running) / len(running)
                running = []
                marker = ""
                if wer < best["wer"]:
                    best = {"wer": wer, "step": step}
                    model.save_pretrained(lora_dir)
                    marker = "  <- best so far, saved"
                elapsed = (time.time() - started) / 60
                print(f"  step {step:5}: loss {mean_loss:.3f}, validation WER {wer:.2f}"
                      f"  ({elapsed:.1f} min){marker}", flush=True)
                history.append({"step": step, "loss": round(mean_loss, 4), "validation_wer": round(wer, 3)})
    write_json(run_dir(job) / "scores" / "training-history.json", history)
    print(f"\n  kept the LoRA from step {best['step']} (validation WER {best['wer']:.2f}) in {lora_dir}")
    if best["step"] == 0:
        raise Stop("validation WER never went below where it started; training didn't help. "
                   "Check the learning rate and the data.")
    return {"best_step": best["step"], "best_validation_wer": round(best["wer"], 3),
            "batch_size": batch_size, "gradient_accumulation": accumulation, "steps": total_steps}


if __name__ == "__main__":
    run_stage("W4", "train the LoRA", work)
