"""Q4 - train: a LoRA on all of Qwen's linear layers.

Only the answer is learned: every token of volis's prompt (up to and
including the empty <think></think> block) has label -100; the translation and
its closing <|im_end|> are trained, so the model learns to stop.

Memory: a few real steps on the longest examples are measured first; if the
GPU is over 90% full, the batch is halved and accumulation doubled. A tenth of
the training pairs (at least 20) is held back as validation, scored by loss
each epoch; the LoRA with the lowest validation loss is kept.
"""

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, gb, heading, resolve_input, run_dir, run_stage  # noqa: E402
from qwen_common import read_jsonl, setting  # noqa: E402
from whisper_common import write_json  # noqa: E402

SEED = 0
MAX_TOKENS = 512


def encode(tokenizer, rows: list) -> list:
    out = []
    for r in rows:
        prompt = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        answer = tokenizer(r["answer"], add_special_tokens=False)["input_ids"]
        ids = (prompt + answer)[:MAX_TOKENS]
        labels = ([-100] * len(prompt) + answer)[:MAX_TOKENS]
        if any(l != -100 for l in labels):
            out.append((ids, labels))
    return out


def collate(batch: list, pad_id: int):
    import torch

    longest = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for n, (ids, lab) in enumerate(batch):
        input_ids[n, :len(ids)] = torch.tensor(ids)
        labels[n, :len(lab)] = torch.tensor(lab)
        mask[n, :len(ids)] = 1
    return input_ids.to("cuda"), labels.to("cuda"), mask.to("cuda")


def loss_on(model, examples: list, batch_size: int, pad_id: int) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(examples), batch_size):
            ids, labels, mask = collate(examples[s:s + batch_size], pad_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            total += loss.item() * len(examples[s:s + batch_size])
            count += len(examples[s:s + batch_size])
    model.train()
    return total / max(count, 1)


def fit_batch(model, examples, batch_size, accumulation, pad_id):
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    longest = sorted(examples, key=lambda e: -len(e[0]))
    while True:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            ids, labels, mask = collate(longest[:batch_size], pad_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            peak = total
        share = peak / total
        print(f"  batch {batch_size}: peak {gb(peak)} of {gb(total)} ({share:.0%})")
        if share <= 0.9:
            return batch_size, accumulation
        if batch_size == 1:
            raise Stop("even a batch of one fills over 90% of the GPU")
        batch_size, accumulation = batch_size // 2, accumulation * 2


def work(job: dict) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

    from qwen_common import load_tokenizer

    torch.manual_seed(SEED)
    base = resolve_input(job["base_model"], "models", job)
    tokenizer = load_tokenizer(base)
    rows = read_jsonl(run_dir(job) / "data" / "train.jsonl")
    examples = encode(tokenizer, rows)
    random.Random(SEED).shuffle(examples)
    n_val = max(20, len(examples) // 10)
    validation, train = examples[:n_val], examples[n_val:]
    lora_dir = run_dir(job) / "lora"

    heading("Model")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=setting(job, "lora", "rank"), lora_alpha=setting(job, "lora", "alpha"),
        lora_dropout=setting(job, "lora", "dropout"), target_modules="all-linear", task_type="CAUSAL_LM"))
    model.to("cuda")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA on every linear layer: {trainable:,} trainable parameters")
    print(f"  {len(train):,} training examples, {len(validation):,} held back for validation")

    heading("Memory")
    batch_size, accumulation = fit_batch(model, train, setting(job, "training", "batch_size"),
                                         setting(job, "training", "gradient_accumulation"),
                                         tokenizer.pad_token_id)
    epochs = setting(job, "training", "epochs")
    steps = math.ceil(len(train) / batch_size / accumulation) * epochs
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=setting(job, "training", "learning_rate"))
    schedule = get_linear_schedule_with_warmup(optimizer, max(1, steps // 20), steps)

    heading(f"Training: {epochs} epochs, {steps} steps (batch {batch_size} x {accumulation})")
    best = {"loss": loss_on(model, validation, batch_size, tokenizer.pad_token_id), "epoch": 0}
    print(f"  epoch 0: validation loss {best['loss']:.4f} (before training)", flush=True)
    model.save_pretrained(lora_dir)
    history, started, rng = [{"epoch": 0, "validation_loss": round(best["loss"], 4)}], time.time(), random.Random(SEED)
    model.train()
    for epoch in range(1, epochs + 1):
        order = train[:]
        rng.shuffle(order)
        batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
        running = []
        for n, batch in enumerate(batches, 1):
            ids, labels, mask = collate(batch, tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / accumulation
            loss.backward()
            running.append(loss.item() * accumulation)
            if n % accumulation and n != len(batches):
                continue
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
        val = loss_on(model, validation, batch_size, tokenizer.pad_token_id)
        mark = ""
        if val < best["loss"]:
            best = {"loss": val, "epoch": epoch}
            model.save_pretrained(lora_dir)
            mark = "  <- best so far, saved"
        print(f"  epoch {epoch}: training loss {sum(running) / len(running):.4f}, validation loss "
              f"{val:.4f}  ({(time.time() - started) / 60:.1f} min){mark}", flush=True)
        history.append({"epoch": epoch, "training_loss": round(sum(running) / len(running), 4),
                        "validation_loss": round(val, 4)})
    write_json(run_dir(job) / "scores" / "training-history.json", history)
    if best["epoch"] == 0:
        raise Stop("validation loss never went below where it started; training didn't help")
    print(f"\n  kept the LoRA from epoch {best['epoch']} in {lora_dir}")
    return {"best_epoch": best["epoch"], "best_validation_loss": round(best["loss"], 4),
            "batch_size": batch_size, "gradient_accumulation": accumulation, "steps": steps}


if __name__ == "__main__":
    run_stage("Q4", "train the LoRA", work)
