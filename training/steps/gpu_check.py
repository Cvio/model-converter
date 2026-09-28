"""Check this machine can train: PyTorch, CUDA, the GPU, and a ten-step LoRA.

    uv run --project training python training/steps/gpu_check.py

setup.ps1 runs it after installing the training environment. The ten steps
train a LoRA on whisper-small with made-up audio features: the point is that
the whole path (CUDA PyTorch, peft, the GPU, backward passes) works, not what
it learns. whisper-small is fetched into inputs/models/whisper-small/ the first
time (about 1 GB).
"""

import time

from common import INPUTS, HfRef, Stop, gb, heading, main

STEPS = 10


def step() -> None:
    import peft
    import torch
    import transformers

    heading("Versions")
    print(f"  torch          {torch.__version__}")
    print(f"  CUDA (torch)   {torch.version.cuda}")
    print(f"  transformers   {transformers.__version__}")
    print(f"  peft           {peft.__version__}")

    heading("GPU")
    if not torch.cuda.is_available():
        raise Stop("PyTorch sees no CUDA GPU. Check the NVIDIA driver is installed and up to "
                   "date (nvidia-smi should list the card), then run .\\setup.ps1 -Reinstall.")
    if torch.version.cuda is None or tuple(int(x) for x in torch.version.cuda.split(".")[:2]) < (12, 8):
        raise Stop(f"PyTorch is built for CUDA {torch.version.cuda}; 12.8 or newer is needed "
                   "(the RTX 5090 fails on older builds). training/pyproject.toml pins the cu128 index.")
    props = torch.cuda.get_device_properties(0)
    print(f"  {props.name}, {gb(props.total_memory)}, compute capability {props.major}.{props.minor}")
    if props.total_memory < 12e9:
        print("  NOTE: under 12 GB. Enough for this check and for Qwen 1.7B; Whisper turbo "
              "training belongs on the 5090 desktop.")

    heading(f"A {STEPS}-step LoRA on whisper-small")
    from fetch import fetch
    folder = INPUTS / "models" / "whisper-small"
    if not (folder / "fetched.json").is_file():
        fetch(HfRef("hf:openai/whisper-small"), "models")
    model = transformers.WhisperForConditionalGeneration.from_pretrained(folder).to("cuda")
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    lora = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                           target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
    model = peft.get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters: {trainable:,}")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)

    torch.manual_seed(0)
    mels = model.config.num_mel_bins
    features = torch.randn(4, mels, 3000, device="cuda")
    labels = torch.randint(0, 1000, (4, 12), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    losses = []
    model.train()
    for _ in range(STEPS):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_features=features, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    torch.cuda.synchronize()
    seconds = time.time() - started
    print(f"  loss: {losses[0]:.2f} -> {losses[-1]:.2f} over {STEPS} steps, {seconds:.1f} s")
    print(f"  peak GPU memory: {gb(torch.cuda.max_memory_allocated())}")
    if not losses[-1] < losses[0]:
        raise Stop("the loss did not go down on a batch trained ten times over; training is not "
                   "learning. Check the PyTorch and peft versions printed above.")
    print("\nOK. This machine can train.")


if __name__ == "__main__":
    main(step)
