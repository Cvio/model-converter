"""Step 2: download only what the conversion needs, and record the architecture.

model_id may be a Hugging Face ID or a local folder, such as a model the LoRA
training app has merged. Training leftovers (optimizer.pt and the rest) are
never downloaded; they are often larger than the model.
"""

import json
import shutil
from pathlib import Path

from common import Stop, args, fresh_dir, heading, load_config, main, run_dir, write_json

WEIGHTS = ("model.safetensors", "pytorch_model.bin")
MODEL_FILES = ("config.json", "generation_config.json", "preprocessor_config.json")
OPTIONAL = ("generation_config.json", "preprocessor_config.json")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
)
# trainer_state.json is not here: it is small, and read below for the error
# rate reported during training.
LEFTOVERS = (
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "training_args.bin",
)


class Source:
    """A model on Hugging Face, or a folder on disk."""

    def __init__(self, ref: str):
        self.ref = ref
        self.local = Path(ref).expanduser()
        self.is_local = self.local.is_dir()
        if self.is_local:
            self.local = self.local.resolve()

    def files(self) -> list:
        if self.is_local:
            return sorted(p.name for p in self.local.iterdir() if p.is_file())
        from huggingface_hub import list_repo_files

        try:
            return sorted(list_repo_files(self.ref))
        except Exception as e:  # noqa: BLE001 - any failure here is a stop
            raise Stop(f"cannot list {self.ref} on Hugging Face: {e}") from e

    def fetch(self, name: str, into: Path) -> Path:
        target = into / name
        if self.is_local:
            shutil.copy2(self.local / name, target)
        else:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(self.ref, name, local_dir=into)
            target = Path(downloaded)
        return target


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def step() -> None:
    a = args(__doc__)
    config = load_config(a.config)
    run = run_dir(config)
    if not (run / "setup" / "setup.json").is_file():
        raise Stop("run step 1 first")
    hf = fresh_dir(run / "hf", a.force)

    model = Source(config["model_id"])
    base = Source(config["base_model"])

    heading(f"Model: {config['model_id']}" + (" (local folder)" if model.is_local else ""))
    listing = model.files()
    print("  files: " + ", ".join(listing))

    if "adapter_config.json" in listing:
        raise Stop(
            f"{config['model_id']} is a LoRA adapter (it has adapter_config.json), not a full "
            f"model. An adapter cannot be converted on its own: merge it into its base model "
            f"first (step 6 of the LoRA training app), then point model_id at the merged model."
        )

    weights = next((w for w in WEIGHTS if w in listing), None)
    if weights is None:
        sharded = [f for f in listing if f.endswith(".safetensors.index.json")]
        raise Stop(
            f"{config['model_id']} has neither {' nor '.join(WEIGHTS)}"
            + (f" (it is sharded: {sharded[0]}; sharded weights aren't supported yet)" if sharded else "")
        )
    skipped = [f for f in LEFTOVERS if f in listing]
    if skipped:
        print(f"  not downloading training leftovers: {', '.join(skipped)}")

    heading("Downloading")
    for name in (weights, *MODEL_FILES):
        if name not in listing:
            if name in OPTIONAL:
                print(f"  {name}: not in the model; it will come from the base model")
                continue
            raise Stop(f"{config['model_id']} has no {name}")
        path = model.fetch(name, hf)
        print(f"  {name}  {path.stat().st_size / 1e6:,.1f} MB")

    base_listing = base.files()
    for name in OPTIONAL:
        if not (hf / name).is_file() and name in base_listing:
            base.fetch(name, hf)
            print(f"  {name}  (from {config['base_model']})")

    # The tokenizer comes from the base model: many fine-tunes don't ship it,
    # and a fine-tune that changed its vocabulary is refused below anyway.
    tokenizer_dir = hf / "tokenizer"
    tokenizer_dir.mkdir()
    got = []
    for name in TOKENIZER_FILES:
        if name in base_listing:
            base.fetch(name, tokenizer_dir)
            got.append(name)
    if "tokenizer.json" not in got and not {"vocab.json", "merges.txt"} <= set(got):
        raise Stop(f"{config['base_model']} has no usable tokenizer files")
    print(f"  tokenizer from {config['base_model']}: {', '.join(got)}")

    heading("Architecture, from config.json")
    cfg = read_config(hf / "config.json")
    if cfg.get("model_type") != "whisper":
        raise Stop(f"config.json says model_type={cfg.get('model_type')!r}, not 'whisper'")
    arch = {
        "encoder_layers": cfg["encoder_layers"],
        "decoder_layers": cfg["decoder_layers"],
        "d_model": cfg["d_model"],
        "encoder_attention_heads": cfg["encoder_attention_heads"],
        "decoder_attention_heads": cfg["decoder_attention_heads"],
        "encoder_ffn_dim": cfg["encoder_ffn_dim"],
        "decoder_ffn_dim": cfg["decoder_ffn_dim"],
        "vocab_size": cfg["vocab_size"],
        "num_mel_bins": cfg["num_mel_bins"],
        "max_source_positions": cfg["max_source_positions"],
        "max_target_positions": cfg["max_target_positions"],
        "weights_file": weights,
    }
    for key, value in arch.items():
        print(f"  {key:25} {value}")

    # openai-whisper's model hard-codes a feed-forward width of 4 x d_model.
    for side in ("encoder", "decoder"):
        if arch[f"{side}_ffn_dim"] != 4 * arch["d_model"]:
            raise Stop(
                f"{side}_ffn_dim is {arch[f'{side}_ffn_dim']}, not 4 x d_model "
                f"({4 * arch['d_model']}). openai-whisper's model cannot hold this architecture, "
                f"so sherpa-onnx's export cannot handle it (distilled or pruned models do this)."
            )

    # sherpa-onnx writes tokens.txt from openai-whisper's own vocabulary. A
    # fine-tune that added tokens would decode to the wrong text.
    base_config_path = hf / "tokenizer" / "_base_config.json"
    base.fetch("config.json", hf / "tokenizer")
    (hf / "tokenizer" / "config.json").rename(base_config_path)
    base_vocab = read_config(base_config_path)["vocab_size"]
    if arch["vocab_size"] != base_vocab:
        raise Stop(
            f"the model's vocab_size is {arch['vocab_size']} but {config['base_model']}'s is "
            f"{base_vocab}. It has added or removed tokens, and sherpa-onnx's tokens.txt comes "
            f"from Whisper's standard vocabulary, so it would decode to the wrong text. Check "
            f"base_model, or this model cannot be converted this way."
        )
    print(f"  vocab_size matches {config['base_model']} ({base_vocab})")

    if "trainer_state.json" in listing:
        heading("Reported during training (unverified)")
        path = model.fetch("trainer_state.json", hf)
        state = json.loads(path.read_text(encoding="utf-8"))
        metric = state.get("best_metric")
        print(
            f"  best metric: {metric}  ({state.get('best_model_checkpoint') or 'checkpoint not named'})"
            if metric is not None
            else "  no best_metric recorded"
        )
        print(
            "  This is whatever the trainer measured, usually WER on an unnamed test set. "
            "It is not a verdict on the model."
        )
        path.unlink()

    write_json(run / "arch.json", arch)
    print(f"\nOK. Wrote {run / 'arch.json'}")


if __name__ == "__main__":
    main(step)
