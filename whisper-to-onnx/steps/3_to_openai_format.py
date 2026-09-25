"""Step 3: rename every weight from Hugging Face's names to OpenAI's, then
prove the conversion by transcribing the test recording both ways.

This is the step that breaks. A wrong mapping still exports to ONNX without
complaint and produces a model that transcribes fluent nonsense, so the two
transcripts must agree before anything else runs.

How the mapping is built, so that nothing is written from memory:
- The OpenAI names are the exact keys of openai-whisper's own model, built
  with the dimensions from arch.json.
- Each is turned into its Hugging Face name with WHISPER_MAPPING, copied from
  transformers' convert_openai_to_hf.py (v4.46.0,
  src/transformers/models/whisper/convert_openai_to_hf.py; a copy is in
  vendor/transformers/). That script converts in the other direction, so it is
  applied forward here, to each OpenAI key, which inverts it exactly.
"""

import torch

from common import (
    Stop,
    arch as load_arch,
    args,
    fresh_dir,
    heading,
    load_config,
    main,
    run_dir,
    similarity,
    test_wav,
    write_json,
)

# Copied verbatim from transformers' convert_openai_to_hf.py (see the module
# docstring). Order matters: it is applied as a sequence of substring
# replacements, exactly as rename_keys() there applies it.
WHISPER_MAPPING = {
    "blocks": "layers",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
    ".attn.query": ".self_attn.q_proj",
    ".attn.key": ".self_attn.k_proj",
    ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm",
    ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj",
    ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj",
    ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj",
    "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.",
    "token_embedding": "embed_tokens",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "ln_post": "layer_norm",
}

# Float rounding between two implementations of the same computation stays far
# below this; a misplaced weight is far above it.
LOGIT_TOLERANCE = 1e-2

# Hugging Face keeps these; OpenAI's format does not store them.
#   proj_out.weight: the output projection, tied to the token embedding.
DROPPED = {"proj_out.weight"}


def openai_to_hf(key: str) -> str:
    new_key = key
    for old, new in WHISPER_MAPPING.items():
        if old in key:
            new_key = new_key.replace(old, new)
    # transformers' model keeps the encoder and decoder under "model.".
    return "model." + new_key


def dims_from(arch: dict) -> dict:
    """OpenAI's ModelDimensions, from config.json's values only."""
    return {
        "n_mels": arch["num_mel_bins"],
        "n_audio_ctx": arch["max_source_positions"],
        "n_audio_state": arch["d_model"],
        "n_audio_head": arch["encoder_attention_heads"],
        "n_audio_layer": arch["encoder_layers"],
        "n_vocab": arch["vocab_size"],
        "n_text_ctx": arch["max_target_positions"],
        "n_text_state": arch["d_model"],
        "n_text_head": arch["decoder_attention_heads"],
        "n_text_layer": arch["decoder_layers"],
    }


def load_hf_state(hf_dir, weights_file: str) -> dict:
    path = hf_dir / weights_file
    if weights_file.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)


def transcribe_hf(hf_dir, wav, language: str) -> str:
    import soundfile as sf
    from transformers import (
        AutoTokenizer,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir / "tokenizer"))
    # The model's own feature extractor settings, which carry its mel count.
    features = WhisperFeatureExtractor.from_pretrained(str(hf_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(hf_dir), torch_dtype=torch.float32
    ).eval()
    audio, sr = sf.read(str(wav), dtype="float32")
    inputs = features(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            inputs.input_features,
            language=language,
            task="transcribe",
            num_beams=1,
            do_sample=False,
        )
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


def transcribe_openai(checkpoint, wav, language: str) -> str:
    import soundfile as sf
    import whisper

    model = whisper.load_model(str(checkpoint), device="cpu")
    audio, _ = sf.read(str(wav), dtype="float32")
    result = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        temperature=0.0,
        beam_size=None,
        best_of=None,
        fp16=False,
        condition_on_previous_text=False,
    )
    return result["text"].strip()


def compare_logits(hf_dir, checkpoint, wav, language: str) -> float:
    """Feed both models the same features and the same decoder prompt, and
    return the largest difference between their output logits.

    This is the exact check. Transcripts can differ through decoding settings
    alone; the logits of a correctly mapped model agree to within float
    rounding, and a misplaced weight moves them by whole units.
    """
    import soundfile as sf
    import whisper
    from transformers import WhisperForConditionalGeneration

    oa = whisper.load_model(str(checkpoint), device="cpu").eval()
    hf = WhisperForConditionalGeneration.from_pretrained(
        str(hf_dir), torch_dtype=torch.float32
    ).eval()

    audio, _ = sf.read(str(wav), dtype="float32")
    audio = whisper.pad_or_trim(torch.from_numpy(audio))
    mel = whisper.log_mel_spectrogram(audio, n_mels=oa.dims.n_mels).unsqueeze(0)

    tokenizer = whisper.tokenizer.get_tokenizer(
        oa.is_multilingual, num_languages=oa.num_languages, language=language, task="transcribe"
    )
    prompt = torch.tensor([list(tokenizer.sot_sequence_including_notimestamps)])

    with torch.no_grad():
        oa_logits = oa(mel, prompt)
        hf_logits = hf(input_features=mel, decoder_input_ids=prompt).logits
    return (oa_logits - hf_logits).abs().max().item()


def step() -> None:
    a = args(__doc__)
    config = load_config(a.config)
    run = run_dir(config)
    arch = load_arch(config)
    out = fresh_dir(run / "openai", a.force)
    hf = run / "hf"

    heading("Building the mapping")
    import whisper.model as wm

    dims = dims_from(arch)
    print("  dims: " + ", ".join(f"{k}={v}" for k, v in dims.items()))
    skeleton = wm.Whisper(wm.ModelDimensions(**dims))
    expected = skeleton.state_dict()
    mapping = {key: openai_to_hf(key) for key in expected}

    hf_state = load_hf_state(hf, arch["weights_file"])
    print(f"  {len(expected)} tensors expected in OpenAI format, {len(hf_state)} in the model")

    # Both lists must be empty. On a model with a different layer count or a
    # different base, this is the first thing that catches it.
    missing = sorted(key for key, hf_key in mapping.items() if hf_key not in hf_state)
    used = set(mapping.values())
    unconsumed = sorted(k for k in hf_state if k not in used and k not in DROPPED)
    if missing:
        print("  expected but not found:\n    " + "\n    ".join(f"{k} <- {mapping[k]}" for k in missing))
    if unconsumed:
        print("  in the model but not used:\n    " + "\n    ".join(unconsumed))
    if missing or unconsumed:
        raise Stop(
            f"{len(missing)} expected weight(s) missing and {len(unconsumed)} unused. The mapping "
            f"does not fit this model; do not export it."
        )

    state = {}
    for key, hf_key in mapping.items():
        tensor = hf_state[hf_key]
        if tuple(tensor.shape) != tuple(expected[key].shape):
            raise Stop(
                f"{hf_key} has shape {tuple(tensor.shape)} but OpenAI's {key} needs "
                f"{tuple(expected[key].shape)}"
            )
        state[key] = tensor.to(torch.float32).contiguous()

    if "proj_out.weight" in hf_state:
        tied = torch.equal(
            hf_state["proj_out.weight"].to(torch.float32),
            state["decoder.token_embedding.weight"],
        )
        if not tied:
            raise Stop(
                "proj_out.weight differs from the token embedding. OpenAI's format has no "
                "separate output projection, so this model cannot be stored in it."
            )
        print("  proj_out.weight equals the token embedding (tied); dropped")
    print(f"  all {len(state)} tensors mapped, every shape matches")

    checkpoint = out / "model.pt"
    torch.save({"dims": dims, "model_state_dict": state}, checkpoint)
    print(f"  wrote {checkpoint}  ({checkpoint.stat().st_size / 1e6:,.0f} MB)")
    del hf_state, state, skeleton

    heading("Proving it: the same input, both formats")
    wav = test_wav(config)
    language = config["language"]
    diff = compare_logits(hf, checkpoint, wav, language)
    print(f"  largest logit difference: {diff:.2e}")
    if diff > LOGIT_TOLERANCE:
        raise Stop(
            f"the converted model's outputs differ from the original's by {diff:.3g} (more than "
            f"{LOGIT_TOLERANCE}). A weight is in the wrong place; do not export this."
        )
    print(f"  within {LOGIT_TOLERANCE}: every weight is where it should be")

    hf_text = transcribe_hf(hf, wav, language)
    print(f"  Hugging Face: {hf_text}")
    openai_text = transcribe_openai(checkpoint, wav, language)
    print(f"  OpenAI:       {openai_text}")
    agreement = similarity(hf_text, openai_text)
    print(f"  agreement: {agreement:.0%} of words")

    write_json(
        out / "transcripts.json",
        {
            "hugging_face": hf_text,
            "openai": openai_text,
            "agreement": agreement,
            "max_logit_difference": diff,
        },
    )
    if not openai_text.strip():
        raise Stop("the converted checkpoint transcribed nothing")
    # The logits already prove the mapping. The two libraries decode a little
    # differently, so a word or two may differ; much more means something else
    # is wrong, such as the language or the test recording.
    if agreement < 0.8:
        raise Stop(
            "the transcripts differ by far more than decoding differences explain. Check the "
            "language and the test recording before going further."
        )
    print("\nOK. The converted checkpoint computes what the original computes.")


if __name__ == "__main__":
    main(step)
