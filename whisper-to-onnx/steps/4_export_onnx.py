"""Step 4: export the converted checkpoint with sherpa-onnx's own export script.

sherpa-onnx will not load generic ONNX files: it needs a separate encoder and
decoder, a tokens file, and metadata only its export script writes. The
script is run from sherpa-onnx at the pinned tag, with the patch in
patches/ applied to a copy (never to the checkout), so it takes a checkpoint
path, reads the mel count from the checkpoint, sizes external weight files by
size, and passes dynamo=False where PyTorch needs it.

The script writes int8 copies itself; step 5 decides whether to use them.

The export is checked two ways: exactly, by comparing the ONNX files' outputs
with the checkpoint's on the same input, and in practice, by transcribing the
test recording with the engine volis runs.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from common import (
    EXPORT_SCRIPT,
    PATCH,
    Stop,
    args,
    fresh_dir,
    heading,
    load_config,
    main,
    read_json,
    run_dir,
    similarity,
    test_wav,
    transcribe_like_cnverc,
    write_json,
)

RELATIVE = Path("scripts/whisper/export-onnx.py")

# ONNX Runtime and PyTorch order their float arithmetic differently, so the
# logits of a faithful export differ by rounding. A broken export differs by
# whole units.
ONNX_TOLERANCE = 5e-2


def patched_script(into: Path) -> Path:
    """The export script with the patch applied, written to `into`."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        target = work / RELATIVE
        target.parent.mkdir(parents=True)
        # The checkout may have Windows line endings; the patch has Unix ones.
        text = EXPORT_SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n")
        target.write_text(text, encoding="utf-8", newline="\n")
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(PATCH)],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise Stop(
                f"the patch no longer applies to {EXPORT_SCRIPT}:\n{result.stderr.strip()}\n"
                f"sherpa-onnx's script has changed; update {PATCH}."
            )
        out = into / "export-onnx.patched.py"
        shutil.copy2(target, out)
        return out


def onnx_vs_pytorch(encoder: Path, decoder: Path, checkpoint: Path, wav: Path, language: str) -> float:
    """Run the exported encoder and decoder on exactly the input the PyTorch
    checkpoint gets, and return the largest difference between their logits.

    This is the exact check of the export. Transcribing is not: volis's engine
    computes its features with a different library from PyTorch Whisper's, which
    on an ambiguous stretch of audio can change a word even when the export is
    perfect.
    """
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    import torch
    import whisper

    model = whisper.load_model(str(checkpoint), device="cpu").eval()
    audio, _ = sf.read(str(wav), dtype="float32")
    audio = whisper.pad_or_trim(torch.from_numpy(audio))
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).unsqueeze(0)
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages, language=language, task="transcribe"
    )
    prompt = torch.tensor([list(tokenizer.sot_sequence_including_notimestamps)])
    with torch.no_grad():
        expected = model(mel, prompt).numpy()

    options = ort.SessionOptions()
    options.log_severity_level = 3
    enc = ort.InferenceSession(str(encoder), options, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(decoder), options, providers=["CPUExecutionProvider"])
    cross_k, cross_v = enc.run(None, {"mel": mel.numpy()})
    d = model.dims
    cache = np.zeros((d.n_text_layer, 1, d.n_text_ctx, d.n_text_state), dtype=np.float32)
    logits = dec.run(
        None,
        {
            "tokens": prompt.numpy().astype(np.int64),
            "in_n_layer_self_k_cache": cache,
            "in_n_layer_self_v_cache": cache.copy(),
            "n_layer_cross_k": cross_k,
            "n_layer_cross_v": cross_v,
            "offset": np.zeros(1, dtype=np.int64),
        },
    )[0]
    return float(np.abs(logits - expected).max())


def step() -> None:
    a = args(__doc__)
    config = load_config(a.config)
    run = run_dir(config)
    checkpoint = run / "openai" / "model.pt"
    step3 = read_json(run / "openai" / "transcripts.json", "step 3")
    if not checkpoint.is_file():
        raise Stop(f"{checkpoint} is missing. Run step 3 first.")
    out = fresh_dir(run / "onnx", a.force)
    name = config["run_name"]

    heading("Patching the export script")
    script = patched_script(out)
    print(f"  {EXPORT_SCRIPT}\n  + {PATCH.name}\n  = {script}")

    heading("Exporting (this takes a few minutes; the script uses one thread)")
    log = out / "export.log"
    with open(log, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, str(script), "--checkpoint", str(checkpoint), "--name", name],
            cwd=out,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        raise Stop("the export failed. Last lines of " + str(log) + ":\n  " + "\n  ".join(tail))
    print(f"  log: {log}")

    heading("Files written")
    expected = {
        "encoder": out / f"{name}-encoder.onnx",
        "decoder": out / f"{name}-decoder.onnx",
        "encoder_int8": out / f"{name}-encoder.int8.onnx",
        "decoder_int8": out / f"{name}-decoder.int8.onnx",
        "tokens": out / f"{name}-tokens.txt",
    }
    for role, path in expected.items():
        if not path.is_file():
            raise Stop(f"the export did not write {path}")

    # A model over 2 GB is first exported by PyTorch as one file per tensor;
    # the script then gathers them into a single <name>-<part>.weights file and
    # leaves the loose ones behind. Remove them before the check below, so it
    # proves the files that are kept are enough on their own.
    keep = {p.name for p in expected.values()} | {
        f"{name}-encoder.weights",
        f"{name}-decoder.weights",
        script.name,
        log.name,
    }
    loose = [p for p in out.iterdir() if p.is_file() and p.name not in keep]
    for path in loose:
        path.unlink()
    if loose:
        print(f"  removed {len(loose)} loose tensor files left by PyTorch's large-model export")

    for path in sorted(out.iterdir()):
        if path.is_file():
            print(f"  {path.name:45} {path.stat().st_size / 1e6:10,.1f} MB")

    # The encoder's weights: in the .onnx itself, or beside it in .weights.
    arch = read_json(run / "arch.json", "step 2")
    enc_bytes = expected["encoder"].stat().st_size
    weights = out / f"{name}-encoder.weights"
    if weights.is_file():
        enc_bytes += weights.stat().st_size
    d, layers = arch["d_model"], arch["encoder_layers"]
    rough = 4 * (12 * d * d * layers)  # fp32 attention + feed-forward weights
    print(f"  encoder weights: {enc_bytes / 1e6:,.0f} MB (expected roughly {rough / 1e6:,.0f} MB)")
    if not 0.7 * rough < enc_bytes < 1.5 * rough:
        raise Stop("the encoder is not the size its architecture implies")

    heading("Proving it: the same input, PyTorch and ONNX")
    wav = test_wav(config)
    diff = onnx_vs_pytorch(
        expected["encoder"], expected["decoder"], checkpoint, wav, config["language"]
    )
    print(f"  largest logit difference: {diff:.2e}")
    if diff > ONNX_TOLERANCE:
        raise Stop(
            f"the ONNX files compute something different from the checkpoint (logits differ by "
            f"{diff:.3g}, more than {ONNX_TOLERANCE}). Read {log}."
        )
    print(f"  within {ONNX_TOLERANCE}: the export computes what the checkpoint computes")

    heading("Transcribing with the exported files, in volis's engine")
    text = transcribe_like_cnverc(
        expected["encoder"], expected["decoder"], expected["tokens"], wav, config["language"]
    )
    agreement = similarity(step3["openai"], text)
    print(f"  step 3 (PyTorch): {step3['openai']}")
    print(f"  ONNX fp32:        {text}")
    print(
        f"  agreement: {agreement:.0%} of words. volis's engine computes its audio features "
        f"with a different library from PyTorch Whisper's, so a word may differ on an unclear "
        f"stretch; the exact check above is what proves the export."
    )
    write_json(
        out / "fp32.json",
        {"text": text, "agreement_with_step3": agreement, "max_logit_difference": diff},
    )
    if not text.strip():
        raise Stop("volis's engine transcribed nothing with the exported files")
    print("\nOK. The ONNX export computes what the checkpoint computes.")


if __name__ == "__main__":
    main(step)
