"""What every step shares: the config, the run folder, and arch.json.

Nothing model-specific lives here or in any step. It all comes from the
config file named with --config.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

# model-converter/
REPO = Path(__file__).resolve().parents[2]
SHERPA = REPO / "vendor" / "sherpa-onnx"
SHERPA_TAG = "v1.13.8"  # the version cnverc links; see README
EXPORT_SCRIPT = SHERPA / "scripts" / "whisper" / "export-onnx.py"
PATCH = REPO / "whisper-to-onnx" / "patches" / "export-onnx-local-checkpoint.patch"


class Stop(Exception):
    """A check failed. The run must not continue."""


def args(description: str, extra=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="path to a config .yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="redo this step even though its output already exists",
    )
    if extra:
        extra(parser)
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise Stop(f"no config at {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key in ("model_id", "base_model", "run_name", "language", "engine", "test"):
        if key not in config:
            raise Stop(f"{config_path} has no '{key}'")
    for key in ("folder_name", "display_name", "languages"):
        if key not in config["engine"]:
            raise Stop(f"{config_path} has no 'engine.{key}'")
    config["_path"] = config_path
    return config


def run_dir(config: dict) -> Path:
    return REPO / "runs" / config["run_name"]


def test_wav(config: dict) -> Path:
    wav = Path(config["test"]["wav"])
    if not wav.is_absolute():
        wav = REPO / wav
    if not wav.is_file():
        raise Stop(f"the test wav is not at {wav}")
    return wav


def fresh_dir(path: Path, force: bool) -> Path:
    """A step's output folder. An earlier run's output is never overwritten
    unless --force is given."""
    if path.exists():
        if not force:
            raise Stop(
                f"{path} already exists. This step has run before; pass --force to redo "
                f"it, or change run_name in the config for a new run."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, made_by: str) -> dict:
    if not path.is_file():
        raise Stop(f"{path} is missing. Run {made_by} first.")
    return json.loads(path.read_text(encoding="utf-8"))


def arch(config: dict) -> dict:
    return read_json(run_dir(config) / "arch.json", "step 2")


def main(step) -> None:
    """Run a step, turning a failed check into a clear message and a non-zero
    exit rather than a traceback."""
    # Transcripts are printed; Windows consoles default to a code page that
    # can't show "país".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        step()
    except Stop as e:
        print(f"\nSTOP: {e}", file=sys.stderr)
        sys.exit(1)


def heading(text: str) -> None:
    print(f"\n== {text}")


def transcribe_like_cnverc(encoder, decoder, tokens, wav, language: str) -> str:
    """Transcribe with the engine cnverc runs, given the settings cnverc uses.

    The sherpa-onnx Python package at the version cnverc links wraps the same
    C++ recognizer, and WhisperAsr in cnverc/src/asr.rs configures it this way.
    This, not sherpa-onnx's scripts/whisper/test.py, is the judge: test.py
    decodes in its own Python loop, and on whisper-small its int8 transcript
    was far worse than what the C++ engine makes of the same files.
    """
    import os

    # Before sherpa_onnx: its extension looks for onnxruntime.dll beside
    # itself, then in folders added with add_dll_directory, then in System32,
    # which on Windows 11 holds an old copy (1.17.1) it cannot use. Point it at
    # the onnxruntime package's own DLL first.
    if hasattr(os, "add_dll_directory"):
        import onnxruntime

        os.add_dll_directory(str(Path(onnxruntime.__file__).parent / "capi"))
    import sherpa_onnx
    import soundfile as sf

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(encoder),
        decoder=str(decoder),
        tokens=str(tokens),
        language=language,
        task="transcribe",
        tail_paddings=0,
        num_threads=6,
        provider="cpu",
    )
    audio, sample_rate = sf.read(str(wav), dtype="float32")
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def similarity(a: str, b: str) -> float:
    """Word-level agreement between two transcripts, 0..1, after lowercasing and
    removing punctuation. 1.0 means the same words in the same order."""
    import difflib
    import re

    def words(s: str) -> list:
        return re.sub(r"[^\w\s]", " ", s.lower()).split()

    wa, wb = words(a), words(b)
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(None, wa, wb).ratio()
