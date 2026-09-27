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
# The onnxruntime sherpa-onnx 1.13.8 is built against. sherpa-onnx-core ships
# it, and cnverc.exe links it statically.
SHERPA_ORT_VERSION = "1.28.2"
ENGINE_WORKER = Path(__file__).resolve().parent / "engine_worker.py"
# This machine's paths, kept out of the committed configs.
MACHINE = REPO / "machine.yaml"
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
    machine = load_machine()
    if machine.get("cnverc_path") and not config.get("cnverc", {}).get("path"):
        config.setdefault("cnverc", {})["path"] = machine["cnverc_path"]
    return config


def load_machine() -> dict:
    """machine.yaml: what differs between PCs (see machine.example.yaml).
    A model config's own value wins over it."""
    if not MACHINE.is_file():
        return {}
    with open(MACHINE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_dir(config: dict) -> Path:
    return REPO / "runs" / config["run_name"]


def test_wav(config: dict) -> Path:
    wav = Path(config["test"]["wav"])
    if not wav.is_absolute():
        wav = REPO / wav
    if not wav.is_file():
        raise Stop(
            f"the test wav is not at {wav}. Recordings are not committed (they are voices), "
            f"so a fresh clone has none: record a few seconds in {config['language']!r} with "
            f"'cnverc --listen --wav' (it saves 16 kHz mono to logs/segments/), or convert any "
            f"clip with 'ffmpeg -i in.wav -ar 16000 -ac 1 {wav.name}', and put it there."
        )
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

    It runs in its own process (engine_worker.py) so it gets onnxruntime
    SHERPA_ORT_VERSION from sherpa-onnx-core, never the onnxruntime package a
    step may already have loaded; the worker refuses to transcribe otherwise.
    """
    result = _engine(
        "--encoder", encoder, "--decoder", decoder, "--tokens", tokens,
        "--wav", wav, "--language", language,
    )
    global _ENGINE_SHOWN
    if not _ENGINE_SHOWN:
        _ENGINE_SHOWN = True
        print(f"  (engine: {engine_description(result)})")
    return result["text"]


_ENGINE_SHOWN = False


def engine_selftest() -> dict:
    """Load cnverc's engine without a model and report which runtime it got."""
    return _engine("--selftest")


def engine_description(result: dict) -> str:
    ort = result.get("onnxruntime", {})
    where = ort.get("path", ort.get("why", ""))
    return (
        f"sherpa-onnx {result.get('sherpa_onnx')} on onnxruntime "
        f"{ort.get('version', '?')}, {where}"
    )


def _engine(*argv) -> dict:
    import subprocess

    command = [sys.executable, str(ENGINE_WORKER), "--expect-ort", SHERPA_ORT_VERSION]
    command += [str(x) for x in argv]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    result = json.loads(lines[-1]) if lines else None
    if done.returncode != 0 or not result or not result.get("ok"):
        why = result["error"] if result else f"it exited with code {done.returncode} and no result"
        tail = done.stderr.strip()[-1500:]
        raise Stop(
            f"cnverc's engine could not run: {why}"
            + (f"\n{tail}" if tail else "")
            + "\nRun whisper-to-onnx/steps/0_doctor.py to see which file or DLL is at fault."
        )
    return result


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
