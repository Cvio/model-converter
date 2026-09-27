"""Step 1: check the tools, and read the export script that everything else
has to satisfy.

Prints the versions in use, makes sure sherpa-onnx is checked out at the tag
volis links, and reads scripts/whisper/export-onnx.py as it is on disk: its
behaviour has changed between versions, so nothing is assumed about it.
"""

import importlib.metadata
import platform
import re
import subprocess
import sys
from pathlib import Path

from common import (
    EXPORT_SCRIPT,
    PATCH,
    SHERPA,
    SHERPA_TAG,
    Stop,
    args,
    fresh_dir,
    heading,
    load_config,
    main,
    run_dir,
    test_wav,
    write_json,
)


def git(*argv, cwd=SHERPA) -> str:
    result = subprocess.run(
        ["git", *argv], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise Stop(f"git {' '.join(argv)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def doctor() -> None:
    """Step 0, in its own process: it loads volis's engine, which must not
    share a process with anything else."""
    heading("This machine (0_doctor.py)")
    sys.stdout.flush()
    done = subprocess.run([sys.executable, str(Path(__file__).with_name("0_doctor.py"))])
    if done.returncode != 0:
        raise Stop("0_doctor.py found a problem with this machine; see above.")


def step() -> None:
    a = args(__doc__)
    config = load_config(a.config)
    doctor()
    run = run_dir(config)
    run.mkdir(parents=True, exist_ok=True)
    setup_dir = fresh_dir(run / "setup", a.force)

    heading("Versions")
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "onnx", "onnxruntime", "openai-whisper"):
        versions[package] = importlib.metadata.version(package)
    for name, version in versions.items():
        print(f"  {name:15} {version}")

    import torch

    torch_version = tuple(int(x) for x in re.findall(r"\d+", torch.__version__)[:2])
    needs_no_dynamo = torch_version >= (2, 9)
    print(
        f"\n  PyTorch {torch.__version__}: "
        + (
            "the export needs dynamo=False (the newer exporter fails on Whisper's "
            "positional embeddings); the patch passes it."
            if needs_no_dynamo
            else "dynamo=False is not needed."
        )
    )

    heading(f"sherpa-onnx at {SHERPA_TAG}")
    if not SHERPA.is_dir():
        print(f"  cloning into {SHERPA}")
        SHERPA.parent.mkdir(parents=True, exist_ok=True)
        git(
            "clone",
            # Byte-identical on every machine, whatever this one's git defaults to.
            "-c",
            "core.autocrlf=false",
            "--depth",
            "1",
            "--branch",
            SHERPA_TAG,
            "https://github.com/k2-fsa/sherpa-onnx.git",
            str(SHERPA),
            cwd=SHERPA.parent,
        )
    commit = git("rev-parse", "HEAD")
    described = git("describe", "--tags", "--always")
    print(f"  {SHERPA}\n  commit {commit} ({described})")
    if described != SHERPA_TAG:
        raise Stop(
            f"sherpa-onnx is at {described}, not {SHERPA_TAG}. volis links {SHERPA_TAG}, and "
            f"the ONNX metadata must match what its runtime reads. Delete {SHERPA} and rerun."
        )
    if git("status", "--porcelain"):
        raise Stop(
            f"{SHERPA} has local changes. The patch is applied to a copy, never to the "
            f"checkout; restore it with 'git -C {SHERPA} checkout .'"
        )

    heading("What the export script does")
    if not EXPORT_SCRIPT.is_file():
        raise Stop(f"no export script at {EXPORT_SCRIPT}")
    source = EXPORT_SCRIPT.read_text(encoding="utf-8")

    choices = re.search(r'"--model".*?choices=\[(.*?)\]', source, re.S)
    names = re.findall(r'"([^"]+)"', choices.group(1)) if choices else []
    print(f"  --model accepts only these names: {', '.join(names) or '(none found)'}")

    loads_by_name = "whisper.load_model(name)" in source
    print(
        "  loading: "
        + (
            "whisper.load_model(name) for standard names, fixed file names for distil and "
            "icefall models. No option takes a checkpoint path, so the patch adds one."
            if loads_by_name
            else "not the expected whisper.load_model(name); read the script and check the "
            "patch still fits."
        )
    )

    mels_by_name = bool(re.search(r"if args\.model in .*?n_mels = 128", source, re.S))
    print(
        "  n_mels: "
        + (
            "chosen from the model NAME (128 only for large, large-v3, turbo and "
            "distil-large-v3*), not from the checkpoint, for the input the export is traced "
            "with. A large-v3 or turbo fine-tune loaded from a file would get 80 and the "
            "export would fail. The patch takes it from model.dims.n_mels."
            if mels_by_name
            else "not chosen by name in this version; check the patch still applies cleanly."
        )
    )

    external_by_name = '"large" in filename or "turbo" in filename' in source
    print(
        "  large models: "
        + (
            "weights are saved to a separate .weights file only when the NAME contains "
            "'large' or 'turbo'. The patch decides by size instead, so a model over the 2 GB "
            "ONNX limit is handled whatever it is called."
            if external_by_name
            else "external-data handling differs from what the patch expects."
        )
    )
    print("  int8: the script quantizes both files itself (quantize_dynamic, MatMul, QInt8).")

    heading("The patch")
    if not PATCH.is_file():
        raise Stop(f"no patch at {PATCH}")
    git("apply", "--check", str(PATCH))
    print(f"  {PATCH.name} applies cleanly to {SHERPA_TAG}")

    heading("The test recording")
    wav = test_wav(config)
    import soundfile as sf

    info = sf.info(str(wav))
    print(f"  {wav}: {info.samplerate} Hz, {info.channels} channel(s), {info.duration:.1f} s")
    if info.samplerate != 16000 or info.channels != 1:
        raise Stop(
            f"{wav} must be 16 kHz mono. Convert it with: "
            f"ffmpeg -i in.wav -ar 16000 -ac 1 out.wav"
        )

    write_json(
        setup_dir / "setup.json",
        {
            "versions": versions,
            "sherpa_onnx": {"tag": SHERPA_TAG, "commit": commit},
            "dynamo_false": needs_no_dynamo,
        },
    )
    print(f"\nOK. Wrote {setup_dir / 'setup.json'}")


if __name__ == "__main__":
    main(step)
