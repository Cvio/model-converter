"""What the Whisper job's stages share: reading the job's data, audio at
16 kHz mono, the prepared data files, loading a model, and transcribing.

Prepared data (runs/<job>/data/<split>.parquet) has one row per clip:
    audio   16 kHz mono PCM, int16, as bytes
    text    the transcript (punctuated, for train and validation)
    speaker who is speaking ("" when the data has no speaker column)
    seconds the clip's length
Everything after W2 reads only these files.
"""

import io
import json
from pathlib import Path

import numpy as np

from common import Stop, resolve_input, run_dir

RATE = 16000

WHISPER_DEFAULTS = {
    "lora": {"rank": 32, "alpha": 64, "dropout": 0.05},
    "training": {"learning_rate": 1.0e-4, "warmup_steps": 50, "epochs": 3, "batch_size": 8,
                 "gradient_accumulation": 2, "eval_every_steps": 200},
    "data": {"min_seconds": 0.5, "max_seconds": 30},
}


def job_setting(job: dict, section: str, key: str):
    return job.get(section, {}).get(key, WHISPER_DEFAULTS.get(section, {}).get(key))


def data_dir(job: dict) -> Path:
    return run_dir(job) / "data"


# --- Reading the job's inputs -----------------------------------------------------


def source_rows(job: dict, which: str) -> list:
    """Rows of dicts {audio, text, speaker} from data.<which> (train, test or
    regression): a folder of .parquet, or a .jsonl with audio paths."""
    data = job.get("data", {})
    if which == "mix":
        mix = data.get("mix") or {}
        if not mix.get("source"):
            return []
        path = resolve_input(mix["source"], "data", job)
        columns = dict(mix.get("columns", {}))
    else:
        if not data.get(which):
            return []
        path = resolve_input(data[which], "data", job)
        columns = dict(data.get("columns", {}))
        if which == "regression" and data.get("regression_columns"):
            columns = dict(data["regression_columns"])
    names = {"audio": columns.get("audio", "audio"), "text": columns.get("text", "text"),
             "speaker": columns.get("speaker")}
    if path.is_file() and path.suffix == ".jsonl":
        return jsonl_rows(path, names)
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise Stop(f"{path} holds no .parquet files (data.{which})")
    import pyarrow.parquet as pq

    wanted = [names["audio"], names["text"]] + ([names["speaker"]] if names["speaker"] else [])
    rows = []
    for f in files:
        schema = pq.read_schema(f).names
        missing = [c for c in wanted if c not in schema]
        if missing:
            raise Stop(f"{f} has no column {missing} (data.{which}); it has {schema}. "
                       f"Map the names under data.columns in the job file.")
        table = pq.read_table(f, columns=wanted).to_pylist()
        for r in table:
            rows.append({"audio": r[names["audio"]], "text": r[names["text"]] or "",
                         "speaker": str(r[names["speaker"]]) if names["speaker"] else ""})
    return rows


def jsonl_rows(path: Path, names: dict) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            audio = path.parent / r[names["audio"]]
            if not audio.is_file():
                raise Stop(f"{path} line {n}: {audio} does not exist")
            rows.append({"audio": {"path": str(audio)}, "text": r[names["text"]],
                         "speaker": str(r.get(names["speaker"] or "speaker", ""))})
    return rows


def decode(audio) -> np.ndarray:
    """Any audio cell (Hugging Face {bytes, path}, raw bytes, or a path) as
    16 kHz mono float32."""
    import librosa
    import soundfile as sf

    if isinstance(audio, dict):
        data = audio.get("bytes")
        source = io.BytesIO(data) if data else audio.get("path")
    elif isinstance(audio, (bytes, bytearray)):
        source = io.BytesIO(audio)
    else:
        source = audio
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if rate != RATE:
        samples = librosa.resample(samples, orig_sr=rate, target_sr=RATE)
    return samples.astype(np.float32)


# --- Prepared data ------------------------------------------------------------------


def to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()


def from_pcm16(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767


def write_split(job: dict, split: str, rows: list) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = data_dir(job) / f"{split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "audio": [r["pcm"] for r in rows],
        "text": [r["text"] for r in rows],
        "speaker": [r["speaker"] for r in rows],
        "seconds": [r["seconds"] for r in rows],
    })
    pq.write_table(table, out)
    return out


def read_split(job: dict, split: str) -> list:
    import pyarrow.parquet as pq

    path = data_dir(job) / f"{split}.parquet"
    if not path.is_file():
        raise Stop(f"{path} is missing; run the prepare stage (W2) first")
    return pq.read_table(path).to_pylist()


# --- Models and transcribing --------------------------------------------------------


def load_processor(base: Path):
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(base)


def load_model(folder: Path, lora: Path = None, dtype=None):
    """A Whisper model on the GPU, optionally with a LoRA attached."""
    import torch
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(folder, torch_dtype=dtype or torch.float32)
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora)
    return model.to("cuda").eval()


def transcribe(model, processor, clips: list, language: str, batch_size: int = 16) -> list:
    """Transcripts for 16 kHz float32 clips, with the language and task set so
    Whisper never guesses the language. Greedy decoding."""
    import torch

    texts = []
    for start in range(0, len(clips), batch_size):
        batch = clips[start:start + batch_size]
        features = processor.feature_extractor(batch, sampling_rate=RATE, return_tensors="pt").input_features
        features = features.to("cuda", dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            ids = model.generate(input_features=features, language=language, task="transcribe",
                                 num_beams=1, do_sample=False)
        texts.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return [t.strip() for t in texts]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, made_by: str) -> dict:
    if not path.is_file():
        raise Stop(f"{path} is missing; run {made_by} first")
    return json.loads(path.read_text(encoding="utf-8"))
