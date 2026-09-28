"""What the pairs job's stages share.

The pairs job makes translation pairs between a dialect and English, in both
directions, for the Qwen job to train on (train-and-convert-app.md, "Stages of
a pairs job"). The dialect side is always real speech from real speakers: the
transcripts the Whisper job prepared. A model never writes the dialect side.

Pairs are JSONL, one per line:
    {"source_lang": "es-MX", "target_lang": "en", "source": "...", "target": "..."}
"""

import json
from pathlib import Path

from common import RUNS, Stop, run_dir

PAIRS_DEFAULTS = {"other": "en", "min_words": 5, "review_rows": 50, "review_pass": 0.9}


def setting(job: dict, key: str):
    return job.get(key, PAIRS_DEFAULTS.get(key))


def out_dir(job: dict) -> Path:
    return run_dir(job) / "out"


def whisper_data(job: dict) -> Path:
    """The prepared data folder of the Whisper job the pairs come from."""
    name = job.get("from_whisper_job")
    if not name:
        raise Stop("the job has no from_whisper_job (the Whisper job whose transcripts to use)")
    folder = RUNS / name / "data"
    if not (folder / "train.parquet").is_file():
        raise Stop(f"{folder} has no prepared data; run that Whisper job's W2 first "
                   f"(.\\train.ps1 jobs\\{name}.yaml -To W2)")
    return folder


def dialect_transcripts(job: dict, split: str) -> list:
    """Transcripts from the Whisper job's split, dialect speakers only. Clips
    mixed in from ordinary speech (data.mix) have no speaker and are left out:
    they aren't the dialect."""
    import pyarrow.parquet as pq

    path = whisper_data(job) / f"{split}.parquet"
    if not path.is_file():
        return []
    rows = pq.read_table(path, columns=["text", "speaker"]).to_pylist()
    return [r["text"].strip() for r in rows if r["speaker"] and r["text"].strip()]


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list:
    if not path.is_file():
        raise Stop(f"{path} is missing; run the stage that writes it first")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pair(source_lang: str, target_lang: str, source: str, target: str) -> dict:
    return {"source_lang": source_lang, "target_lang": target_lang, "source": source, "target": target}
