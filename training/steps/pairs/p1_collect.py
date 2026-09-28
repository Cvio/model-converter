"""P1 - collect: the dialect transcripts that will become pairs.

From the Whisper job named by from_whisper_job: its train and validation
splits become training pairs, its test split (test speakers only) becomes the
pair test set, so no test speaker's sentence ever reaches translation
training. Lines under min_words are dropped (fragments like "para que sea"
teach nothing), and duplicates removed. Rehearsal limit: max_lines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, heading, run_stage  # noqa: E402
from pairs_common import dialect_transcripts, out_dir, setting, write_jsonl  # noqa: E402


def collect(job: dict, splits: tuple, seen: set) -> tuple:
    kept, short, duplicate = [], 0, 0
    for split in splits:
        for text in dialect_transcripts(job, split):
            if len(text.split()) < setting(job, "min_words"):
                short += 1
            elif text.lower() in seen:
                duplicate += 1
            else:
                seen.add(text.lower())
                kept.append(text)
    return kept, short, duplicate


def spread(rows: list, limit) -> list:
    if not limit or len(rows) <= limit:
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit)]


def work(job: dict) -> dict:
    if job["kind"] != "pairs":
        raise Stop(f"{job['name']} is a '{job['kind']}' job, not a pairs job")
    for key in ("variety", "variety_name"):
        if not job.get(key):
            raise Stop(f"the job has no {key} (for example es-MX and 'Mexican Spanish')")
    seen = set()
    # Test first, so a sentence said by both a test and a training speaker
    # stays in test and out of training.
    test, t_short, t_dup = collect(job, ("test",), seen)
    train, short, dup = collect(job, ("train", "validation"), seen)
    train = spread(train, job.get("max_lines"))
    test = spread(test, job.get("max_test_lines"))
    heading("Transcripts")
    print(f"  train: {len(train):,} kept ({short:,} under {setting(job, 'min_words')} words, {dup:,} duplicates)")
    print(f"  test:  {len(test):,} kept ({t_short:,} under {setting(job, 'min_words')} words, {t_dup:,} duplicates)")
    if not train or not test:
        raise Stop("no transcripts left for train or test")
    write_jsonl(out_dir(job) / "transcripts-train.jsonl", [{"text": t} for t in train])
    write_jsonl(out_dir(job) / "transcripts-test.jsonl", [{"text": t} for t in test])
    return {"train": len(train), "test": len(test)}


if __name__ == "__main__":
    run_stage("P1", "collect the dialect transcripts", work)
