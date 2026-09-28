"""W2 - prepare: the job's data as train, validation, test and regression
files at 16 kHz mono, split by speaker, with punctuation restored.

- Clips outside data.min_seconds..max_seconds are dropped and counted.
- Test: data.test if given; otherwise 5% of training speakers (never clips).
  Validation: 5% of the remaining training speakers.
- Speaker IDs are labelled with the dataset they come from. Separate datasets
  often number speakers independently (CIEMPIESS LIGHT and TEST both have an
  F_01, and they are different people), so a bare ID would report overlap
  where there is none. Train/test speaker overlap must be 0.
- restore_punctuation (default: on when no training transcript has any
  punctuation): the teacher adds capitals and punctuation to train and
  validation transcripts. A line whose words changed is thrown away and
  counted; over 5% thrown away stops the stage. Test and regression keep their
  transcripts: scoring removes case and punctuation anyway.
- Rehearsal limits: data.max_train_hours (whole speakers, until reached),
  data.max_test_clips, data.max_regression_clips.
- Up to 300 test clips are also written as WAV files with a manifest, for W8,
  which scores the converted model in the converter's environment.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training" / "steps"))
sys.path.insert(0, str(REPO / "teacher"))

from common import Stop, heading, run_dir, run_stage  # noqa: E402
from whisper_common import (RATE, data_dir, decode, job_setting, source_rows, to_pcm16,  # noqa: E402
                            write_split)

SEED = 0
HOLD_BACK = 0.05
MAX_THROWN = 0.05
W8_CLIPS = 300


def has_punctuation(texts: list) -> bool:
    return any(any(c in t for c in ".,;:?!¿¡") or any(ch.isupper() for ch in t) for t in texts)


def spread(rows: list, limit) -> list:
    """At most limit rows, taken evenly through the list (so many speakers)."""
    if not limit or len(rows) <= limit:
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit)]


def decoded(rows: list, source: str, job: dict, counts: dict) -> list:
    low, high = job_setting(job, "data", "min_seconds"), job_setting(job, "data", "max_seconds")
    out = []
    for r in rows:
        samples = decode(r["audio"])
        seconds = len(samples) / RATE
        if not low <= seconds <= high:
            counts["dropped_length"] = counts.get("dropped_length", 0) + 1
            continue
        if not r["text"].strip():
            counts["dropped_empty"] = counts.get("dropped_empty", 0) + 1
            continue
        out.append({"pcm": to_pcm16(samples), "text": r["text"].strip(), "seconds": seconds,
                    "speaker": f"{source}:{r['speaker']}" if r["speaker"] else ""})
    return out


def by_speaker(rows: list) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault(r["speaker"], []).append(r)
    return groups


def summary(name: str, rows: list) -> dict:
    hours = sum(r["seconds"] for r in rows) / 3600
    speakers = len({r["speaker"] for r in rows if r["speaker"]})
    print(f"  {name:11} {len(rows):6,} clips  {hours:6.2f} h  {speakers:4} speakers")
    return {"clips": len(rows), "hours": round(hours, 3), "speakers": speakers}


def restore_punctuation(job: dict, splits: dict) -> dict:
    from teacher import PUNCTUATE_SYSTEM, Teacher, punctuate, teacher_ggufs, words

    lines = [r["text"] for s in ("train", "validation") for r in splits[s]]
    heading(f"Restoring punctuation in {len(lines):,} transcripts (the teacher)")
    # The teacher's answers are cached by line and prompt, so a re-run of this
    # stage (say after a fix further down) doesn't redo hours of teacher work.
    cache_path = data_dir(job) / "punctuation-cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text("utf-8")) if cache_path.is_file() else {}
    key = hashlib.sha256(PUNCTUATE_SYSTEM.encode("utf-8")).hexdigest()[:12]
    todo = [line for line in lines if f"{key}|{line}" not in cache]
    if todo:
        print(f"  {len(lines) - len(todo):,} already done; asking the teacher for {len(todo):,}")
        with Teacher(teacher_ggufs()[0], run_dir(job) / "logs") as teacher:
            for before, answer, _ in punctuate(teacher, todo):
                cache[f"{key}|{before}"] = answer
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    results = [(line, cache[f"{key}|{line}"], words(cache[f"{key}|{line}"]) == words(line))
               for line in lines]
    fixed = {before: after for before, after, kept in results if kept}
    thrown = [(b, a) for b, a, kept in results if not kept]
    for s in ("train", "validation"):
        splits[s] = [dict(r, text=fixed[r["text"]]) for r in splits[s] if r["text"] in fixed]
    share = len(thrown) / max(len(lines), 1)
    print(f"  thrown away: {len(thrown):,} of {len(lines):,} ({share:.1%}): the teacher changed words")
    print("\n  Ten examples, before and after:")
    kept_examples = [(b, a) for b, a, k in results if k]
    for b, a in random.Random(SEED).sample(kept_examples, min(10, len(kept_examples))):
        print(f"    {b}\n    -> {a}\n")
    (data_dir(job) / "punctuation-thrown-away.json").write_text(
        json.dumps([{"before": b, "teacher": a} for b, a in thrown], indent=2, ensure_ascii=False),
        encoding="utf-8")
    if share > MAX_THROWN:
        raise Stop(f"the teacher changed words in {share:.1%} of transcripts (over 5%): it is "
                   "rewriting, not punctuating. See data/punctuation-thrown-away.json; a stronger "
                   "teacher (teacher_gguf in machine.yaml) is the fix.")
    return {"thrown_away": len(thrown), "of": len(lines)}


def add_mix(job: dict, splits: dict, counts: dict) -> dict:
    """data.mix: ordinary speech mixed into train so the model keeps general
    Spanish (the regression check). share is the mix's part of the training
    hours after mixing (0.2 = a fifth). Added after punctuation is restored:
    the mix keeps its own transcripts. Must not be the regression data."""
    mix = job["data"].get("mix")
    if not mix:
        return {}
    if mix.get("source") == job["data"].get("regression"):
        raise Stop("data.mix.source is the regression data; mixing it in would make the "
                   "regression check meaningless. Use another split (for example @train).")
    share = float(mix.get("share", 0.2))
    if not 0 < share < 1:
        raise Stop(f"data.mix.share must be between 0 and 1, not {share}")
    train_hours = sum(r["seconds"] for r in splits["train"]) / 3600
    wanted = train_hours * share / (1 - share)
    heading(f"Mixing in ordinary speech: {wanted:.2f} h ({share:.0%} of training)")
    source = source_rows(job, "mix")
    random.Random(SEED).shuffle(source)
    added, hours = [], 0.0
    for row in source:
        for clip in decoded([row], "mix", job, counts):
            added.append(clip)
            hours += clip["seconds"] / 3600
        if hours >= wanted:
            break
    if hours < wanted * 0.9:
        print(f"  NOTE: the mix source had only {hours:.2f} h; using all of it")
    splits["train"] = splits["train"] + added
    print(f"  added {len(added):,} clips, {hours:.2f} h")
    return {"clips": len(added), "hours": round(hours, 3), "share": share}


def write_w8_clips(job: dict, test: list) -> None:
    import soundfile as sf

    from whisper_common import from_pcm16
    folder = data_dir(job) / "test_wav"
    folder.mkdir(parents=True, exist_ok=True)
    # index: the clip's row in test.parquet, so W8 compares like with like.
    chosen = spread(list(range(len(test))), W8_CLIPS)
    with open(data_dir(job) / "test_wav.jsonl", "w", encoding="utf-8") as manifest:
        for i, index in enumerate(chosen):
            name = f"{i:04}.wav"
            sf.write(folder / name, from_pcm16(test[index]["pcm"]), RATE, subtype="PCM_16")
            manifest.write(json.dumps({"wav": f"test_wav/{name}", "text": test[index]["text"],
                                       "index": index}, ensure_ascii=False) + "\n")


def work(job: dict) -> dict:
    data = job["data"]
    rng = random.Random(SEED)
    counts = {}

    heading("Reading and resampling to 16 kHz mono")
    train_source = source_rows(job, "train")
    groups = by_speaker(train_source)
    speakers = sorted(groups)
    rng.shuffle(speakers)
    max_hours = data.get("max_train_hours")
    pool, hours = [], 0.0
    for spk in speakers:
        clips = decoded(groups[spk], "train", job, counts)
        pool.extend(clips)
        hours += sum(c["seconds"] for c in clips) / 3600
        if max_hours and hours >= max_hours:
            break

    pool_groups = by_speaker(pool)
    pool_speakers = [s for s in speakers if s and f"train:{s}" in pool_groups]
    if data.get("test"):
        test = decoded(spread(source_rows(job, "test"), data.get("max_test_clips")), "test", job, counts)
        held = []
    else:
        n = max(1, round(len(pool_speakers) * HOLD_BACK))
        held = [f"train:{s}" for s in pool_speakers[:n]]
        test = [r for s in held for r in pool_groups[s]]
        test = spread(test, data.get("max_test_clips"))
    remaining = [f"train:{s}" for s in pool_speakers if f"train:{s}" not in held]
    n_val = max(1, round(len(remaining) * HOLD_BACK))
    val_speakers = set(remaining[:n_val])
    validation = [r for s in val_speakers for r in pool_groups[s]]
    train = [r for s in remaining if s not in val_speakers for r in pool_groups[s]]
    regression = []
    if data.get("regression"):
        regression = decoded(spread(source_rows(job, "regression"), data.get("max_regression_clips")),
                             "regression", job, counts)
    if not pool_speakers:
        raise Stop("the training data has no speaker column, so it can't be split by speaker. "
                   "Map one under data.columns (speaker: ...).")
    splits = {"train": train, "validation": validation, "test": test, "regression": regression}

    heading("Splits")
    for name, rows in splits.items():
        summary(name, rows)
    for reason, n in counts.items():
        print(f"  {reason.replace('_', ' ')}: {n:,}")
    overlap = ({r["speaker"] for r in train} & {r["speaker"] for r in test}) - {""}
    print(f"  speaker overlap between train and test: {len(overlap)}")
    if overlap:
        raise Stop(f"{len(overlap)} speakers are in both train and test: {sorted(overlap)[:10]}")
    if not train or not validation or not test:
        raise Stop("a split is empty; the job needs more data or more speakers")

    punct = {}
    restore = data.get("restore_punctuation")
    if restore is None:
        restore = not has_punctuation([r["text"] for r in train[:500]])
    if restore:
        punct = restore_punctuation(job, splits)
    else:
        print("\n  Punctuation: kept as it is (restore_punctuation is off).")

    mixed = add_mix(job, splits, counts)

    heading("Writing")
    result = {"punctuation": punct, "dropped": counts, "mix": mixed}
    for name, rows in splits.items():
        if rows:
            path = write_split(job, name, rows)
            result[name] = summary(name, rows)
            print(f"             -> {path}")
    write_w8_clips(job, test)
    print(f"  {min(len(test), W8_CLIPS)} test clips as WAV for W8 -> {data_dir(job) / 'test_wav'}")
    return result


if __name__ == "__main__":
    run_stage("W2", "prepare the data", work)
