"""P3 - translate: the pairs, in both directions, and the general pairs.

- Dialect -> English: the transcript as spoken (the source a live session's
  recognizer produces), the teacher's English as the target.
- English -> dialect: the same pairs reversed, so the dialect side is always
  real speech from real speakers. That side is tidied first: the teacher
  removes false starts, repeats and hesitations, and nothing else. A line
  where it did anything but remove words keeps its original wording.
- General: FLORES+ dev (never devtest, which P2 scores on) in both directions,
  between the dialect's plain language (es, not es-MX: FLORES+ is standard)
  and English. The Qwen job mixes these in so general translation doesn't get
  worse.

Writes out/train.jsonl, out/test.jsonl and out/general.jsonl. The teacher's
answers are cached, so a re-run doesn't repeat hours of work.
"""

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training" / "steps"))
sys.path.insert(0, str(REPO / "teacher"))

from common import Stop, heading, run_dir, run_stage, stage_result  # noqa: E402
from pairs_common import out_dir, pair, read_jsonl, setting, write_jsonl  # noqa: E402
from teacher import (CLEAN_SYSTEM, Teacher, clean_disfluencies, is_removal_only,  # noqa: E402
                     translate, translate_system)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_teacher import flores  # noqa: E402


class Cache:
    """Teacher answers by task and line, kept in the run folder."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text("utf-8")) if path.is_file() else {}

    def key(self, task: str, line: str) -> str:
        return hashlib.sha256(f"{task}\0{line}".encode("utf-8")).hexdigest()

    def missing(self, task: str, lines: list) -> list:
        return [l for l in dict.fromkeys(lines) if self.key(task, l) not in self.data]

    def put(self, task: str, lines: list, answers: list) -> None:
        for line, answer in zip(lines, answers):
            self.data[self.key(task, line)] = answer
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")

    def get(self, task: str, line: str) -> str:
        return self.data[self.key(task, line)]


def work(job: dict) -> dict:
    variety, other = job["variety"], setting(job, "other")
    language = variety.split("-")[0]
    name = job["variety_name"]
    teacher_path = Path(stage_result(job, "P2")["teacher"])
    train = [r["text"] for r in read_jsonl(out_dir(job) / "transcripts-train.jsonl")]
    test = [r["text"] for r in read_jsonl(out_dir(job) / "transcripts-test.jsonl")]
    lines = train + test
    cache = Cache(run_dir(job) / "teacher-cache.json")
    notes = job.get("teacher_notes", "")
    to_english = "translate:" + translate_system(name, "English", notes)
    tidy = "clean:" + CLEAN_SYSTEM

    todo_t, todo_c = cache.missing(to_english, lines), cache.missing(tidy, lines)
    if todo_t or todo_c:
        heading(f"The teacher ({teacher_path.name}): {len(todo_t):,} to translate, {len(todo_c):,} to tidy")
        with Teacher(teacher_path, run_dir(job) / "logs") as teacher:
            if todo_t:
                cache.put(to_english, todo_t, translate(teacher, todo_t, name, "English", notes))
            if todo_c:
                results = clean_disfluencies(teacher, todo_c)
                cache.put(tidy, todo_c, [tidied for _, tidied, _ in results])
    else:
        print("  every line already translated and tidied (cached)")

    def pairs_for(texts: list) -> tuple:
        rows, kept_original = [], 0
        for text in texts:
            english = cache.get(to_english, text).strip()
            if not english:
                continue
            tidied = cache.get(tidy, text).strip()
            if not is_removal_only(text, tidied):
                tidied, kept_original = text, kept_original + 1
            rows.append(pair(variety, other, text, english))
            rows.append(pair(other, variety, english, tidied))
        return rows, kept_original

    train_pairs, train_kept = pairs_for(train)
    test_pairs, test_kept = pairs_for(test)
    general = []
    if job.get("flores_code"):
        for source, english in zip(flores(job["flores_code"], "dev"), flores("eng_Latn", "dev")):
            general.append(pair(language, other, source, english))
            general.append(pair(other, language, english, source))

    write_jsonl(out_dir(job) / "train.jsonl", train_pairs)
    write_jsonl(out_dir(job) / "test.jsonl", test_pairs)
    write_jsonl(out_dir(job) / "general.jsonl", general)
    heading("Pairs")
    print(f"  train:   {len(train_pairs):,} ({len(train_pairs) // 2:,} each way)")
    print(f"  test:    {len(test_pairs):,}")
    print(f"  general: {len(general):,} (FLORES+ dev, {language} <-> {other})")
    tidy_failed = train_kept + test_kept
    print(f"  tidying: the teacher did more than remove words in {tidy_failed:,} of {len(lines):,} "
          "lines; those keep their original wording")
    print("\n  Five examples:")
    for p in train_pairs[:10:2]:
        print(f"    {p['source']}\n    -> {p['target']}\n")
    if not train_pairs:
        raise Stop("no training pairs were made")
    return {"train": len(train_pairs), "test": len(test_pairs), "general": len(general),
            "tidy_kept_original": tidy_failed}


if __name__ == "__main__":
    run_stage("P3", "translate, and make the pairs", work)
