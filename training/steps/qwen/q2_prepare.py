"""Q2 - prepare: each pair as the exact text volis sends, plus the answer.

- The prompt comes from volis (--print-prompt); the source sentence replaces
  {text}; the answer is the target followed by <|im_end|>.
- Pairs with an empty side, or a target over three times the source's length,
  are dropped (volis's own length guard would refuse such an answer).
- General pairs (FLORES+) are mixed in to make about general_share (a fifth)
  of training, so general translation doesn't get worse.
- Stops if any test sentence appears in training.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, heading, run_dir, run_stage  # noqa: E402
from qwen_common import END, LENGTH_RATIO, pairs_dir, prompt_for, read_jsonl, setting, write_jsonl  # noqa: E402

SEED = 0


def usable(p: dict) -> bool:
    s, t = p["source"].strip(), p["target"].strip()
    return bool(s) and bool(t) and len(t) <= LENGTH_RATIO * max(len(s), 1)


def work(job: dict) -> dict:
    rng = random.Random(SEED)
    train = [p for p in read_jsonl(pairs_dir(job) / "train.jsonl") if usable(p)]
    test = [p for p in read_jsonl(pairs_dir(job) / "test.jsonl") if usable(p)]
    general_path = pairs_dir(job) / "general.jsonl"
    general = [p for p in read_jsonl(general_path) if usable(p)] if general_path.is_file() else []

    share = setting(job, "general_share")
    wanted = int(len(train) * share / (1 - share)) if general else 0
    rng.shuffle(general)
    mixed = train + general[:wanted]
    rng.shuffle(mixed)

    seen_test = {p["source"].strip().lower() for p in test} | {p["target"].strip().lower() for p in test}
    leaked = [p for p in mixed if p["source"].strip().lower() in seen_test
              or p["target"].strip().lower() in seen_test]
    if leaked:
        raise Stop(f"{len(leaked)} training pairs contain a test sentence, e.g. {leaked[0]['source'][:80]!r}")

    def rows(pairs):
        return [{"prompt": prompt_for(job, p), "answer": p["target"].strip() + END, **p} for p in pairs]

    write_jsonl(run_dir(job) / "data" / "train.jsonl", rows(mixed))
    write_jsonl(run_dir(job) / "data" / "test.jsonl", rows(test))
    heading("Prepared")
    print(f"  train: {len(mixed):,} ({len(train):,} dialect pairs + {min(wanted, len(general)):,} general)")
    print(f"  test:  {len(test):,}")
    example = rows(mixed[:1])[0]
    print("\n  One training example, exactly as the model will see it:\n")
    print("  " + (example["prompt"] + example["answer"]).replace("\n", "\n  "))
    return {"train": len(mixed), "test": len(test), "general_mixed": min(wanted, len(general))}


if __name__ == "__main__":
    run_stage("Q2", "prepare the training text", work)
