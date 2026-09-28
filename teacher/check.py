"""Item 2's check: the teacher punctuates ten CIEMPIESS lines, changing no word.

    uv run --project training python teacher/check.py

Fetches ciempiess/ciempiess_light into inputs/data/ if it isn't there (about
1.1 GB; the Whisper job needs it anyway), takes ten transcripts of five words
or more, and has the first teacher in machine.yaml restore their capitals and
punctuation, on the GPU. Prints each before and after. Passes only if no line's
words changed.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training" / "steps"))
sys.path.insert(0, str(REPO / "teacher"))

from common import INPUTS, RUNS, HfRef, Stop, heading, main  # noqa: E402
from teacher import Teacher, punctuate, teacher_ggufs, write_json  # noqa: E402

LINES = 10
MIN_WORDS = 5


def sample_lines() -> list:
    import pyarrow.parquet as pq

    from fetch import fetch
    ref = HfRef("hf:ciempiess/ciempiess_light")
    folder = INPUTS / "data" / ref.folder_name
    if not (folder / "fetched.json").is_file():
        fetch(ref, "data")
    first = sorted(folder.rglob("*.parquet"))[0]
    texts = pq.read_table(first, columns=["normalized_text"]).column(0).to_pylist()
    lines = [t.strip() for t in texts if t and len(t.split()) >= MIN_WORDS]
    # Spread through the file rather than the first ten, which may share a speaker.
    step = max(1, len(lines) // LINES)
    return lines[::step][:LINES]


def step() -> None:
    lines = sample_lines()
    gguf = teacher_ggufs()[0]
    out = RUNS / "_teacher-check"
    heading(f"Punctuating {len(lines)} CIEMPIESS lines with {gguf.name}")
    with Teacher(gguf, out) as teacher:
        results = punctuate(teacher, lines)
    changed = 0
    for i, (before, after, kept) in enumerate(results, 1):
        print(f"\n  {i:2}. before: {before}")
        print(f"      after:  {after}")
        if not kept:
            changed += 1
            print("      THROWN AWAY: the teacher changed a word")
    write_json(out / "punctuated.json",
               [{"before": b, "after": a, "kept": k} for b, a, k in results])
    print(f"\n  {len(results) - changed} of {len(results)} kept, {changed} thrown away "
          f"(saved to {out / 'punctuated.json'})")
    if changed:
        raise Stop(f"the teacher changed words in {changed} of {len(results)} lines. "
                   "The check needs every word unchanged.")
    print("\nOK. The teacher punctuates without changing a word, on the GPU.")


if __name__ == "__main__":
    main(step)
