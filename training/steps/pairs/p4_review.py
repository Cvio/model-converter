"""P4 - the fluent-speaker check.

Writes out/review.csv: review_rows (50) random test pairs, dialect -> English,
with an empty "ok" column, and stops, asking for a fluent speaker of the
dialect to mark each row y or n. Run again after marking: at review_pass (90%)
y or more the job passes. Below that it stops: the teacher isn't good enough,
and everything the Qwen job learns would be capped by its mistakes.

The file opens in Excel. Save it as CSV (UTF-8) after marking.
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, heading, run_stage  # noqa: E402
from pairs_common import out_dir, read_jsonl, setting  # noqa: E402

SEED = 0


def work(job: dict) -> dict:
    path = out_dir(job) / "review.csv"
    rows_wanted = setting(job, "review_rows")
    if not path.is_file():
        pairs = [p for p in read_jsonl(out_dir(job) / "test.jsonl") if p["source_lang"] == job["variety"]]
        sample = random.Random(SEED).sample(pairs, min(rows_wanted, len(pairs)))
        # utf-8-sig: so Excel shows accents and Arabic correctly.
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([job["variety_name"], "English (teacher)", "ok"])
            for p in sample:
                writer.writerow([p["source"], p["target"], ""])
        raise Stop(
            f"a fluent {job['variety_name']} speaker needs to check {len(sample)} translations.\n"
            f"  1. Open {path} (Excel works).\n"
            f"  2. In the 'ok' column, write y if the English is a correct, natural translation "
            f"of the {job['variety_name']}, or n if it isn't.\n"
            f"  3. Save it (as CSV), then run this stage again."
        )
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    marks = [r.get("ok", "").strip().lower() for r in rows]
    unmarked = [i + 2 for i, m in enumerate(marks) if m not in ("y", "n")]
    if unmarked:
        raise Stop(f"{len(unmarked)} rows of {path} aren't marked y or n yet (rows {unmarked[:10]}...)")
    share = marks.count("y") / len(marks)
    heading("The fluent-speaker check")
    print(f"  {marks.count('y')} of {len(marks)} marked y ({share:.0%}); {setting(job, 'review_pass'):.0%} needed")
    if share < setting(job, "review_pass"):
        wrong = [r for r, m in zip(rows, marks) if m == "n"]
        for r in wrong[:15]:
            print(f"    n: {list(r.values())[0]}\n       {list(r.values())[1]}")
        raise Stop("the teacher's translations aren't good enough: the Qwen job would learn its "
                   "mistakes. Try a stronger teacher (teacher_gguf in machine.yaml), then rerun P2-P4 "
                   "with -Force, deleting review.csv first.")
    return {"approved": marks.count("y"), "of": len(marks)}


if __name__ == "__main__":
    run_stage("P4", "the fluent-speaker check", work)
