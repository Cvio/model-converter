"""P2 - choose the teacher: score each teacher_gguf on FLORES+ and pick the best.

Each teacher in machine.yaml translates the FLORES+ devtest sentences in the
dialect's language (flores_code) into English, scored by chrF against FLORES+'s
own English. The highest scorer is recorded and used by P3. With one teacher,
it is still scored and the number printed.

FLORES+ has no Mexican Spanish: spa_Latn is a neutral written standard, so for
es-MX this measures general Spanish ability, the best proxy available. For
Iraqi Arabic, acm_Arab (Mesopotamian) is much closer than Standard Arabic.
Rehearsal limit: flores_limit (sentences).
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training" / "steps"))
sys.path.insert(0, str(REPO / "teacher"))

from common import INPUTS, Stop, heading, run_dir, run_stage  # noqa: E402
from teacher import Teacher, teacher_ggufs, translate  # noqa: E402


def flores(code: str, split: str) -> list:
    path = INPUTS / "data" / "flores_plus" / split / f"{code}.jsonl"
    if not path.is_file():
        raise Stop(f"{path} is missing. Run .\\fetch.ps1 on this job (FLORES+ is gated: accept "
                   "its terms at https://huggingface.co/datasets/openlanguagedata/flores_plus first)")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r["text"] for r in sorted(rows, key=lambda r: r["id"])]


def work(job: dict) -> dict:
    import sacrebleu

    code = job.get("flores_code")
    if not code:
        raise Stop("the job has no flores_code (the FLORES+ language closest to the dialect)")
    source = flores(code, "devtest")
    english = flores("eng_Latn", "devtest")
    if len(source) != len(english):
        raise Stop(f"FLORES+ devtest has {len(source)} {code} sentences but {len(english)} English")
    limit = job.get("flores_limit")
    if limit:
        source, english = source[:limit], english[:limit]
    scores = {}
    for gguf in teacher_ggufs():
        heading(f"{gguf.name}: {code} -> English on {len(source)} FLORES+ devtest sentences")
        with Teacher(gguf, run_dir(job) / "logs") as teacher:
            hyps = translate(teacher, source, job["variety_name"], "English")
        score = sacrebleu.CHRF().corpus_score(hyps, [english]).score
        scores[str(gguf)] = round(score, 2)
        print(f"  chrF {score:.1f}")
        for s, h, e in list(zip(source, hyps, english))[:3]:
            print(f"    {s}\n    teacher:   {h}\n    reference: {e}\n")
    best = max(scores, key=scores.get)
    print(f"  teacher chosen: {Path(best).name} (chrF {scores[best]})")
    return {"teacher": best, "chrF": scores}


if __name__ == "__main__":
    run_stage("P2", "choose the teacher", work)
