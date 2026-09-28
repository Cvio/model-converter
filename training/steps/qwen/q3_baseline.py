"""Q3 - baseline: the base model translates the test set with volis's exact
prompt and volis's decoding (greedy). Scored by chrF per direction.

This is the full-precision model; Q8 later scores the compressed file through
volis itself, which is the fair comparison for what volis will run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import heading, resolve_input, run_dir, run_stage  # noqa: E402
from qwen_common import chrf_by_direction, generate, load_model, load_tokenizer, read_jsonl  # noqa: E402
from whisper_common import write_json  # noqa: E402


def score(job: dict, model, tokenizer, label: str) -> dict:
    test = read_jsonl(run_dir(job) / "data" / "test.jsonl")
    hyps = generate(model, tokenizer, [t["prompt"] for t in test])
    scores = chrf_by_direction(test, hyps)
    for direction, s in scores.items():
        print(f"  {label:6} {direction:10} chrF {s['chrF']:6.2f}  ({s['pairs']} pairs)", flush=True)
    write_json(run_dir(job) / "scores" / f"{label}-test.json",
               {"scores": scores, "translations": [{"source": t["source"], "reference": t["target"],
                                                    "direction": f"{t['source_lang']}>{t['target_lang']}",
                                                    "hypothesis": h} for t, h in zip(test, hyps)]})
    return scores


def work(job: dict) -> dict:
    base = resolve_input(job["base_model"], "models", job)
    heading(f"Scoring the base model ({base.name})")
    return score(job, load_model(base), load_tokenizer(base), "base")


if __name__ == "__main__":
    run_stage("Q3", "score the base model", work)
