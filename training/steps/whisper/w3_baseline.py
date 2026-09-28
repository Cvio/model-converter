"""W3 - baseline: score the base model on test and regression data.

The language and task are set on every generate call, so Whisper never
guesses the language. Scored by WER and CER after textnorm.normalize (one
cleaning function for every stage). Transcripts are saved beside the scores,
for W5's side-by-side examples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import heading, resolve_input, run_dir, run_stage  # noqa: E402
from textnorm import error_rates  # noqa: E402
from whisper_common import from_pcm16, load_model, load_processor, read_split, transcribe, write_json  # noqa: E402


def score(job: dict, model, processor, label: str) -> dict:
    results = {}
    for split in ("test", "regression"):
        path = run_dir(job) / "data" / f"{split}.parquet"
        if not path.is_file():
            continue
        rows = read_split(job, split)
        hyps = transcribe(model, processor, [from_pcm16(r["audio"]) for r in rows], job["language"])
        rates = error_rates([r["text"] for r in rows], hyps, job["language"])
        print(f"  {label:6} {split:11} WER {rates['wer']:6.2f}  CER {rates['cer']:6.2f}  "
              f"({rates['clips']} clips, {rates['words']:,} words)", flush=True)
        write_json(run_dir(job) / "scores" / f"{label}-{split}.json",
                   {**rates, "transcripts": [{"reference": r["text"], "hypothesis": h}
                                             for r, h in zip(rows, hyps)]})
        results[split] = {k: round(v, 3) if isinstance(v, float) else v for k, v in rates.items()}
    return results


def work(job: dict) -> dict:
    base = resolve_input(job["base_model"], "models", job)
    heading(f"Scoring the base model ({base.name})")
    model = load_model(base)
    return score(job, model, load_processor(base), "base")


if __name__ == "__main__":
    run_stage("W3", "score the base model", work)
