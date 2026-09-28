"""W5 - evaluate: the base model against the tuned one (base + the kept LoRA).

Same data, cleaning and decoding as W3. Prints ten test clips where the two
differ, side by side. Stops if the test error didn't go down, or if the
regression error (ordinary speech) rose by more than one point.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, heading, resolve_input, run_dir, run_stage, stage_result  # noqa: E402
from whisper_common import load_model, load_processor  # noqa: E402
from w3_baseline import score  # noqa: E402

REGRESSION_LIMIT = 1.0


def work(job: dict) -> dict:
    base_scores = {k: v for k, v in stage_result(job, "W3").items() if isinstance(v, dict)}
    base = resolve_input(job["base_model"], "models", job)
    lora = run_dir(job) / "lora"
    if not (lora / "adapter_config.json").is_file():
        raise Stop(f"{lora} has no LoRA; run W4 first")

    heading("Scoring the tuned model (base + LoRA)")
    tuned = score(job, load_model(base, lora=lora), load_processor(base), "tuned")

    heading("Base against tuned")
    for split in tuned:
        b, t = base_scores[split], tuned[split]
        print(f"  {split:11} WER {b['wer']:6.2f} -> {t['wer']:6.2f} ({t['wer'] - b['wer']:+.2f})   "
              f"CER {b['cer']:6.2f} -> {t['cer']:6.2f} ({t['cer'] - b['cer']:+.2f})")

    heading("Ten test clips where they differ")
    scores = run_dir(job) / "scores"
    b_rows = json.loads((scores / "base-test.json").read_text("utf-8"))["transcripts"]
    t_rows = json.loads((scores / "tuned-test.json").read_text("utf-8"))["transcripts"]
    shown = 0
    for b, t in zip(b_rows, t_rows):
        if b["hypothesis"] != t["hypothesis"] and shown < 10:
            shown += 1
            print(f"  reference: {b['reference']}\n  base:      {b['hypothesis']}\n"
                  f"  tuned:     {t['hypothesis']}\n")

    if tuned["test"]["wer"] >= base_scores["test"]["wer"]:
        raise Stop(f"the test error didn't go down ({base_scores['test']['wer']:.2f} -> "
                   f"{tuned['test']['wer']:.2f}); the LoRA didn't help")
    if "regression" in tuned:
        rise = tuned["regression"]["wer"] - base_scores["regression"]["wer"]
        limit = job.get("regression_limit", REGRESSION_LIMIT)
        if limit != REGRESSION_LIMIT:
            print(f"  NOTE: this job sets regression_limit to {limit} (the default is {REGRESSION_LIMIT}).")
        if rise > limit:
            raise Stop(f"ordinary speech got worse: regression WER rose {rise:.2f} points "
                       f"(limit {limit}). The model learned the dialect at the cost of general "
                       "Spanish. Mixing ordinary speech into training (data.mix) is the usual fix.")
    print("  Listen too: the numbers can improve while speech gets worse (see the instructions).")
    return tuned


if __name__ == "__main__":
    run_stage("W5", "evaluate base against tuned", work)
