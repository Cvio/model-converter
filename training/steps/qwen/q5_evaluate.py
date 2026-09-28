"""Q5 - evaluate: chrF per direction, base against tuned (base + LoRA), and
20 examples side by side. Stops if either direction got worse.

Read the examples too: a score can rise while the output gets stiffer.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, heading, resolve_input, run_dir, run_stage, stage_result  # noqa: E402
from qwen_common import load_model, load_tokenizer  # noqa: E402
from q3_baseline import score  # noqa: E402


def work(job: dict) -> dict:
    base_scores = {k: v for k, v in stage_result(job, "Q3").items() if isinstance(v, dict)}
    base = resolve_input(job["base_model"], "models", job)
    heading("Scoring the tuned model (base + LoRA)")
    tuned = score(job, load_model(base, lora=run_dir(job) / "lora"), load_tokenizer(base), "tuned")

    heading("Base against tuned")
    worse = []
    for d in tuned:
        b, t = base_scores[d]["chrF"], tuned[d]["chrF"]
        print(f"  {d:10} chrF {b:6.2f} -> {t:6.2f} ({t - b:+.2f})")
        if t < b:
            worse.append(d)

    heading("Twenty examples")
    scores = run_dir(job) / "scores"
    b_rows = json.loads((scores / "base-test.json").read_text("utf-8"))["translations"]
    t_rows = json.loads((scores / "tuned-test.json").read_text("utf-8"))["translations"]
    for b, t in list(zip(b_rows, t_rows))[::max(1, len(b_rows) // 20)][:20]:
        print(f"  [{b['direction']}] {b['source']}\n    reference: {b['reference']}\n"
              f"    base:      {b['hypothesis']}\n    tuned:     {t['hypothesis']}\n")
    if worse:
        raise Stop(f"translation got worse in {', '.join(worse)}; the LoRA didn't help there")
    return tuned


if __name__ == "__main__":
    run_stage("Q5", "evaluate base against tuned", work)
