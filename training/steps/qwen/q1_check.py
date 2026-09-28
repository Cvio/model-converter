"""Q1 - check: the base model is a Qwen3, the pairs are sound, volis can
print its prompt, and the pairs were approved by a fluent speaker.

- The base model folder has its weights, config.json and tokenizer, and
  config.json says Qwen3 (volis's prompt is written for Qwen's chat format).
- Every pair has the four fields and language tags volis knows (checked by
  asking volis for each pair's prompt).
- The pairs job's P4 (the fluent-speaker check) passed. A rehearsal may set
  allow_unreviewed_pairs, and must say why.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import RUNS, Stop, heading, resolve_input, run_stage  # noqa: E402
from qwen_common import pairs_dir, prompt_template, read_jsonl  # noqa: E402

FIELDS = ("source_lang", "target_lang", "source", "target")


def work(job: dict) -> dict:
    if job["kind"] != "qwen":
        raise Stop(f"{job['name']} is a '{job['kind']}' job, not a qwen job")
    for key in ("base_model", "from_pairs_job", "output"):
        if key not in job:
            raise Stop(f"the job file has no '{key}'")

    heading("Base model")
    base = resolve_input(job["base_model"], "models", job)
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    arch = config.get("architectures", [])
    print(f"  {base}\n  {arch}, {config.get('num_hidden_layers')} layers, hidden {config.get('hidden_size')}")
    if not any(a.startswith("Qwen3") for a in arch):
        raise Stop(f"{base} is {arch}, not a Qwen3 model; volis's prompt is written for Qwen3")
    if not list(base.glob("*.safetensors")) or not (base / "tokenizer.json").is_file():
        raise Stop(f"{base} is missing its safetensors weights or tokenizer.json")

    heading("Pairs")
    review = RUNS / job["from_pairs_job"] / "stages" / "P4.json"
    if not review.is_file():
        if not job.get("allow_unreviewed_pairs"):
            raise Stop(f"the pairs job {job['from_pairs_job']} hasn't passed its fluent-speaker "
                       "check (P4). Training on unreviewed pairs would teach the teacher's mistakes.")
        print(f"  NOTE: {job['from_pairs_job']} hasn't passed P4; this job allows it "
              "(allow_unreviewed_pairs). Rehearsal only.")
    counts = {}
    for name in ("train", "test", "general"):
        path = pairs_dir(job) / f"{name}.jsonl"
        if name == "general" and not path.is_file():
            continue
        rows = read_jsonl(path)
        for n, p in enumerate(rows, 1):
            missing = [f for f in FIELDS if not isinstance(p.get(f), str)]
            if missing:
                raise Stop(f"{path} line {n} has no {missing}")
        tags = sorted({(p["source_lang"], p["target_lang"]) for p in rows})
        for source, target in tags:
            prompt_template(job, source, target)   # volis refuses a tag it doesn't know
        counts[name] = len(rows)
        print(f"  {name:8} {len(rows):6,} pairs; directions {', '.join(f'{s}>{t}' for s, t in tags)}")
    return {"pairs": counts, "architecture": arch}


if __name__ == "__main__":
    run_stage("Q1", "check the model, the pairs and volis", work)
