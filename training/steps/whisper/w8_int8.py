"""W8 - score what volis will run: the converted int8 model, on the test set.

W5 and W6 score the model in full precision; the converter's step 5 checks
the int8 files on one clip. This scores the int8 folder W7 built on up to 300
test clips (written by W2 as WAV), with volis's engine: the sherpa-onnx
recognizer the converter uses (transcribe_many_like_cnverc), so it runs in the
converter's environment, not the training one.

Compared with the merged model's transcripts of the same clips (W6's scoring
of the full-precision model, from W5's tuned transcripts). Stops if int8 is
more than 1.5 WER points worse: re-convert with engine.use_fp32: true.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training" / "steps"))
sys.path.insert(1, str(REPO / "whisper-to-onnx" / "steps"))

import tomllib  # noqa: E402

from common import Stop, heading, run_dir, run_stage  # noqa: E402  (training's common)
from textnorm import error_rates  # noqa: E402

LIMIT = 1.5


def converter_common():
    """The converter's common.py, imported under another name: both folders
    have a common.py, and this stage needs the training one as 'common'."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "converter_common", REPO / "whisper-to-onnx" / "steps" / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def work(job: dict) -> dict:
    data = run_dir(job) / "data"
    manifest = [json.loads(line) for line in
                (data / "test_wav.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    folder = run_dir(job) / "out" / job["engine"]["folder_name"]
    engine = tomllib.loads((folder / "engine.toml").read_text(encoding="utf-8"))
    files = {role: folder / name for role, name in engine["files"].items()}

    heading(f"Transcribing {len(manifest)} test clips with the int8 model, in volis's engine")
    texts = converter_common().transcribe_many_like_cnverc(
        files["encoder"], files["decoder"], files["tokens"],
        [data / m["wav"] for m in manifest], job["language"])
    refs = [m["text"] for m in manifest]
    int8 = error_rates(refs, texts, job["language"])

    tuned = json.loads((run_dir(job) / "scores" / "tuned-test.json").read_text("utf-8"))["transcripts"]
    by_index = [tuned[m["index"]]["hypothesis"] for m in manifest] if "index" in manifest[0] \
        else [t["hypothesis"] for t in tuned[:len(manifest)]]
    full = error_rates(refs, by_index, job["language"])
    print(f"  full precision (base + LoRA) WER {full['wer']:6.2f}   int8 in volis's engine WER "
          f"{int8['wer']:6.2f}   ({int8['wer'] - full['wer']:+.2f}, {int8['clips']} clips)")
    (run_dir(job) / "scores" / "int8-test.json").write_text(json.dumps(
        {**int8, "transcripts": [{"reference": r, "hypothesis": h} for r, h in zip(refs, texts)]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    if int8["wer"] - full["wer"] > LIMIT:
        raise Stop(f"the int8 model is {int8['wer'] - full['wer']:.2f} WER points worse than full "
                   f"precision (limit {LIMIT}). Set engine.use_fp32: true in the job and rerun W7.")
    return {"int8_wer": round(int8["wer"], 3), "full_precision_wer": round(full["wer"], 3)}


if __name__ == "__main__":
    run_stage("W8", "score the int8 model volis will run", work)
