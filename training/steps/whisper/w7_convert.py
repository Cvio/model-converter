"""W7 - convert: the merged model through the existing converter, unchanged.

Writes runs/<job>/converter.yaml from the job (model_id = the merged folder,
base_model = the base model folder, language, engine, test wav) and runs the
converter's seven steps in the converter's own environment. Its run folder is
the job's run folder, so everything stays under runs/<job>/.

engine.languages is [language] unless the job says otherwise; engine.varieties
is [variety] when the job has one.
"""

import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import REPO, Stop, heading, resolve_input, run_dir, run_stage  # noqa: E402

STEPS = ("1_check_setup", "2_download", "3_to_openai_format", "4_export_onnx",
         "5_check_int8", "6_assemble", "7_verify")


def converter_config(job: dict) -> Path:
    engine = dict(job["engine"])
    engine.setdefault("languages", [job["language"]])
    if job.get("variety") and "varieties" not in engine:
        engine["varieties"] = [job["variety"]]
    use_fp32 = engine.pop("use_fp32", False)
    config = {
        "model_id": str(run_dir(job) / "merged"),
        "base_model": str(resolve_input(job["base_model"], "models", job)),
        "run_name": job["name"],
        "language": job["language"],
        "engine": engine,
        "use_fp32": use_fp32,
        "test": {"wav": job.get("test_wav", "test_audio/es-16k.wav")},
    }
    path = run_dir(job) / "converter.yaml"
    path.write_text("# Written by W7 from " + job["_path"].name + "; don't edit, edit the job.\n"
                    + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def work(job: dict) -> dict:
    if not (run_dir(job) / "merged" / "model.safetensors").is_file():
        raise Stop("there is no merged model; run W6 first")
    config = converter_config(job)
    print(f"  converter config: {config}")
    force = "--force" in sys.argv
    for step in STEPS:
        heading(f"Converter step {step}")
        command = ["uv", "run", "--project", str(REPO), "python",
                   str(REPO / "whisper-to-onnx" / "steps" / f"{step}.py"), "--config", str(config)]
        if force or step != "1_check_setup":
            # Each converter step refuses to overwrite its own earlier output;
            # W7 as a whole is the unit that is or isn't repeated.
            command.append("--force")
        done = subprocess.run(command, cwd=REPO)
        if done.returncode != 0:
            raise Stop(f"converter step {step} stopped (see above). Fix it, then rerun W7 with -Force.")
    folder = run_dir(job) / "out" / job["engine"]["folder_name"]
    return {"folder": str(folder)}


if __name__ == "__main__":
    run_stage("W7", "convert with the existing converter", work)
