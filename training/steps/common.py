"""What every training stage shares: the job file, machine.yaml, hf: inputs,
the run folder, disk space, and the STOP convention.

Plain Python with pathlib and no Windows-only tools, so the same stages run in
the Linux training container (train-and-convert-app.md, "Running training on
another machine"). PowerShell only ever calls these; it never does the work.
"""

import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "inputs"
RUNS = REPO / "runs"
MACHINE = REPO / "machine.yaml"


class Stop(Exception):
    """A check failed. The job must not continue."""


def heading(text: str) -> None:
    print(f"\n== {text}", flush=True)


def main(step) -> None:
    """Run a stage; print STOP and exit 1 on a failed check."""
    try:
        step()
    except Stop as e:
        print(f"\nSTOP: {e}", flush=True)
        sys.exit(1)


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_machine() -> dict:
    """machine.yaml: what differs between PCs (see machine.example.yaml)."""
    return load_yaml(MACHINE) if MACHINE.is_file() else {}


def load_job(path: str) -> dict:
    job_path = Path(path)
    if not job_path.is_file():
        raise Stop(f"there is no job file at {job_path.resolve()}")
    job = load_yaml(job_path)
    for key in ("kind", "name"):
        if key not in job:
            raise Stop(f"{job_path} has no '{key}'")
    job["_path"] = job_path.resolve()
    return job


def run_dir(job: dict) -> Path:
    return RUNS / job["name"]


# --- Stages -----------------------------------------------------------------------
# Each stage is a script run as:  <stage>.py <job file> [--force]
# A stage that finished writes runs/<job>/stages/<ID>.json, and is not repeated
# unless --force is given (train.ps1 -Force, or the stage -From starts at).


def run_stage(stage_id: str, title: str, work) -> None:
    """Run work(job) as stage stage_id of the job named on the command line."""
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("job")
    parser.add_argument("--force", action="store_true", help="redo the stage even if it finished")
    a = parser.parse_args()

    def step():
        job = load_job(a.job)
        marker = run_dir(job) / "stages" / f"{stage_id}.json"
        print(f"{stage_id}: {title}  ({job['name']})", flush=True)
        if marker.is_file() and not a.force:
            print(f"  already done ({marker}); pass --force to redo it")
            return
        started = time.time()
        result = work(job) or {}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"stage": stage_id, "seconds": round(time.time() - started),
                                      **result}, indent=2, ensure_ascii=False), encoding="utf-8")

    main(step)


def stage_result(job: dict, stage_id: str) -> dict:
    """What an earlier stage recorded when it finished."""
    import json

    marker = run_dir(job) / "stages" / f"{stage_id}.json"
    if not marker.is_file():
        raise Stop(f"stage {stage_id} hasn't finished for {job['name']}; run it first")
    return json.loads(marker.read_text(encoding="utf-8"))


# --- hf: references -------------------------------------------------------------
# A job may name a Hugging Face model or dataset instead of a folder:
#   hf:openai/whisper-large-v3-turbo
#   hf:ciempiess/ciempiess_light@train
#   hf:google/fleurs:es_419@test          (id:config@split)
# fetch.ps1 downloads each into inputs/models/<name>/ or inputs/data/<name>/.


class HfRef:
    def __init__(self, text: str):
        if not text.startswith("hf:"):
            raise ValueError(text)
        rest = text[3:]
        self.split = None
        if "@" in rest:
            rest, self.split = rest.split("@", 1)
        self.config = None
        if ":" in rest:
            rest, self.config = rest.split(":", 1)
        self.repo_id = rest
        if self.repo_id.count("/") != 1:
            raise Stop(f"{text!r} is not a Hugging Face ID like hf:owner/name")

    @property
    def folder_name(self) -> str:
        """The folder under inputs/: the repo's name, plus config and split."""
        name = self.repo_id.split("/")[1]
        if self.config:
            name += f"-{self.config}"
        if self.split:
            name += f"-{self.split}"
        return name

    def __str__(self) -> str:
        text = f"hf:{self.repo_id}"
        if self.config:
            text += f":{self.config}"
        if self.split:
            text += f"@{self.split}"
        return text


def resolve_input(value: str, kind: str, job: dict) -> Path:
    """A job's model or data entry as a folder or file on disk. kind is
    'models' or 'data'. An hf: entry must have been fetched already."""
    if value.startswith("hf:"):
        ref = HfRef(value)
        folder = INPUTS / kind / ref.folder_name
        if not folder.is_dir():
            raise Stop(
                f"{value} has not been downloaded (expected {folder}). Run:\n"
                f"    .\\fetch.ps1 {job['_path']}"
            )
        return folder
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        raise Stop(f"{value} does not exist (looked for {path})")
    return path


# --- disk space -----------------------------------------------------------------


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def gb(n: float) -> str:
    return f"{n / 1e9:,.1f} GB"


def folder_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def check_disk(needed: int, where: Path, what: str) -> None:
    """Print the estimate, and stop unless the disk has it with 20% to spare."""
    free = free_bytes(where)
    wanted = int(needed * 1.2)
    print(f"  disk: {what} needs about {gb(needed)}; with 20% to spare, {gb(wanted)}. "
          f"Free on {where.anchor or where}: {gb(free)}")
    if free < wanted:
        raise Stop(
            f"not enough disk space for {what}: about {gb(wanted)} wanted (including 20% to "
            f"spare), {gb(free)} free. Free some space, or delete finished runs under {RUNS}."
        )
