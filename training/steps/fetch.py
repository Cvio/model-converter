"""Download every hf: model and dataset a job file names into inputs/.

    uv run --project training python training/steps/fetch.py jobs/es-mx-whisper.yaml

(fetch.ps1 runs exactly that.) Anything already downloaded is skipped. The job
file is never rewritten: the stages resolve hf: entries to the inputs/ folders
themselves.

Rules (train-and-convert-app.md, "Getting the inputs"):
- Models: only weights, configs, tokenizer and processor files, never training
  leftovers. The lists are read from the converter's step 2, so there is one.
- Datasets: Parquet only. A dataset published only as a loading script is taken
  from Hugging Face's automatic Parquet copy (the refs/convert/parquet branch).
- Gated datasets stop with what to do; nothing tries to get around the gate.
- Anything over 5 GB has its size printed first, and the disk is checked.
"""

import ast
import json
import sys
from pathlib import Path

from common import INPUTS, REPO, HfRef, Stop, check_disk, gb, heading, load_job, main

BIG = 5e9
PARQUET_BRANCH = "refs/convert/parquet"


def converter_lists() -> dict:
    """WEIGHTS, MODEL_FILES, TOKENIZER_FILES and LEFTOVERS from the converter's
    step 2, read as source so this environment doesn't import the converter's."""
    source = (REPO / "whisper-to-onnx" / "steps" / "2_download.py").read_text(encoding="utf-8")
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("WEIGHTS", "MODEL_FILES", "TOKENIZER_FILES", "LEFTOVERS"):
                found[name] = ast.literal_eval(node.value)
    missing = {"WEIGHTS", "MODEL_FILES", "TOKENIZER_FILES", "LEFTOVERS"} - set(found)
    if missing:
        raise Stop(f"whisper-to-onnx/steps/2_download.py no longer defines {sorted(missing)}")
    return found


def model_files(listing: dict) -> dict:
    """The files of a model repo worth downloading, from {path: size}."""
    lists = converter_lists()
    wanted_names = set(lists["WEIGHTS"]) | set(lists["MODEL_FILES"]) | set(lists["TOKENIZER_FILES"])
    # Larger models (Qwen) shard their weights and ship a chat template.
    wanted_names |= {"model.safetensors.index.json", "chat_template.jinja", "processor_config.json"}
    keep = {}
    for path, size in listing.items():
        name = path.rsplit("/", 1)[-1]
        if "/" in path or name in lists["LEFTOVERS"]:
            continue
        if name in wanted_names or (name.startswith("model-") and name.endswith(".safetensors")):
            keep[path] = size
    # Many repos publish the same weights twice. Safetensors when there are any;
    # pytorch_model.bin only for models that have nothing else.
    if any(p.endswith(".safetensors") for p in keep):
        keep = {p: s for p, s in keep.items() if not p.endswith(".bin")}
    return keep


def repo_listing(repo_id: str, repo_type: str, revision=None) -> dict:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        entries = HfApi().list_repo_tree(
            repo_id, repo_type=repo_type, revision=revision, recursive=True, expand=False
        )
        return {e.path: getattr(e, "size", 0) or 0 for e in entries if hasattr(e, "size")}
    except GatedRepoError as e:
        raise Stop(gated_message(repo_id, repo_type)) from e
    except RepositoryNotFoundError as e:
        if repo_type == "dataset" and revision is None:
            # A gated dataset can answer "not found" to an anonymous request.
            raise Stop(
                f"{repo_id} was not found on Hugging Face. If it exists and is gated, "
                + gated_message(repo_id, repo_type)
            ) from e
        raise


def gated_message(repo_id: str, repo_type: str) -> str:
    page = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
    return (
        f"{repo_id} is gated. Open {page}, accept its terms while logged in, then run\n"
        f"    uv run hf auth login\n"
        f"in this folder, and fetch again."
    )


def dataset_files(ref: HfRef) -> tuple:
    """(revision, {path: size}) of the Parquet files for a dataset reference."""
    listing = repo_listing(ref.repo_id, "dataset")
    parquet = {p: s for p, s in listing.items() if p.endswith(".parquet")}
    if parquet:
        try:
            return None, select(parquet, ref)
        except Stop as on_main:
            # Some script datasets (google/fleurs) keep unrelated Parquet files
            # on main; the config asked for may be in the automatic copy.
            first_problem = on_main
    else:
        first_problem = None
    # Published only as a loading script: use the automatic Parquet copy.
    try:
        branch = {p: s for p, s in repo_listing(ref.repo_id, "dataset", PARQUET_BRANCH).items()
                  if p.endswith(".parquet")}
    except Exception:  # noqa: BLE001 - no such branch: report the first problem
        branch = {}
    if not branch:
        if first_problem:
            raise first_problem
        raise Stop(
            f"{ref.repo_id} has no Parquet files, and no automatic Parquet copy on "
            f"{PARQUET_BRANCH}. Only Parquet datasets are fetched."
        )
    return PARQUET_BRANCH, select(branch, ref)


def select(parquet: dict, ref: HfRef) -> dict:
    """Narrow a dataset's Parquet files to the config and split asked for.

    Native Parquet repos lay files out as <config>/<split>-NNNNN-of-NNNNN.parquet
    (or data/<split>-...); the automatic copy as <config>/<split>/NNNN.parquet."""
    configs = sorted({p.split("/")[0] for p in parquet if "/" in p})
    chosen = parquet
    if ref.config:
        chosen = {p: s for p, s in chosen.items() if p.split("/")[0] == ref.config}
        if not chosen:
            raise Stop(f"{ref} names config {ref.config!r}; the dataset has {configs}")
    elif len(configs) > 1 and "default" in configs:
        chosen = {p: s for p, s in chosen.items() if p.split("/")[0] == "default"}
    if ref.split:
        def in_split(path: str) -> bool:
            parts = path.split("/")
            name = parts[-1]
            return (ref.split in parts[1:-1] or f"partial-{ref.split}" in parts[1:-1]
                    or name.startswith(f"{ref.split}-") or name == f"{ref.split}.parquet")
        chosen = {p: s for p, s in chosen.items() if in_split(p)}
        if not chosen:
            raise Stop(f"{ref} names split {ref.split!r}, and no Parquet file matches it")
    return chosen


def download(ref: HfRef, repo_type: str, files: dict, revision, folder: Path) -> None:
    from huggingface_hub import hf_hub_download

    total = sum(files.values())
    if total > BIG:
        print(f"  {ref} is {gb(total)}.")
    check_disk(total, folder.parent, str(ref))
    folder.mkdir(parents=True, exist_ok=True)
    for i, (path, size) in enumerate(sorted(files.items()), 1):
        print(f"  [{i}/{len(files)}] {path}  ({size / 1e6:,.0f} MB)", flush=True)
        hf_hub_download(ref.repo_id, path, repo_type=repo_type, revision=revision,
                        local_dir=folder)
    (folder / "fetched.json").write_text(
        json.dumps({"source": str(ref), "revision": revision or "main",
                    "files": sorted(files)}, indent=2),
        encoding="utf-8",
    )


def print_columns(folder: Path) -> None:
    import pyarrow.parquet as pq

    first = sorted(folder.rglob("*.parquet"))[0]
    schema = pq.read_schema(first)
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in folder.rglob("*.parquet"))
    print(f"  columns: {', '.join(schema.names)}")
    print(f"  rows: {rows:,}")


def entries(job: dict) -> list:
    """(hf reference, 'models' or 'data') for every hf: value the job names."""
    found = []
    def visit(value, kind):
        if isinstance(value, str) and value.startswith("hf:"):
            found.append((HfRef(value), kind))
        elif isinstance(value, dict):
            for k, v in value.items():
                if k != "columns":
                    visit(v, kind)
        elif isinstance(value, list):
            for v in value:
                visit(v, kind)
    visit(job.get("base_model"), "models")
    visit(job.get("data"), "data")
    visit(job.get("sources"), "data")
    visit(job.get("teacher_check"), "data")
    return found


def fetch(ref: HfRef, kind: str) -> Path:
    folder = INPUTS / kind / ref.folder_name
    heading(f"{ref}  ->  {folder}")
    if (folder / "fetched.json").is_file():
        print("  already downloaded")
    elif kind == "models":
        files = model_files(repo_listing(ref.repo_id, "model"))
        if not any(p.endswith((".safetensors", ".bin")) for p in files):
            raise Stop(f"{ref.repo_id} has no model.safetensors or pytorch_model.bin at its top level")
        download(ref, "model", files, None, folder)
    else:
        revision, files = dataset_files(ref)
        if revision:
            print(f"  published as a loading script; using the automatic Parquet copy ({revision})")
        download(ref, "dataset", files, revision, folder)
    if kind == "data":
        print_columns(folder)
    return folder


def step() -> None:
    if len(sys.argv) != 2:
        raise Stop("usage: fetch.py <job file>")
    job = load_job(sys.argv[1])
    found = entries(job)
    if not found:
        print("The job names nothing on Hugging Face (no hf: entries); nothing to fetch.")
        return
    for ref, kind in found:
        fetch(ref, kind)
    print(f"\nOK. Everything {job['_path'].name} names is in {INPUTS}.")


if __name__ == "__main__":
    main(step)
