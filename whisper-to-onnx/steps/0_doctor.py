"""Step 0: check this machine's environment before anything runs on it.

Needs no config. setup.ps1 runs it after installing, and step 1 runs it first.

  1. Every native file (.dll, .pyd) the installed packages list is present and
     unaltered. Security software that quarantines a DLL leaves a package
     that imports fine until the step that needs that DLL.
  2. cnverc's engine loads, on the onnxruntime cnverc links (in its own
     process), and torch and onnxruntime import (in another).
  3. Informational: other onnxruntime.dll copies on this machine, and whether
     McAfee/Trellix is installed, with the folder to ask to exclude.

Writes runs/_doctor.json, so two machines can be compared line by line.
"""

import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from common import (
    REPO,
    SHERPA_ORT_VERSION,
    Stop,
    engine_description,
    engine_selftest,
    heading,
    main,
    write_json,
)

NATIVE = {".dll", ".pyd", ".so"}
# McAfee / Trellix Endpoint Security services.
SECURITY_SERVICES = ["mfemms", "mfefire", "mfeesp", "mfetp", "masvc", "mfeatp"]
SECURITY_FOLDERS = ["McAfee", "Trellix"]


def sha256_record(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()


def check_native_files() -> dict:
    missing, altered, unreadable, checked = [], [], [], 0
    for dist in importlib.metadata.distributions():
        # RECORD itself, not dist.files: since Python 3.12 dist.files leaves
        # out files that no longer exist, which are exactly the ones wanted.
        for row in csv.reader((dist.read_text("RECORD") or "").splitlines()):
            if not row or Path(row[0]).suffix.lower() not in NATIVE:
                continue
            name, hash_ = row[0], row[1] if len(row) > 1 else ""
            path = Path(dist.locate_file(name))
            label = f"{dist.metadata['Name']}: {name}"
            if not path.is_file():
                missing.append(label)
                continue
            checked += 1
            if not hash_.startswith("sha256="):
                continue
            try:
                if sha256_record(path) != hash_.removeprefix("sha256="):
                    altered.append(label)
            except OSError as e:  # locked or access denied, e.g. mid-scan
                unreadable.append(f"{label} ({e.strerror})")
    return {"checked": checked, "missing": missing, "altered": altered, "unreadable": unreadable}


def check_imports() -> dict:
    # Also which onnxruntime.dll the package really loaded: sherpa-onnx-core
    # puts a second copy in the environment's Scripts folder, beside python.exe.
    code = (
        "import json, os, torch, onnxruntime\n"
        "dll = None\n"
        "if os.name == 'nt':\n"
        "    from engine_worker import loaded_modules, file_version\n"
        "    found = [p for p in loaded_modules() if p.lower().endswith('\\\\onnxruntime.dll')]\n"
        "    dll = [f'{p} ({file_version(p)})' for p in found]\n"
        "print(json.dumps({'torch': torch.__version__, 'onnxruntime': onnxruntime.__version__,"
        " 'onnxruntime_dll': dll}))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=Path(__file__).parent,
    )
    if done.returncode != 0:
        return {"ok": False, "error": done.stderr.strip()[-1500:]}
    return {"ok": True, **json.loads(done.stdout.strip().splitlines()[-1])}


def other_onnxruntime_copies() -> list:
    if os.name != "nt":
        return []
    places = [Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"]
    places += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    found = {}
    for place in places:
        candidate = place / "onnxruntime.dll"
        if candidate.is_file():
            found.setdefault(os.path.normcase(os.path.realpath(candidate)), str(candidate))
    for root in security_folders():
        try:
            for candidate in root.rglob("onnxruntime*.dll"):
                found.setdefault(os.path.normcase(os.path.realpath(candidate)), str(candidate))
        except OSError:
            pass
    return sorted(found.values())


def security_folders() -> list:
    roots = []
    for env in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
        base = os.environ.get(env)
        if not base:
            continue
        for name in SECURITY_FOLDERS:
            if (Path(base) / name).is_dir():
                roots.append(Path(base) / name)
    return roots


def security_software() -> dict:
    if os.name != "nt":
        return {"services": [], "folders": []}
    running = []
    for name in SECURITY_SERVICES:
        done = subprocess.run(["sc.exe", "query", name], capture_output=True, text=True)
        if done.returncode == 0:
            running.append(name)
    return {"services": running, "folders": [str(p) for p in security_folders()]}


def with_ort_version(path: str) -> str:
    """'path (version)' for a DLL, using the engine worker's version reader."""
    try:
        from engine_worker import file_version

        return f"{path} ({file_version(path)})"
    except Exception:
        return path


def step() -> None:
    report = {
        "python": platform.python_version(),
        "python_home": sys.base_prefix,
        "venv": sys.prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uv_cache_dir": os.environ.get("UV_CACHE_DIR", "(uv's default, in your user profile)"),
        "uv_python_install_dir": os.environ.get("UV_PYTHON_INSTALL_DIR", "(uv's default)"),
    }
    problems = []

    heading("Where things are")
    for key in ("python", "python_home", "venv", "uv_cache_dir", "uv_python_install_dir"):
        print(f"  {key:22} {report[key]}")
    outside = [
        p for p in (sys.base_prefix, sys.prefix) if not Path(p).resolve().is_relative_to(REPO)
    ]
    if outside:
        print(
            f"  note: {', '.join(outside)} is outside {REPO}. setup.ps1 keeps Python and the "
            f"environment inside it, so one exclusion covers everything."
        )
    if platform.machine().lower() not in ("amd64", "x86_64"):
        problems.append(f"this is a {platform.machine()} machine; only x64 has been tested")

    heading("Native files the packages installed")
    files = check_native_files()
    report["native_files"] = files
    print(f"  {files['checked']} present")
    for kind in ("missing", "altered", "unreadable"):
        for label in files[kind]:
            print(f"  {kind.upper():10} {label}")
    if files["missing"] or files["altered"]:
        problems.append(
            f"{len(files['missing'])} native file(s) missing and {len(files['altered'])} altered "
            f"since install. That is what security software quarantining or 'repairing' a DLL "
            f"looks like."
        )

    heading("cnverc's engine (its own process)")
    try:
        engine = engine_selftest()
        report["engine"] = engine
        print(f"  {engine_description(engine)}")
    except Stop as e:
        report["engine"] = {"ok": False, "error": str(e)}
        print(f"  FAILED: {e}")
        problems.append("cnverc's engine does not load on the runtime cnverc links")

    heading("torch and onnxruntime (another process)")
    imports = check_imports()
    report["imports"] = imports
    if imports["ok"]:
        print(f"  torch {imports['torch']}, onnxruntime {imports['onnxruntime']}")
        dlls = imports.get("onnxruntime_dll") or []
        for dll in dlls:
            print(f"  loaded {dll}")
        if imports.get("onnxruntime_dll") == []:
            print("  (the onnxruntime package links its runtime into its own extension; no "
                  "onnxruntime.dll is loaded, so it cannot collide with sherpa-onnx's)")
        if dlls and not any(f"({imports['onnxruntime']}" in d for d in dlls):
            problems.append(
                f"the onnxruntime package ({imports['onnxruntime']}) is running on another "
                f"onnxruntime.dll: {dlls}"
            )
    else:
        print(f"  FAILED:\n{imports['error']}")
        problems.append("torch or onnxruntime does not import")

    heading("Other onnxruntime.dll copies on this machine (information)")
    copies = other_onnxruntime_copies()
    report["other_onnxruntime_dlls"] = copies
    for copy in copies:
        print(f"  {with_ort_version(copy)}")
    if not copies:
        print("  none found in System32, on PATH, or under McAfee/Trellix")
    else:
        print(
            f"  These are harmless as long as the engine line above names the copy in the "
            f"environment and onnxruntime {SHERPA_ORT_VERSION}."
        )

    heading("Security software")
    security = security_software()
    report["security_software"] = security
    if security["services"] or security["folders"]:
        print(f"  McAfee/Trellix present: services {security['services'] or 'none'}, "
              f"folders {security['folders'] or 'none'}")
        print(
            f"  If anything above is MISSING or FAILED, ask whoever manages it for an on-access "
            f"scan exclusion for this folder and everything in it:\n"
            f"      {REPO}\n"
            f"  ('Python ML toolchain: uv-managed Python, a virtual environment and PyTorch / "
            f"ONNX Runtime DLLs installed from PyPI; the files are replaced when it is "
            f"reinstalled.')\n"
            f"  Then reinstall: .\\setup.ps1 -Reinstall"
        )
    else:
        print("  no McAfee or Trellix found")

    report["problems"] = problems
    out = REPO / "runs" / "_doctor.json"
    out.parent.mkdir(exist_ok=True)
    write_json(out, report)
    if problems:
        raise Stop(
            "\n  - " + "\n  - ".join(problems)
            + f"\nThe details are above and in {out}. After fixing, run .\\setup.ps1 -Reinstall."
        )
    print(f"\nOK. This machine can run every step. Wrote {out}")


if __name__ == "__main__":
    main(step)
