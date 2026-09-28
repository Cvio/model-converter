"""Download the pinned prebuilt llama-server (CUDA) into teacher/llama.cpp/.

    uv run --project training python teacher/get_server.py

The teacher's llama.cpp does not have to match the one volis links: it never
ships, it only writes training text. A prebuilt CUDA 13.4 build is used
because CUDA 12.4 builds don't support the RTX 5090 (Blackwell), and 13.4
supports it and the laptop's RTX 4070 alike. No CUDA toolkit is needed: the
runtime DLLs come in their own archive.

The release is pinned, and each archive is checked against the SHA-256 GitHub
publishes for it.
"""

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training" / "steps"))

from common import Stop, heading, main  # noqa: E402

RELEASE = "b11223"
ASSETS = (f"llama-{RELEASE}-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip")
TARGET = REPO / "teacher" / "llama.cpp"


def step() -> None:
    import requests

    exe = TARGET / "llama-server.exe"
    marker = TARGET / "release.json"
    if exe.is_file() and marker.is_file() and json.loads(marker.read_text("utf-8"))["release"] == RELEASE:
        print(f"llama-server {RELEASE} is already in {TARGET}")
        return
    api = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{RELEASE}"
    release = requests.get(api, timeout=60).json()
    assets = {a["name"]: a for a in release.get("assets", [])}
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        if name not in assets:
            raise Stop(f"llama.cpp release {RELEASE} has no {name}")
        asset = assets[name]
        heading(f"{name} ({asset['size'] / 1e6:,.0f} MB)")
        data = requests.get(asset["browser_download_url"], timeout=1800).content
        digest = asset.get("digest", "")
        if digest.startswith("sha256:"):
            if hashlib.sha256(data).hexdigest() != digest.split(":", 1)[1]:
                raise Stop(f"{name} does not match the SHA-256 GitHub publishes for it")
            print("  SHA-256 matches")
        else:
            print("  (GitHub publishes no SHA-256 for this file; size checked only)")
            if len(data) != asset["size"]:
                raise Stop(f"{name} downloaded {len(data)} bytes, expected {asset['size']}")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for member in z.infolist():
                # Flatten: the archives hold the programs and DLLs at their top level
                # or in one folder; llama-server needs its DLLs beside it.
                name_only = Path(member.filename).name
                if member.is_dir() or not name_only:
                    continue
                (TARGET / name_only).write_bytes(z.read(member))
    if not exe.is_file():
        raise Stop(f"the archives did not contain llama-server.exe (looked in {TARGET})")
    marker.write_text(json.dumps({"release": RELEASE, "assets": list(ASSETS)}, indent=2), "utf-8")
    print(f"\nOK. llama-server {RELEASE} is in {TARGET}")


if __name__ == "__main__":
    main(step)
