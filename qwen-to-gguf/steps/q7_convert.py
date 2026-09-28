"""Q7 - convert: the merged Qwen model to the .gguf file volis runs.

volis links llama-cpp-2 0.1.156, which builds llama.cpp at a pinned commit.
The GGUF is written by exactly that commit, so the file is written by the same
llama.cpp that will read it (train-and-convert-app.md, "Qwen -> GGUF"):

1. llama.cpp is cloned at LLAMA_CPP_COMMIT into qwen-to-gguf/llama.cpp/ (not
   committed). The Cargo registry's copy can't be used: it has no gguf-py and
   no quantize tool.
2. convert_hf_to_gguf.py (with that clone's own gguf-py) writes a full-size GGUF.
3. llama-quantize, built from the same clone with CMake (no CUDA needed),
   compresses it to output.quantization (Q4_K_M, like volis's current model).

The commit is recorded in runs/<job>/gguf/llama.cpp-commit.txt. The file is
checked to be roughly the size of volis's current translator.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training" / "steps"))

from common import Stop, heading, run_dir, run_stage  # noqa: E402

# llama-cpp-2 0.1.156's llama.cpp submodule (utilityai/llama-cpp-rs, tag 0.1.156).
LLAMA_CPP_COMMIT = "e79e4bf660e19f2ad851e06c6913f7a8c5852621"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"
CLONE = REPO / "qwen-to-gguf" / "llama.cpp"
VSWHERE = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")


def run(command: list, **kwargs) -> None:
    print(f"  $ {' '.join(str(c) for c in command)}", flush=True)
    done = subprocess.run([str(c) for c in command], **kwargs)
    if done.returncode != 0:
        raise Stop(f"{Path(str(command[0])).name} failed (exit {done.returncode}); see above")


def clone_llama_cpp() -> None:
    head = CLONE / ".git"
    if head.exists():
        current = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
        if current == LLAMA_CPP_COMMIT:
            print(f"  llama.cpp is already at {LLAMA_CPP_COMMIT[:12]} in {CLONE}")
            return
    else:
        run(["git", "clone", "--filter=blob:none", "--no-checkout", LLAMA_CPP_URL, CLONE])
    run(["git", "-C", CLONE, "fetch", "--depth", "1", "origin", LLAMA_CPP_COMMIT])
    run(["git", "-C", CLONE, "checkout", "--force", LLAMA_CPP_COMMIT])


def cmake() -> str:
    found = shutil.which("cmake")
    if found:
        return found
    if VSWHERE.is_file():
        out = subprocess.run([str(VSWHERE), "-latest", "-products", "*", "-find",
                              r"**\CMake\bin\cmake.exe"], capture_output=True, text=True).stdout
        if out.strip():
            return out.strip().splitlines()[0]
    raise Stop("CMake wasn't found. Install the Visual Studio Build Tools with 'C++ CMake tools "
               "for Windows' (the same tools that build volis).")


def quantize_tool() -> Path:
    build = CLONE / "build"
    for candidate in (build / "bin" / "Release" / "llama-quantize.exe", build / "bin" / "llama-quantize"):
        if candidate.is_file():
            return candidate
    tool = cmake()
    run([tool, "-S", CLONE, "-B", build, "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=OFF",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF"])
    run([tool, "--build", build, "--config", "Release", "--target", "llama-quantize", "-j", "4"])
    for candidate in (build / "bin" / "Release" / "llama-quantize.exe", build / "bin" / "llama-quantize"):
        if candidate.is_file():
            return candidate
    raise Stop(f"the build finished but llama-quantize isn't under {build / 'bin'}")


def current_volis_gguf_size() -> int:
    try:
        from qwen_common import volis_dir

        files = list((volis_dir() / "models" / "mt").glob("*.gguf"))
        return files[0].stat().st_size if files else 0
    except Stop:
        return 0


def work(job: dict) -> dict:
    merged = run_dir(job) / "merged"
    if not (merged / "model.safetensors").is_file():
        raise Stop("there is no merged model; run Q6 first")
    out = run_dir(job) / "gguf"
    out.mkdir(parents=True, exist_ok=True)
    name = job["output"]["file_name"]
    quant = job["output"].get("quantization", "Q4_K_M")

    heading(f"llama.cpp at {LLAMA_CPP_COMMIT[:12]} (the commit volis's llama-cpp-2 0.1.156 uses)")
    clone_llama_cpp()
    (out / "llama.cpp-commit.txt").write_text(LLAMA_CPP_COMMIT + "\n", encoding="utf-8")

    heading("Converting the merged model to a full-size GGUF")
    full = out / (Path(name).stem + "-bf16.gguf")
    env = dict(os.environ, PYTHONPATH=str(CLONE / "gguf-py"), PYTHONIOENCODING="utf-8")
    run([sys.executable, CLONE / "convert_hf_to_gguf.py", merged, "--outfile", full,
         "--outtype", "bf16"], env=env)

    heading(f"Building llama-quantize and compressing to {quant}")
    tool = quantize_tool()
    final = out / name
    run([tool, full, final, quant])
    size = final.stat().st_size
    reference = current_volis_gguf_size()
    print(f"\n  {final}  ({size / 1e6:,.0f} MB)")
    if reference:
        print(f"  volis's current translator: {reference / 1e6:,.0f} MB")
        if not 0.7 * reference <= size <= 1.3 * reference:
            raise Stop("the GGUF is far from the size of volis's current translator; something "
                       "went wrong in conversion or quantization")
    full.unlink()   # the full-size file is only a step on the way
    return {"gguf": str(final), "size_mb": round(size / 1e6), "llama_cpp_commit": LLAMA_CPP_COMMIT}


if __name__ == "__main__":
    run_stage("Q7", "convert to GGUF", work)
