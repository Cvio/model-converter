"""The teacher: a large local model, run through llama.cpp's llama-server,
used once to prepare training data and never shipped.

    with Teacher(gguf) as t:
        answers = t.complete_all(messages_list)

It starts llama-server on localhost, checks the model really loaded onto the
GPU (a CPU fallback would take days), sends many requests at once, and stops
the server at the end, whatever happens. Decoding is greedy with thinking off,
so the output is repeatable.

Which llama-server and which model come from machine.yaml:

    llama_server: teacher/llama.cpp/llama-server.exe
    teacher_gguf:
      - inputs/teacher/Qwen3-8B-Q4_K_M.gguf

The server is a training tool on this machine only; volis never talks to it.
"""

import json
import re
import socket
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training" / "steps"))

from common import Stop, load_machine  # noqa: E402

PARALLEL = 4
CONTEXT_PER_SLOT = 2048


def machine_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def teacher_ggufs() -> list:
    machine = load_machine()
    listed = machine.get("teacher_gguf")
    if not listed:
        raise Stop("machine.yaml names no teacher model. Add, for example:\n"
                   "    teacher_gguf:\n      - inputs/teacher/Qwen3-8B-Q4_K_M.gguf")
    if isinstance(listed, str):
        raise Stop("teacher_gguf in machine.yaml must be a list, even with one entry:\n"
                   f"    teacher_gguf:\n      - {listed}")
    paths = [machine_path(p) for p in listed]
    for p in paths:
        if not p.is_file():
            raise Stop(f"the teacher model {p} (teacher_gguf in machine.yaml) does not exist")
    return paths


def server_exe() -> Path:
    machine = load_machine()
    exe = machine_path(machine.get("llama_server", "teacher/llama.cpp/llama-server.exe"))
    if not exe.is_file():
        raise Stop(f"llama-server is not at {exe}. Run:\n"
                   f"    uv run --project training python teacher/get_server.py\n"
                   f"or set llama_server in machine.yaml.")
    return exe


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Teacher:
    def __init__(self, gguf: Path, log_dir: Path, parallel: int = PARALLEL):
        self.gguf = gguf
        self.parallel = parallel
        self.port = free_port()
        self.log_path = log_dir / f"llama-server-{gguf.stem}.log"
        self.process = None

    def __enter__(self):
        import requests

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        command = [
            str(server_exe()), "--model", str(self.gguf),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--parallel", str(self.parallel),
            "--ctx-size", str(CONTEXT_PER_SLOT * self.parallel),
            "--n-gpu-layers", "999",           # everything on the GPU
            "--jinja",                         # the model's own chat template
            "--reasoning-budget", "0",         # thinking off, as in volis
        ]
        print(f"  starting llama-server with {self.gguf.name} (log: {self.log_path})", flush=True)
        try:
            self.baseline_mb = self.total_gpu_used_mb()
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            self.baseline_mb = 0
        self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise Stop(f"llama-server exited while loading (code {self.process.returncode}). "
                           f"The end of {self.log_path} says why.")
            try:
                if requests.get(self.url("/health"), timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            self.stop()
            raise Stop(f"llama-server did not finish loading within 10 minutes; see {self.log_path}")
        self.check_gpu()
        return self

    def check_gpu(self) -> None:
        """Stop unless the server process holds the model in GPU memory.

        Asked of the NVIDIA driver (nvidia-smi), not read from the log: the
        log's wording changes between llama.cpp versions, and GPU memory is
        what a CPU fallback would lack. Most of the model file must be there;
        a model too big for the card shows as partly loaded, with a note."""
        try:
            listing = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory,name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30,
            ).stdout
            gpu_name = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError) as e:
            self.stop()
            raise Stop(f"nvidia-smi could not be run to check the GPU ({e}); is the NVIDIA driver installed?") from e
        used_mb = 0
        for line in listing.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] == str(self.process.pid) and parts[1].isdigit():
                used_mb = int(parts[1])
        model_mb = self.gguf.stat().st_size / 1e6
        # Windows' WDDM driver reports per-process memory only for some setups;
        # when it reports nothing for anyone, fall back to total memory in use.
        if used_mb == 0 and "[N/A]" in listing or (used_mb == 0 and not listing.strip()):
            used_mb = self.total_gpu_used_mb() - self.baseline_mb
        share = used_mb / model_mb
        print(f"  on the GPU: {gpu_name}, {used_mb:,} MB in use for a {model_mb:,.0f} MB model", flush=True)
        if share < 0.5:
            self.stop()
            raise Stop("llama-server did not put the model on the GPU (a CPU teacher would take "
                       f"days). Check the CUDA build and driver; see {self.log_path}")
        if share < 0.9:
            print("  NOTE: part of the model runs on the CPU (it doesn't all fit); this is slow.")

    @staticmethod
    def total_gpu_used_mb() -> int:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def complete(self, messages: list, max_tokens: int = 512) -> str:
        import requests

        body = {
            "messages": messages,
            "temperature": 0,        # greedy: the same answer every time
            "top_k": 1,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = requests.post(self.url("/v1/chat/completions"), json=body, timeout=600)
        if response.status_code != 200:
            raise Stop(f"the teacher refused a request ({response.status_code}): {response.text[:300]}")
        text = response.json()["choices"][0]["message"]["content"] or ""
        return strip_think(text).strip()

    def complete_all(self, many: list, max_tokens: int = 512) -> list:
        """Answers for a list of message lists, in order, several at a time."""
        with ThreadPoolExecutor(self.parallel) as pool:
            return list(pool.map(lambda m: self.complete(m, max_tokens), many))

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __exit__(self, *exc):
        self.stop()
        return False


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


# --- Restoring punctuation ------------------------------------------------------

PUNCTUATE_SYSTEM = (
    "You add capital letters and punctuation to a word-for-word transcript of "
    "Spanish speech. Add capitals, commas, full stops, and the opening and closing "
    "question and exclamation marks (¿ ? ¡ !) where they belong. You may add the "
    "written accent a question word takes (qué, cómo, dónde, cuál).\n\n"
    "Every word must stay exactly as it is, in the same order, because the "
    "transcript records what was really said. In particular:\n"
    "- Keep repeated words: \"que que ver\" stays \"que que ver\".\n"
    "- Keep false starts and cut-off words: \"es contra contratada\" stays as it is.\n"
    "- Keep hesitation sounds such as \"e\" and \"este\": \"e la mujer\" stays \"e la mujer\".\n"
    "- Keep misspellings and grammar mistakes exactly as written (\"echo\", \"hada\", "
    "\"dejé\"), and colloquial spellings (pus, namás): never correct them.\n"
    "- Never join or split words: \"porciento\" stays one word, \"por que\" stays two.\n"
    "- Add no word and remove no word.\n\n"
    "Reply with the punctuated transcript only."
)


def words(text: str) -> list:
    """The words of a line for comparing: case, punctuation and written accents
    removed. Accents go because adding one (que -> qué, como -> cómo) belongs
    with adding ¿ and ?; it doesn't change the word. ñ is kept: año and ano are
    different words."""
    text = unicodedata.normalize("NFC", text).lower().replace("ñ", "\0")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = unicodedata.normalize("NFC", text).replace("\0", "ñ")
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def punctuate(teacher: Teacher, lines: list, system: str = PUNCTUATE_SYSTEM) -> list:
    """(original, teacher's answer, kept) per line. A line whose words changed
    is not kept: the teacher rewrote it, and it must be thrown away and counted.
    The answer is returned either way, so a thrown-away line can be inspected."""
    answers = teacher.complete_all(
        [[{"role": "system", "content": system}, {"role": "user", "content": line}] for line in lines]
    )
    return [(line, answer, words(answer) == words(line)) for line, answer in zip(lines, answers)]


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
