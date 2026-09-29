"""volis's engine, alone in its own process.

Transcribes one wav with the sherpa-onnx Python package, given the settings
volis uses, and prints one line of JSON. common.transcribe_like_cnverc runs
this as a subprocess; nothing else should import it.

Why a separate process: Windows reuses a DLL that is already loaded under the
same name. If the step had already imported the onnxruntime Python package
(1.30, for the logits check and quantization), sherpa-onnx would silently run
on that instead of the onnxruntime.dll sherpa-onnx-core ships beside it
(1.28.2, the one volis links). This process never imports onnxruntime, and
checks which onnxruntime.dll it actually got before trusting a transcript.

    python engine_worker.py --selftest --expect-ort 1.28.2
    python engine_worker.py --encoder E --decoder D --tokens T --wav W --language es --expect-ort 1.28.2
    python engine_worker.py ... --wav-list clips.txt ...   (one wav path per line; the
        model loads once; "texts" holds one transcript per line, "seconds" the time
        each took, reading the file included)
    python engine_worker.py --encoder E --decoder D --joiner J --tokens T ...   (a NeMo
        transducer such as Parakeet, instead of Whisper; --language is ignored)
"""

import argparse
import json
import os
import sys
from pathlib import Path


class Refuse(Exception):
    pass


def loaded_modules() -> list:
    """Full paths of every DLL loaded in this process (Windows only)."""
    import ctypes
    from ctypes import wintypes

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    process = kernel32.GetCurrentProcess()
    count = 1024
    while True:
        modules = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(
            process, modules, ctypes.sizeof(modules), ctypes.byref(needed)
        ):
            raise OSError(ctypes.get_last_error(), "EnumProcessModules failed")
        if needed.value <= ctypes.sizeof(modules):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 16
    paths = []
    buffer = ctypes.create_unicode_buffer(32768)
    for i in range(needed.value // ctypes.sizeof(wintypes.HMODULE)):
        if psapi.GetModuleFileNameExW(process, modules[i], buffer, len(buffer)):
            paths.append(buffer.value)
    return paths


def file_version(path: str) -> str:
    """The version in a DLL's version resource, e.g. '1.28.2'."""
    import ctypes
    from ctypes import wintypes

    version = ctypes.WinDLL("version", use_last_error=True)
    size = version.GetFileVersionInfoSizeW(path, None)
    if not size:
        return "unknown"
    data = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(path, 0, size, data):
        return "unknown"
    pointer = ctypes.c_void_p()
    length = wintypes.UINT()
    # The translation table gives the language/codepage of the string table.
    if version.VerQueryValueW(
        data, "\\VarFileInfo\\Translation", ctypes.byref(pointer), ctypes.byref(length)
    ) and length.value >= 4:
        lang, codepage = ctypes.cast(pointer, ctypes.POINTER(wintypes.WORD * 2)).contents
        key = f"\\StringFileInfo\\{lang:04x}{codepage:04x}\\ProductVersion"
        if version.VerQueryValueW(data, key, ctypes.byref(pointer), ctypes.byref(length)):
            return ctypes.wstring_at(pointer, length.value).rstrip("\x00").strip()
    return "unknown"


def check_runtime(expect: str) -> dict:
    """Which onnxruntime.dll this process loaded, refusing anything but the
    one sherpa-onnx-core installed beside sherpa-onnx's extension."""
    import sherpa_onnx

    info = {"sherpa_onnx": getattr(sherpa_onnx, "__version__", "unknown")}
    if os.name != "nt":
        info["onnxruntime"] = {"checked": False, "why": "the DLL check is Windows-only"}
        return info

    expected_dir = (Path(sherpa_onnx.__file__).parent / "lib").resolve()
    loaded = [p for p in loaded_modules() if Path(p).name.lower() == "onnxruntime.dll"]
    if not loaded:
        raise Refuse(
            "sherpa-onnx imported but no onnxruntime.dll is loaded. The package layout is not "
            "what this was written for; run 0_doctor.py."
        )
    if len(loaded) > 1:
        raise Refuse(f"more than one onnxruntime.dll is loaded: {loaded}")
    path = loaded[0]
    found = file_version(path)
    info["onnxruntime"] = {"path": path, "version": found}
    if Path(path).resolve().parent != expected_dir:
        if not (expected_dir / "onnxruntime.dll").is_file():
            raise Refuse(
                f"{expected_dir} has no onnxruntime.dll, so Windows fell back to {path} "
                f"(onnxruntime {found}). It was removed after install, typically by security "
                f"software; run 0_doctor.py, then setup.ps1 -Reinstall."
            )
        raise Refuse(
            f"sherpa-onnx is running on {path} (onnxruntime {found}), not on the copy "
            f"sherpa-onnx-core installed in {expected_dir}. Something loaded another "
            f"onnxruntime.dll into this process first. If that path belongs to security "
            f"software, it is injecting it."
        )
    if found != expect:
        raise Refuse(
            f"{path} is onnxruntime {found}, but volis links {expect}. Run "
            f"'uv sync --locked --reinstall' (through setup.ps1)."
        )
    return info


def recognizer_for(a):
    import sherpa_onnx

    if a.joiner:
        # A NeMo transducer (Parakeet), configured as volis's NemoTransducerAsr
        # is: model_type nemo_transducer, 6 threads, CPU, greedy search.
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=a.encoder,
            decoder=a.decoder,
            joiner=a.joiner,
            tokens=a.tokens,
            model_type="nemo_transducer",
            num_threads=6,
            provider="cpu",
        )
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=a.encoder,
        decoder=a.decoder,
        tokens=a.tokens,
        language=a.language,
        task="transcribe",
        tail_paddings=0,
        num_threads=6,
        provider="cpu",
    )


def transcribe_with(recognizer, wav) -> str:
    import soundfile as sf

    audio, sample_rate = sf.read(wav, dtype="float32")
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def transcribe(a) -> str:
    return transcribe_with(recognizer_for(a), a.wav)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-ort", required=True)
    parser.add_argument("--selftest", action="store_true")
    for name in ("--encoder", "--decoder", "--joiner", "--tokens", "--wav", "--wav-list", "--language"):
        parser.add_argument(name)
    a = parser.parse_args()
    try:
        # sherpa-onnx must be the first thing to load an onnxruntime.dll here.
        if "onnxruntime" in sys.modules:
            raise Refuse("onnxruntime was imported before sherpa-onnx in the engine process")
        result = check_runtime(a.expect_ort)
        if a.wav_list:
            recognizer = recognizer_for(a)
            wavs = [w for w in Path(a.wav_list).read_text(encoding="utf-8").splitlines() if w.strip()]
            import time

            result["texts"], result["seconds"] = [], []
            for w in wavs:
                started = time.perf_counter()
                result["texts"].append(transcribe_with(recognizer, w))
                result["seconds"].append(round(time.perf_counter() - started, 4))
            result["onnxruntime_after"] = check_runtime(a.expect_ort)["onnxruntime"]
        elif not a.selftest:
            result["text"] = transcribe(a)
            result["onnxruntime_after"] = check_runtime(a.expect_ort)["onnxruntime"]
        print(json.dumps({"ok": True, **result}))
    except Refuse as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(2)
    except Exception as e:  # a crash in the native code, a missing DLL, a bad file
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(3)


if __name__ == "__main__":
    main()
