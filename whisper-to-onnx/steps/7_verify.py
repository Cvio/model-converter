"""Step 7: install the folder into cnverc and check cnverc accepts it.

Copies out/<folder_name> into cnverc's models/asr/ and runs cnverc --report.
The new model must be listed as ok, with [+] beside every file. An existing
folder of the same name is never replaced unless --force is given.

Whether the model is any good is a separate question, answered by listening:
the comparison printed at the end runs it beside the model cnverc already has.
"""

import platform
import shutil
import subprocess
from pathlib import Path

from common import (
    SHERPA_TAG,
    Stop,
    args,
    heading,
    load_config,
    main,
    run_dir,
    test_wav,
    transcribe_like_cnverc,
)


def cnverc_exe(root: Path) -> Path:
    exe = root / ("cnverc.exe" if platform.system() == "Windows" else "cnverc")
    if not exe.is_file():
        raise Stop(f"cnverc is not at {exe}. Set cnverc.path in the config to its folder.")
    return exe


def report_block(report: str, folder_name: str) -> list:
    """The lines of `cnverc --report` about one model: its row, then every
    indented line beneath it."""
    lines = report.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("  ") and line.split()[:1] == [folder_name]:
            block = [line]
            # Rows are indented by two spaces; the lines about a row, far more.
            for following in lines[i + 1 :]:
                if following.startswith(" " * 10) and following.strip():
                    block.append(following)
                else:
                    break
            return block
    return []


def step() -> None:
    a = args(__doc__)
    config = load_config(a.config)
    folder_name = config["engine"]["folder_name"]
    built = run_dir(config) / "out" / folder_name
    if not (built / "engine.toml").is_file():
        raise Stop(f"{built} is missing. Run step 6 first.")
    if "cnverc" not in config or "path" not in config["cnverc"]:
        raise Stop("set cnverc.path in the config to the folder cnverc.exe is in")
    root = Path(config["cnverc"]["path"]).expanduser().resolve()
    exe = cnverc_exe(root)
    asr = root / "models" / "asr"
    if not asr.is_dir():
        raise Stop(f"{asr} does not exist; is {root} a cnverc folder?")

    heading(f"Installing into {asr}")
    target = asr / folder_name
    if target.exists():
        if not a.force:
            raise Stop(
                f"{target} already exists. Pass --force to replace it, or choose a different "
                f"engine.folder_name."
            )
        shutil.rmtree(target)
    shutil.copytree(built, target)
    for path in sorted(target.iterdir()):
        print(f"  {path.name}")

    heading("cnverc --report")
    result = subprocess.run(
        [str(exe), "--report"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    block = report_block(result.stdout, folder_name)
    if not block:
        print(result.stdout[-3000:])
        raise Stop(f"cnverc --report does not list {folder_name}")
    print("\n".join(block))

    status = block[0].split()[-1]
    files = [line.strip() for line in block if line.strip().startswith("[")]
    if status != "ok":
        raise Stop(f"cnverc lists {folder_name} as '{' '.join(block[0].split()[3:])}', not ok")
    if not files or any(not f.startswith("[+]") for f in files):
        raise Stop("cnverc could not find every file the engine.toml names")
    print(f"\n  cnverc lists {folder_name} as ok, with every file present.")

    heading(f"Transcribing with sherpa-onnx {SHERPA_TAG}, as cnverc does")
    import tomllib

    files = tomllib.loads((target / "engine.toml").read_text(encoding="utf-8"))["files"]
    text = transcribe_like_cnverc(
        target / files["encoder"],
        target / files["decoder"],
        target / files["tokens"],
        test_wav(config),
        config["language"],
    )
    print(f"  {text}")
    if not text:
        raise Stop(
            "sherpa-onnx loaded the model but transcribed nothing. cnverc would show "
            "'nothing recognised' for every utterance."
        )
    print(f"\nOK. cnverc can load {folder_name} and it transcribes the test recording.")

    heading("Now listen to it")
    print(
        f"  1. Open cnverc ({exe}), tick 'Compare recognizers', press Start and speak "
        f"{config['language']}.\n"
        f"     Or, in a terminal: \"{exe}\" --listen --compare\n"
        f"  2. Every installed recognizer transcribes the same audio, side by side, with its "
        f"timing.\n"
        f"  3. To use it for real, choose '{config['engine']['display_name']}' as the "
        f"Recognizer.\n"
        f"  A converted model that turns out worse than the one you have means the model is "
        f"weak, not that the conversion failed."
    )


if __name__ == "__main__":
    main(step)
