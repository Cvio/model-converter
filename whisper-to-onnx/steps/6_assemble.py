"""Step 6: build the model folder cnverc reads.

    <engine.folder_name>/
      engine.toml
      encoder.int8.onnx
      decoder.int8.onnx
      tokens.txt

engine.toml is checked against what cnverc's model reader accepts
(cnverc/src/models.rs): only the keys name, kind, backend, languages,
data_dir and files; every file it names present; and at least one language.
Anything else and cnverc lists the model as broken.
"""

import re
import shutil
import tomllib

from common import Stop, args, fresh_dir, heading, load_config, main, read_json, run_dir

ALLOWED_KEYS = {"name", "kind", "backend", "languages", "data_dir", "files"}
# The file roles cnverc's Whisper loader asks for (cnverc/src/asr.rs).
WHISPER_ROLES = {"encoder", "decoder", "tokens"}


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def check_engine_toml(folder) -> dict:
    parsed = tomllib.loads((folder / "engine.toml").read_text(encoding="utf-8"))
    extra = set(parsed) - ALLOWED_KEYS
    if extra:
        raise Stop(f"engine.toml has keys cnverc rejects: {', '.join(sorted(extra))}")
    for key in ("name", "kind", "backend", "files"):
        if key not in parsed:
            raise Stop(f"engine.toml has no '{key}'")
    if parsed["kind"] != "segment" or parsed["backend"] != "whisper":
        raise Stop("engine.toml must say kind = \"segment\" and backend = \"whisper\"")
    if not parsed.get("languages"):
        raise Stop("engine.toml declares no languages; cnverc would never pick this model")
    roles = set(parsed["files"])
    if roles != WHISPER_ROLES:
        raise Stop(f"[files] must declare exactly {sorted(WHISPER_ROLES)}, not {sorted(roles)}")
    for role, filename in parsed["files"].items():
        if not (folder / filename).is_file():
            raise Stop(f"engine.toml names {filename} for '{role}', but it is not in {folder}")
    return parsed


def step() -> None:
    a = args(__doc__)
    config = load_config(a.config)
    run = run_dir(config)
    onnx = run / "onnx"
    name = config["run_name"]
    engine = config["engine"]
    chosen = read_json(onnx / "chosen.json", "step 5")
    use_fp32 = chosen["method"] == "fp32"
    if use_fp32 != bool(config.get("use_fp32")):
        raise Stop("use_fp32 changed since step 5 ran; run step 5 again")

    folder_name = engine["folder_name"]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", folder_name):
        raise Stop(f"engine.folder_name {folder_name!r} must be a plain folder name")
    for code in engine["languages"]:
        if not re.fullmatch(r"[a-z]{2,3}", code):
            raise Stop(f"{code!r} is not a language code like 'es' (engine.languages)")
    if config["language"] not in engine["languages"]:
        raise Stop(
            f"language '{config['language']}' is not in engine.languages; cnverc would not "
            f"offer this model for the language it was tested in"
        )

    out_root = fresh_dir(run / "out", a.force)
    folder = out_root / folder_name
    folder.mkdir()

    heading(f"Assembling {folder}")
    print(f"  {chosen['method']} files, as chosen by step 5")
    suffix = ".onnx" if use_fp32 else ".int8.onnx"
    files = {"encoder": f"encoder{suffix}", "decoder": f"decoder{suffix}", "tokens": "tokens.txt"}
    for kind in ("encoder", "decoder"):
        source = onnx / chosen[kind]
        if not source.is_file():
            raise Stop(f"{source} is missing; run step 5 again")
        shutil.copy2(source, folder / files[kind])
        if use_fp32:
            # The .onnx refers to its .weights file by name, so that name stays.
            weights = onnx / f"{name}-{kind}.weights"
            if weights.is_file():
                shutil.copy2(weights, folder / weights.name)
    shutil.copy2(onnx / f"{name}-tokens.txt", folder / files["tokens"])

    languages = ", ".join(toml_string(code) for code in engine["languages"])
    (folder / "engine.toml").write_text(
        f"name = {toml_string(engine['display_name'])}\n"
        f'kind = "segment"\n'
        f'backend = "whisper"\n'
        f"languages = [{languages}]\n"
        f"\n"
        f"# Converted from {config['model_id']} by model-converter (run {name},\n"
        f"# {chosen['method']} weights).\n"
        f"[files]\n"
        f"encoder = {toml_string(files['encoder'])}\n"
        f"decoder = {toml_string(files['decoder'])}\n"
        f"tokens  = {toml_string(files['tokens'])}\n",
        encoding="utf-8",
        newline="\n",
    )

    check_engine_toml(folder)
    for path in sorted(folder.iterdir()):
        print(f"  {path.name:30} {path.stat().st_size / 1e6:10,.1f} MB")
    heading("engine.toml")
    print((folder / "engine.toml").read_text(encoding="utf-8"))
    print(f"OK. {folder} passes cnverc's engine.toml rules. Step 7 installs and checks it.")


if __name__ == "__main__":
    main(step)
