# model-converter

Turns models from Hugging Face into files [cnverc](../cnverc) can run.

cnverc runs its speech models through sherpa-onnx, which only loads ONNX files in its own layout.
Fine-tuned models on Hugging Face come as `model.safetensors` or `pytorch_model.bin`, in Hugging
Face's layout. This project converts one into the other, checks the result at every step, and
produces a model folder you copy into cnverc's `models/asr/`.

**What it converts today:** Whisper speech recognizers, whether full fine-tunes from Hugging Face
or merged models from the LoRA training app. Translation models (Qwen → GGUF) are next and will
sit beside it in their own folder.

This is a separate project on purpose. It's Python, it downloads from the internet, and cnverc
itself must never do either. The two meet only at the model folder.

## Quick start

This is a command-line tool, not a window you open. You run it from PowerShell, one step at a
time, and each step prints what it did. At the end, the converted model appears in cnverc's
recognizer list.

**Once per computer**

1. Install Git and uv, then close PowerShell and open a new one:

   ```powershell
   winget install Git.Git astral-sh.uv
   ```

2. Get the code and set it up. This takes a few minutes the first time:

   ```powershell
   cd D:\AI_Data\projects
   git clone https://github.com/Cvio/model-converter.git
   cd model-converter
   .\setup.ps1
   ```

   It should end with `OK. This machine can run every step.` If it doesn't, see
   [On another PC](#on-another-pc), step 4.

3. Tell it where cnverc is. Copy `machine.example.yaml` to `machine.yaml`, open it in Notepad,
   and set `cnverc_path` to the folder that has `cnverc.exe` in it.

**Each time you convert a model**

1. Pick a config from `whisper-to-onnx\configs\`, or copy one and change the model name in it.
2. Make sure the recording it names under `test:` exists in `test_audio\`.
3. Open PowerShell in the `model-converter` folder and run:

   ```powershell
   .\convert.ps1 whisper-to-onnx\configs\es-small-hitz.yaml
   ```

   Use the config you picked. It runs all seven steps in order and stops at the first one that
   fails. Steps 2 and 4 take the longest: step 2 downloads the model and step 4 converts it. If
   PowerShell refuses to run scripts, start it with
   `powershell -ExecutionPolicy Bypass -File .\convert.ps1 ...` instead.

4. Open cnverc, tick **Compare recognizers**, and speak. The new model is listed beside the
   ones you already have.

**If a step stops**

It prints `STOP:` and the reason, and then the command that picks up where it left off, for
example:

```powershell
.\convert.ps1 whisper-to-onnx\configs\es-small-hitz.yaml -From 4
```

Fix what it says, then run that command. The steps that already passed aren't repeated. To
redo a whole conversion from scratch, add `-Force`.

**Running the steps one at a time**

`convert.ps1` just runs these commands in order. You can run them yourself instead, for
example to look at a step's output before going on. Add `--force` to rerun a step that has
already made its output folder.

```powershell
$c = "whisper-to-onnx\configs\es-small-hitz.yaml"
uv run python whisper-to-onnx\steps\1_check_setup.py --config $c
uv run python whisper-to-onnx\steps\2_download.py --config $c
uv run python whisper-to-onnx\steps\3_to_openai_format.py --config $c
uv run python whisper-to-onnx\steps\4_export_onnx.py --config $c
uv run python whisper-to-onnx\steps\5_check_int8.py --config $c
uv run python whisper-to-onnx\steps\6_assemble.py --config $c
uv run python whisper-to-onnx\steps\7_verify.py --config $c
```

Everything after this section is detail: what the steps do, and why.

## Setting it up

You need 64-bit Windows, [Git](https://git-scm.com) and [uv](https://docs.astral.sh/uv/) 0.11
(`winget install astral-sh.uv`). Then, in the cloned folder:

```powershell
.\setup.ps1
```

That installs Python 3.12.10 and every package at the exact versions in `uv.lock`
(`uv sync --locked`, so nothing is re-resolved), then runs `0_doctor.py` to check the result.
uv's download cache, the Python interpreter and the environment all go inside this folder
(`.uv\` and `.venv\`), so every DLL the project uses lives under one path. Run `setup.ps1`
again, rather than `uv sync`, whenever `uv.lock` changes; add `-Reinstall` to replace every
package.

Then copy `machine.example.yaml` to `machine.yaml` and set `cnverc_path` to the folder
`cnverc.exe` is in. `machine.yaml` holds what differs between PCs and is not committed.

The first run of step 1 also clones sherpa-onnx's source at v1.13.8 into `vendor/`, for its
export script.

### On another PC

These steps take a fresh Windows 11 machine to the point where the steps run. Nothing is copied
from the first machine except the repo, and what isn't committed is listed here.

1. Install Git and uv 0.11 (`winget install Git.Git astral-sh.uv`), then open a new terminal so
   both are on PATH.
2. Clone the repo into the folder where it will live. A later move means reinstalling, because
   the environment records its own path. Keep it out of synced folders such as OneDrive.
3. In that folder, run `.\setup.ps1`. If PowerShell refuses to run scripts, use
   `powershell -ExecutionPolicy Bypass -File .\setup.ps1`.
4. Read the end of its output:
   - `OK. This machine can run every step.` means go on to 5.
   - `MISSING` or `ALTERED` lines mean security software removed or changed those files.
     Send the folder path and sentence the doctor prints to whoever manages it, and wait for the
     exclusion. Then run `.\setup.ps1 -Reinstall` and check for `OK.` again.
   - `cnverc's engine could not run` naming a DLL outside this folder means something else
     loaded its own `onnxruntime.dll` into Python. If that path belongs to security software, it
     is the same conversation: ask for the exclusion, then reinstall.
   - `uv is ...` or `git is not installed` means fix what it names and rerun.
5. Copy `machine.example.yaml` to `machine.yaml` and set `cnverc_path` to where `cnverc.exe`
   is on this machine.
6. Put the test recordings the configs name into `test_audio/` (they are not committed). Copy
   them from the other machine, or make new ones as step 2 of the next section describes.
7. Run step 1 with a config. It runs the doctor again, clones sherpa-onnx into `vendor/`, and
   checks the patch and the recording.

If something differs between the two machines and you can't see why, compare
`runs\_doctor.json` from each: it lists the versions, the DLL each engine loaded, other
`onnxruntime.dll` copies and any security software.

## Converting a Whisper model

1. **Write a config.** Copy one from `whisper-to-onnx/configs/` and change it. Everything about
   a particular model lives in its config. The scripts never change between models.

   ```yaml
   model_id: adriszmar/whisper-large-v3-turbo-es   # a Hugging Face ID, or a folder on disk
   base_model: openai/whisper-large-v3-turbo       # the model it was fine-tuned from
   run_name: es-turbo-adriszmar                     # its folder under runs/
   language: es                                     # the language to test it in

   engine:
     folder_name: whisper-large-v3-turbo-es-adriszmar   # its folder in cnverc's models/asr/
     display_name: Whisper large-v3-turbo Spanish (adriszmar, int8)   # shown in cnverc
     languages: [es]

   use_fp32: false            # true ships full-size files; step 5 says when you need it

   test:
     wav: test_audio/es-16k.wav   # a short clip in that language, 16 kHz mono
   ```

   Where cnverc is comes from `machine.yaml`, not the config.

2. **Put a test recording in `test_audio/`.** Recordings are not committed, so a fresh clone
   has none. Use a few seconds of speech in the model's language, 16 kHz mono. `cnverc --listen --wav` saves exactly that to `logs/segments/`. Any other file
   converts with `ffmpeg -i in.wav -ar 16000 -ac 1 out.wav`.

3. **Run the seven steps in order.** Each one prints what it did and stops with a clear
   message if a check fails:

   ```bash
   uv run python whisper-to-onnx/steps/1_check_setup.py --config whisper-to-onnx/configs/your-model.yaml
   ```

   Step 1 runs `0_doctor.py` first. Then run `2_download.py`, `3_to_openai_format.py`,
   `4_export_onnx.py`, `5_check_int8.py`, `6_assemble.py` and `7_verify.py`, each with the
   same `--config`.

4. **Listen to it.** Step 7 installs the folder into cnverc and tells you how to compare it
   with the recognizer you already have: tick **Compare recognizers** in cnverc and speak.

Everything a run produces goes in `runs/<run_name>/`. A step never overwrites an earlier
run's output unless you pass `--force`.

## What each step does, and what it checks

| Step | Does | Stops if |
|---|---|---|
| 1 Check setup | Prints versions, checks sherpa-onnx is at v1.13.8, and reads the export script on disk | The patch no longer fits the export script; the test clip isn't 16 kHz mono |
| 2 Download | Fetches only the weights and configs; the tokenizer comes from the base model; writes `arch.json` | It's a LoRA adapter (merge it first); the architecture doesn't fit openai-whisper; the vocabulary differs from the base model's |
| 3 To OpenAI format | Renames every weight into OpenAI Whisper's layout | Any weight is left over or missing; **the converted model's outputs differ from the original's on the same input** |
| 4 Export ONNX | Runs sherpa-onnx's export script, patched, on the checkpoint | **The ONNX files' outputs differ from the checkpoint's on the same input**; the encoder is the wrong size |
| 5 Check int8 | Chooses int8 files that still transcribe sensibly in cnverc's engine: the export script's own, or a per-channel requantization if those fail | Neither int8 version is sensible (then set `use_fp32: true`) |
| 6 Assemble | Builds the folder and `engine.toml` | `engine.toml` breaks any of cnverc's rules |
| 7 Verify | Installs into cnverc, runs `--report`, transcribes the installed folder with cnverc's engine | cnverc doesn't list it as `ok`, or it transcribes nothing |

"cnverc's engine" is the sherpa-onnx Python package at 1.13.8, the same C++ recognizer cnverc
links, given the settings cnverc uses. It decides every transcript-based check, because it is what
will actually run the model.

The checks in bold are exact. Transcripts can differ through decoding settings alone, so steps 3
and 4 feed both models the same input and compare their raw outputs. A correct conversion agrees
to within rounding (step 3 measured 0, step 4 about 0.00004), while one misplaced weight is off by
whole units: swapping two weights in one layer on purpose gave a difference of 2.7.

## Things that went wrong, so you don't have to find them again

- **sherpa-onnx's export script only takes model names**, and picks settings from the name.
  The patch in `whisper-to-onnx/patches/` makes it take a checkpoint file, and read the mel
  count and file sizes from the model itself. Without it, a turbo or large-v3 fine-tune fails to
  export (it gets 80 mel bins instead of 128). The patch is applied to a copy of the script,
  never to the sherpa-onnx checkout, and step 1 checks it still fits.
- **Judge int8 with the engine that will run it.** sherpa-onnx's `scripts/whisper/test.py`
  decodes in its own Python loop. On whisper-small, it made the export script's int8 files look
  badly broken (61% agreement with fp32), while cnverc's engine, given the same files, differed
  from fp32 by one word. Steps 4, 5 and 7 therefore transcribe with cnverc's engine and never
  with `test.py`. If int8 does fail there, step 5 tries a per-channel requantization (same
  size) before falling back to fp32.
- **One 5-second clip is a thin basis for int8.** On an unclear stretch, int8 can change a word
  from run to run. Step 7's side-by-side comparison in cnverc, on real speech, is the real
  judge.
- **PyTorch 2.9 and newer** export with a new engine that fails on Whisper. The patch passes
  `dynamo=False`.
- **transformers 5 no longer ships its conversion scripts.** The weight-name mapping comes from
  `convert_openai_to_hf.py` in transformers v4.46.0, copied into step 3 with its source named.
- **sherpa-onnx's own package leaves out its runtime unless asked.** Its wheels require
  `sherpa-onnx-core`, which puts `onnxruntime.dll` 1.28.2 (the version cnverc links) beside
  sherpa-onnx's extension. Its sdist declares no dependencies, and uv locks from the sdist, so
  `pyproject.toml` names `sherpa-onnx-core` itself. Without it, sherpa-onnx finds whatever
  `onnxruntime.dll` Windows offers, and Windows 11 has an old one (1.17) in System32 that crashes it.
- **cnverc's engine runs in a process of its own** (`engine_worker.py`), which checks which
  `onnxruntime.dll` it actually loaded and refuses to transcribe on anything but 1.28.2 from
  the environment. The `onnxruntime` package (1.30) that steps 4 and 5 use for the exact
  logits check and for quantizing links its runtime into its own extension, so the two never meet.
- **PyTorch exports a model over 2 GB as hundreds of loose files**, one per tensor. The export
  script gathers them into one `.weights` file but leaves the loose ones behind; step 4 deletes
  them before checking the export.
- **Python 3.14** doesn't have wheels for everything here yet; `pyproject.toml` pins 3.12.
- **Many fine-tunes ship `pytorch_model.bin`**, not safetensors. Both work.
- **A fine-tune with added tokens can't be converted this way.** sherpa-onnx writes `tokens.txt`
  from Whisper's standard vocabulary, so step 2 refuses a model whose vocabulary size differs from
  its base model's.

## Security software (McAfee, Trellix and the like)

On-access scanners sometimes quarantine or lock DLLs while uv writes them: torch's, onnxruntime's,
sherpa-onnx's. The install then looks finished, and a step fails much later when it first loads
the missing file (usually step 4, the first to load ONNX Runtime and sherpa-onnx).
`0_doctor.py` checks for exactly this, before anything runs:

- every `.dll`/`.pyd` the packages installed is present and matches the hash in the package's
  own `RECORD`; it names each one that is missing or changed;
- cnverc's engine loads, on onnxruntime 1.28.2 from the environment, and names the DLL it got if not;
- torch and onnxruntime import;
- which other `onnxruntime.dll` copies are on the machine, and whether McAfee/Trellix is installed.

If it reports a missing or altered file, ask whoever manages the security software for an
on-access scan exclusion for this folder (the doctor prints the path and a sentence to send),
then run `.\setup.ps1 -Reinstall`. Everything is under that one folder by design. Don't turn the
scanner off to get around it. If no exclusion is possible, steps 1 to 6 could run under WSL2
instead (not set up here); step 7 needs Windows, because it runs `cnverc.exe`.

`runs\_doctor.json` records what the doctor found. Comparing it between two machines is the
quickest way to see what differs.

## Tested with

| Config | Model | Result |
|---|---|---|
| `es-small-hitz.yaml` | `HiTZ/whisper-small-es` (`pytorch_model.bin`, 80 mels, 12 decoder layers) | All seven steps pass. Logits: 0 (step 3), 0.00004 (step 4). int8: 375 MB, 94% agreement with fp32 in cnverc's engine |
| `es-turbo-adriszmar.yaml` | `adriszmar/whisper-large-v3-turbo-es` (safetensors, 128 mels, 4 decoder layers) | All seven steps pass. Logits: 0 (step 3), 0.00008 (step 4). int8: 1,036 MB, 100% agreement with fp32 |

These numbers were measured with cnverc's engine running on onnxruntime 1.30. On 1.28.2 (what
cnverc links, and what the engine uses now) the shipped turbo int8 files give the same words;
the small int8 files differ by one word on the unclear stretch the notes above describe.

Both ran on the same scripts; only the configs differ. Both are installed in cnverc and
listed as `ok`. Whether either beats the stock Whisper turbo on your own speech is step 7's
listening test, which hasn't been done yet.

## Layout

```
model-converter/
  pyproject.toml        the Python environment (uv); uv.lock pins every version
  setup.ps1             installs everything into this folder and runs the doctor
  convert.ps1           runs the seven steps for one config, in order
  machine.example.yaml  copy to machine.yaml: this PC's paths (not committed)
  whisper-to-onnx/
    configs/            one .yaml per model
    steps/              0_doctor.py, the seven steps, common.py, engine_worker.py
    patches/            the change to sherpa-onnx's export script, kept visible
  test_audio/           test recordings (not committed)
  vendor/               sherpa-onnx at v1.13.8, cloned by step 1 (not committed)
  .uv/  .venv/          uv's cache and Python, and the environment (not committed)
  runs/<run_name>/      everything a run produces (not committed)
  whisper-to-onnx-converter.md   the original instructions this was built from
```
``` html
  https://www.kaggle.com/datasets/husseincamus/iraqi-conversational-voice-recordings-dataset
```