# model-converter

Turns models from Hugging Face into files [volis](https://github.com/Cvio/cnverc) (formerly cnverc) can run.

volis runs its speech models through sherpa-onnx, which only loads ONNX files in its own layout.
Fine-tuned models on Hugging Face come as `model.safetensors` or `pytorch_model.bin`, in Hugging
Face's layout. This project converts one into the other, checks the result at every step, and
produces a model folder you copy into volis's `models/asr/`.

**What it converts today:** Whisper speech recognizers, whether full fine-tunes from Hugging Face
or merged models from the LoRA training app. Translation models (Qwen → GGUF) are next and will
sit beside it in their own folder.

This is a separate project on purpose. It's Python, it downloads from the internet, and volis
itself must never do either. The two meet only at the model folder.

## Quick start

This is a command-line tool, not a window you open. You run it from PowerShell, one step at a
time, and each step prints what it did. At the end, the converted model appears in volis's
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

3. Tell it where volis is. Copy `machine.example.yaml` to `machine.yaml`, open it in Notepad,
   and set `volis_path` to the folder that has `volis.exe` in it.

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

4. Open volis, tick **Compare recognizers**, and speak. The new model is listed beside the
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

## Training your own model (quick start)

This is how to teach a speech recognizer an accent, for example Mexican Spanish, and put the
result into volis. You don't need to understand machine learning to do it: two commands do the
work, and every step checks itself and stops with a plain message if something is wrong.

### A few words first

- **Model:** a file that has learned to do one job, here writing down what someone says.
  Whisper is the one volis uses to hear speech.
- **Training** (also called fine-tuning): showing a model thousands of recordings with their
  correct transcripts, so it gets better at that kind of speech. It changes the model a little;
  it doesn't start from nothing.
- **LoRA:** a small add-on file that holds what training taught the model. The original model
  is left untouched until the end, when the two are combined (**merged**).
- **GPU:** the graphics card. Training runs on it, because it is many times faster than the
  main processor. It must be an NVIDIA card.
- **Dataset:** a collection of recordings with their transcripts. The ones used here are
  downloaded from Hugging Face, a public website for models and datasets.
- **Teacher:** a much bigger language model that tidies the transcripts before training (adds
  capital letters and punctuation). It is only a helper and is never put into volis.
- **Job file:** a short text file in `jobs\` that says what to train: which model, which
  recordings, which language. You pick one; you don't have to write one.

### Which computer to use

Training needs an NVIDIA graphics card with plenty of memory. The desktop with the **RTX 5090**
is the one for real runs. A laptop with an 8 GB card, like the RTX 4070, can only do the small
**rehearsal** job: a quick run of every step on about an hour of recordings, to prove everything
works before spending hours on the real thing.

### Once per computer

1. **Install the programs it needs.** Open PowerShell (Start menu, type `PowerShell`) and run:

   ```powershell
   winget install Git.Git astral-sh.uv
   ```

   Git downloads this project; uv installs Python and everything this project uses. Also make sure
   the **NVIDIA driver** is up to date (the NVIDIA app, or nvidia.com): training can't reach the
   graphics card without it. Close PowerShell and open a new one afterwards, so it finds what you
   installed.

2. **Get this project and set it up:**

   ```powershell
   cd D:\AI_Data\projects
   git clone https://github.com/Cvio/model-converter.git
   cd model-converter
   .\setup.ps1
   ```

   This installs two separate toolsets: one for converting models, and one for training (about
   3 GB, because it includes the graphics-card version of PyTorch, the library that does the
   training). It ends by training a tiny test model for ten steps, to prove the graphics card
   works. **Expect** it to finish with `OK. This machine can train.` It takes 10 to 20 minutes the
   first time. If PowerShell refuses to run it, use
   `powershell -ExecutionPolicy Bypass -File .\setup.ps1` instead.

3. **Log in to Hugging Face** (free). Some datasets are only shared with logged-in users. Make an
   account at huggingface.co, create a token under **Settings › Access Tokens**, then run this and
   paste the token when it asks:

   ```powershell
   uv run hf auth login
   ```

   Answer **no** if it asks about a git credential.

4. **Tell it about this computer.** Copy `machine.example.yaml` to `machine.yaml` (it holds
   settings that differ between computers) and open it in Notepad:
   - `volis_path`: the folder that has `volis.exe` in it. The last two steps put the trained
     model into volis there. If volis isn't installed on this computer, see "Training on one
     computer, finishing on another" below.
   - `teacher_gguf`: the teacher model file. On the 5090, use Qwen3-32B:
     ```yaml
     teacher_gguf:
       - inputs/teacher/Qwen3-32B-Q4_K_M.gguf
     ```
     On an 8 GB laptop, `inputs/teacher/Qwen3-8B-Q4_K_M.gguf` instead (only good enough for the
     rehearsal).

5. **Download the teacher and the program that runs it.** The teacher is large (Qwen3-32B is
   about 20 GB), so this takes a while:

   ```powershell
   uv run --project training python teacher/get_server.py
   uv run --project training hf download Qwen/Qwen3-32B-GGUF Qwen3-32B-Q4_K_M.gguf --local-dir inputs/teacher
   ```

   The first command fetches `llama-server`, the program that runs the teacher on the graphics
   card, and checks it downloaded correctly. The second fetches the teacher itself. (On the
   laptop, use `Qwen/Qwen3-8B-GGUF` and `Qwen3-8B-Q4_K_M.gguf`, about 5 GB.)

### Each time you train

1. **Download what the job needs:**

   ```powershell
   .\fetch.ps1 jobs\es-mx-whisper.yaml
   ```

   This downloads the starting model and the recordings the job file names into `inputs\`
   (about 6 GB for this job). **Expect** it to end with `OK. Everything es-mx-whisper.yaml names is in ...`.
   Anything already downloaded is skipped, so it's safe to run again.

2. **Train:**

   ```powershell
   .\train.ps1 jobs\es-mx-whisper.yaml
   ```

   This runs eight stages, W1 to W8, one after another, and prints what each one does:

   | Stage | What happens, simply |
   |---|---|
   | W1 | Checks the model and recordings are there and readable |
   | W2 | Gets the recordings ready: same sound format, split into "learn from" and "test on" sets by speaker (so the test is fair), and the teacher adds punctuation |
   | W3 | Tests the original model, so there's a score to beat |
   | W4 | Trains. This is the long part: several hours on the 5090 |
   | W5 | Tests the trained model on the same recordings, and stops if it didn't get better on the accent, or got worse on ordinary speech |
   | W6 | Combines the add-on with the original model, and checks the result still scores the same |
   | W7 | Converts it into the files volis reads, and installs it into volis |
   | W8 | Tests the converted version the way volis will run it |

   **Expect** it to end with `Every stage passed`. Everything it makes goes in
   `runs\es-mx-whisper\`.

   **If it stops**, it prints `STOP:`, the reason in plain words, and the command to carry on,
   such as `.\train.ps1 jobs\es-mx-whisper.yaml -From W4`. Fix what it says and run that command;
   stages that already passed aren't repeated.

3. **Listen to it.** Numbers can improve while the model gets worse in ways they miss, so a person
   has to try it. Open volis, tick **Compare recognizers**, press **Start** and speak Spanish. The
   new model (for this job, **Whisper large-v3-turbo Mexican Spanish (int8)**) is listed beside the
   others, each writing down what you said, so you can see which hears you best.

### The rehearsal: a quick test run

Before the first real run on a new computer, or on a laptop, run the rehearsal. It does all eight
stages on about an hour of recordings with a small model, so it finishes in about an hour:

```powershell
.\fetch.ps1 jobs\es-mx-whisper-small-rehearsal.yaml
.\train.ps1 jobs\es-mx-whisper-small-rehearsal.yaml
```

Its model shows up in volis as **Whisper small Mexican Spanish (rehearsal, int8)**. Treat it as
proof that everything works, not as a model to use: an hour of recordings is too little to
learn an accent well.

### Training on one computer, finishing on another

The last two stages need volis on the same computer. If the 5090 desktop doesn't have volis,
train there and finish on the computer that does:

1. On the desktop, stop after stage W6:

   ```powershell
   .\train.ps1 jobs\es-mx-whisper.yaml -To W6
   ```

2. Copy the whole `runs\es-mx-whisper\` folder to the same place on the other computer (a USB
   drive works; it's a few GB). That other computer needs this project set up too (steps 1, 2 and
   4 above; no graphics card or teacher needed for these stages), and the job's downloads,
   because the conversion reads the starting model's word list:

   ```powershell
   .\fetch.ps1 jobs\es-mx-whisper.yaml
   ```

3. There, finish from stage W7:

   ```powershell
   .\train.ps1 jobs\es-mx-whisper.yaml -From W7
   ```

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

Then copy `machine.example.yaml` to `machine.yaml` and set `volis_path` to the folder
`volis.exe` is in (the older name `cnverc_path` still works). `machine.yaml` holds what differs between PCs and is not committed.

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
   - `volis's engine could not run` naming a DLL outside this folder means something else
     loaded its own `onnxruntime.dll` into Python. If that path belongs to security software, it
     is the same conversation: ask for the exclusion, then reinstall.
   - `uv is ...` or `git is not installed` means fix what it names and rerun.
5. Copy `machine.example.yaml` to `machine.yaml` and set `volis_path` to where `volis.exe`
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
     folder_name: whisper-large-v3-turbo-es-adriszmar   # its folder in volis's models/asr/
     display_name: Whisper large-v3-turbo Spanish (adriszmar, int8)   # shown in volis
     languages: [es]
     # varieties: [es-MX]   # optional: the dialect it's tuned for (volis ranks it first)

   use_fp32: false            # true ships full-size files; step 5 says when you need it

   test:
     wav: test_audio/es-16k.wav   # a short clip in that language, 16 kHz mono
   ```

   Where volis is comes from `machine.yaml`, not the config.

2. **Put a test recording in `test_audio/`.** Recordings are not committed, so a fresh clone
   has none. Use a few seconds of speech in the model's language, 16 kHz mono. `volis --listen --wav` saves exactly that to `logs/segments/`. Any other file
   converts with `ffmpeg -i in.wav -ar 16000 -ac 1 out.wav`.

3. **Run the seven steps in order.** Each one prints what it did and stops with a clear
   message if a check fails:

   ```bash
   uv run python whisper-to-onnx/steps/1_check_setup.py --config whisper-to-onnx/configs/your-model.yaml
   ```

   Step 1 runs `0_doctor.py` first. Then run `2_download.py`, `3_to_openai_format.py`,
   `4_export_onnx.py`, `5_check_int8.py`, `6_assemble.py` and `7_verify.py`, each with the
   same `--config`.

4. **Listen to it.** Step 7 installs the folder into volis and tells you how to compare it
   with the recognizer you already have: tick **Compare recognizers** in volis and speak.

Everything a run produces goes in `runs/<run_name>/`. A step never overwrites an earlier
run's output unless you pass `--force`.

## What each step does, and what it checks

| Step | Does | Stops if |
|---|---|---|
| 1 Check setup | Prints versions, checks sherpa-onnx is at v1.13.8, and reads the export script on disk | The patch no longer fits the export script; the test clip isn't 16 kHz mono |
| 2 Download | Fetches only the weights and configs; the tokenizer comes from the base model; writes `arch.json` | It's a LoRA adapter (merge it first); the architecture doesn't fit openai-whisper; the vocabulary differs from the base model's |
| 3 To OpenAI format | Renames every weight into OpenAI Whisper's layout | Any weight is left over or missing; **the converted model's outputs differ from the original's on the same input** |
| 4 Export ONNX | Runs sherpa-onnx's export script, patched, on the checkpoint | **The ONNX files' outputs differ from the checkpoint's on the same input**; the encoder is the wrong size |
| 5 Check int8 | Chooses int8 files that still transcribe sensibly in volis's engine: the export script's own, or a per-channel requantization if those fail | Neither int8 version is sensible (then set `use_fp32: true`) |
| 6 Assemble | Builds the folder and `engine.toml` | `engine.toml` breaks any of volis's rules |
| 7 Verify | Installs into volis, runs `--report`, transcribes the installed folder with volis's engine | volis doesn't list it as `ok`, or it transcribes nothing |

"volis's engine" is the sherpa-onnx Python package at 1.13.8, the same C++ recognizer volis
links, given the settings volis uses. It decides every transcript-based check, because it is what
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
  badly broken (61% agreement with fp32), while volis's engine, given the same files, differed
  from fp32 by one word. Steps 4, 5 and 7 therefore transcribe with volis's engine and never
  with `test.py`. If int8 does fail there, step 5 tries a per-channel requantization (same
  size) before falling back to fp32.
- **One 5-second clip is a thin basis for int8.** On an unclear stretch, int8 can change a word
  from run to run. Step 7's side-by-side comparison in volis, on real speech, is the real
  judge.
- **PyTorch 2.9 and newer** export with a new engine that fails on Whisper. The patch passes
  `dynamo=False`.
- **transformers 5 no longer ships its conversion scripts.** The weight-name mapping comes from
  `convert_openai_to_hf.py` in transformers v4.46.0, copied into step 3 with its source named.
- **sherpa-onnx's own package leaves out its runtime unless asked.** Its wheels require
  `sherpa-onnx-core`, which puts `onnxruntime.dll` 1.28.2 (the version volis links) beside
  sherpa-onnx's extension. Its sdist declares no dependencies, and uv locks from the sdist, so
  `pyproject.toml` names `sherpa-onnx-core` itself. Without it, sherpa-onnx finds whatever
  `onnxruntime.dll` Windows offers, and Windows 11 has an old one (1.17) in System32 that crashes it.
- **volis's engine runs in a process of its own** (`engine_worker.py`), which checks which
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

## Training (being built)

model-converter is growing from convert-only into a train-and-convert app
(`train-and-convert-app.md` has the plan). What exists so far:

- **A second environment for training,** in `training/`, with CUDA PyTorch (built for CUDA 12.8,
  which the RTX 5090 needs). The converter's environment stays CPU-only. `setup.ps1` installs
  both, then runs `training/steps/gpu_check.py`: versions, the GPU, and a ten-step LoRA on
  `whisper-small`. On a PC without an NVIDIA GPU it says training isn't possible there; the
  converter still works.
- **`fetch.ps1 <job file>`** downloads every `hf:` model and dataset a job names into `inputs/`:
  only weights, configs and tokenizer files for models (safetensors when a repo has them), and
  Parquet only for datasets. A dataset published only as a loading script comes from Hugging
  Face's automatic Parquet copy. A gated dataset stops and says what to accept, then
  `uv run hf auth login`.
- **The teacher**: a larger model run through llama.cpp's `llama-server`, used to prepare
  training data (restoring punctuation now; translating later). It never ships.
  `uv run --project training python teacher/get_server.py` downloads a pinned prebuilt CUDA
  build into `teacher/llama.cpp/`, checking each archive's SHA-256. `machine.yaml` names the
  teacher models (`teacher_gguf:`, a list) and the server (`llama_server:`).
  `uv run --project training python teacher/check.py` punctuates ten CIEMPIESS lines and checks
  no word changed.
- **Whisper jobs, W1 to W8.** A job file in `jobs/` describes the base model, the data and the
  LoRA; two commands do the rest:

  ```powershell
  .\fetch.ps1 jobs\es-mx-whisper-small-rehearsal.yaml
  .\train.ps1 jobs\es-mx-whisper-small-rehearsal.yaml
  ```

  `train.ps1` runs the stages in order and stops at the first failed check, printing
  `.\train.ps1 <job> -From W4` to carry on. A finished stage isn't repeated; `-Force` redoes
  them. The stages: W1 check the model and data; W2 prepare (16 kHz, split by speaker, restore
  punctuation, mix in ordinary speech); W3 score the base model; W4 train the LoRA (encoder and
  decoder, best validation checkpoint kept); W5 compare base and tuned (stops if the dialect
  didn't improve or ordinary speech got over a point worse); W6 merge, and check the merged model
  scores like base + LoRA; W7 run the converter's seven steps on it; W8 score the int8 model in
  volis's engine. Everything goes in `runs/<job>/`.

  `jobs/es-mx-whisper-small-rehearsal.yaml` is a quick run of every stage on about an hour of
  CIEMPIESS with whisper-small (about an hour on an 8 GB laptop GPU). `jobs/es-mx-whisper.yaml`
  is the real job, for the 5090.

### Things that went wrong while building it

- **Windows PowerShell 5.1 turns a program's stderr into an error when output is redirected.**
  With `$ErrorActionPreference = "Stop"`, Hugging Face's "unauthenticated requests" warning
  then stops a working download (`convert.ps1 ... *> log` fails in step 2 for this reason).
  `fetch.ps1` checks exit codes instead. In a console, without redirection, it doesn't happen.
- **Many model repos publish the same weights twice** (`model.safetensors` and
  `pytorch_model.bin`). `fetch.ps1` takes safetensors when there are any.
- **llama-server's log no longer names the GPU at its default level**, so reading the log to
  prove the model is on the GPU fails. The teacher asks the NVIDIA driver instead. On Windows
  the driver doesn't report per-process memory, so it compares total GPU memory in use before
  and after the model loads.
- **A teacher told only "change no word" still tidies speech.** Qwen3-8B dropped a repeated
  "que que", a false start ("contra contratada") and corrected "dejé". The prompt now names
  those cases; the word-for-word check catches any it still makes.
- **CUDA 12.4 builds of llama.cpp don't run on the RTX 5090** (Blackwell). The pinned build is
  CUDA 13.4, which runs on it and on the laptop's RTX 4070.
- **Separate datasets reuse speaker IDs.** CIEMPIESS LIGHT and CIEMPIESS TEST both have speakers
  F_01 to M_10, and they are different people ("Speakers in the CL are not present in any other
  CIEMPIESS dataset"). W2 labels each ID with its dataset, so the overlap check counts people,
  not labels.
- **The written accent on question words isn't a changed word.** The teacher adds ¿ and ?, and
  with them qué, cómo, dónde; CIEMPIESS leaves those accents out. Counting que -> qué as a
  changed word threw away 15% of lines. The word check now ignores accent marks (not ñ).
- **Gradient checkpointing can silently freeze Whisper's encoder LoRA.**
  `enable_input_require_grads()` only reaches the decoder: the encoder's input comes from a
  frozen convolution, so nothing flowing into its checkpointed layers needs gradients. W4 hooks
  the first convolution and checks every encoder LoRA weight gets a gradient before the first
  step. (Checking that weights moved after the first step doesn't work: the learning rate warms
  up from zero.)
- **A little dialect data can hurt ordinary speech.** The rehearsal (1.6 h, 8 speakers) improved
  CIEMPIESS test WER from 16.5 to 12.7 but made FLEURS 1.67 points worse: CIEMPIESS writes
  numbers as words ("seis y media") and the model copied it. `data.mix` mixes ordinary speech into
  training (the real job: a fifth FLEURS `es_419` train); the rehearsal job loosens the limit to
  2.0 instead, so it could prove the later stages.
- **The converter's engine loads the model once per clip.** Fine for one test clip; slow for
  W8's hundreds. `engine_worker.py` has a `--wav-list` mode (`transcribe_many_like_cnverc`) that
  loads it once; its one-clip behaviour and runtime checks are unchanged.

## Security software (McAfee, Trellix and the like)

On-access scanners sometimes quarantine or lock DLLs while uv writes them: torch's, onnxruntime's,
sherpa-onnx's. The install then looks finished, and a step fails much later when it first loads
the missing file (usually step 4, the first to load ONNX Runtime and sherpa-onnx).
`0_doctor.py` checks for exactly this, before anything runs:

- every `.dll`/`.pyd` the packages installed is present and matches the hash in the package's
  own `RECORD`; it names each one that is missing or changed;
- volis's engine loads, on onnxruntime 1.28.2 from the environment, and names the DLL it got if not;
- torch and onnxruntime import;
- which other `onnxruntime.dll` copies are on the machine, and whether McAfee/Trellix is installed.

If it reports a missing or altered file, ask whoever manages the security software for an
on-access scan exclusion for this folder (the doctor prints the path and a sentence to send),
then run `.\setup.ps1 -Reinstall`. Everything is under that one folder by design. Don't turn the
scanner off to get around it. If no exclusion is possible, steps 1 to 6 could run under WSL2
instead (not set up here); step 7 needs Windows, because it runs `volis.exe`.

`runs\_doctor.json` records what the doctor found. Comparing it between two machines is the
quickest way to see what differs.

## Tested with

| Config | Model | Result |
|---|---|---|
| `es-small-hitz.yaml` | `HiTZ/whisper-small-es` (`pytorch_model.bin`, 80 mels, 12 decoder layers) | All seven steps pass. Logits: 0 (step 3), 0.00004 (step 4). int8: 375 MB, 94% agreement with fp32 in volis's engine |
| `es-turbo-adriszmar.yaml` | `adriszmar/whisper-large-v3-turbo-es` (safetensors, 128 mels, 4 decoder layers) | All seven steps pass. Logits: 0 (step 3), 0.00008 (step 4). int8: 1,036 MB, 100% agreement with fp32 |

These numbers were measured with volis's engine running on onnxruntime 1.30. On 1.28.2 (what
volis links, and what the engine uses now) the shipped turbo int8 files give the same words;
the small int8 files differ by one word on the unclear stretch the notes above describe.

Both ran on the same scripts; only the configs differ. Both are installed in volis and
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
  train.ps1             runs every stage of a training job, in order
  fetch.ps1             downloads a job's hf: models and datasets into inputs/
  jobs/                 one .yaml per training job
  inputs/               base models, datasets and teacher models (not committed)
  training/             the training environment (its own uv project, CUDA PyTorch)
    steps/              common.py, fetch.py, gpu_check.py, textnorm.py, whisper_common.py
      whisper/          the Whisper job's stages, w1_check.py to w8_int8.py
  teacher/              teacher.py (runs llama-server), get_server.py, check.py
    llama.cpp/          the pinned prebuilt llama-server (not committed)
  train-and-convert-app.md   the plan the training parts are built from
  whisper-to-onnx-converter.md   the original instructions this was built from
```