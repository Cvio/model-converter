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

## Setting it up

You need [uv](https://docs.astral.sh/uv/) and Git. uv installs the right Python (3.12) by itself.

```bash
cd model-converter
```

```bash
uv sync
```

That installs PyTorch (the CPU build, which is all the conversion needs), transformers,
openai-whisper, onnx, onnxruntime and sherpa-onnx 1.13.8. The first run of step 1 also clones
sherpa-onnx's source at v1.13.8 into `vendor/`, for its export script.

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

   cnverc:
     path: D:/AI_Data/projects/cnverc/target/release   # the folder cnverc.exe is in
   ```

2. **Put a test recording in `test_audio/`:** a few seconds of speech in the model's language,
   16 kHz mono. `cnverc --listen --wav` saves exactly that to `logs/segments/`. Any other file
   converts with `ffmpeg -i in.wav -ar 16000 -ac 1 out.wav`.

3. **Run the seven steps in order.** Each one prints what it did and stops with a clear
   message if a check fails:

   ```bash
   uv run python whisper-to-onnx/steps/1_check_setup.py --config whisper-to-onnx/configs/your-model.yaml
   ```

   and then `2_download.py`, `3_to_openai_format.py`, `4_export_onnx.py`, `5_check_int8.py`,
   `6_assemble.py`, `7_verify.py`, each with the same `--config`.

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
- **Windows 11 has an old `onnxruntime.dll` in System32** (1.17.1). sherpa-onnx's Python package
  crashes if it loads that one, so `common.transcribe_like_cnverc` points it at the onnxruntime
  package's own DLL first.
- **PyTorch exports a model over 2 GB as hundreds of loose files**, one per tensor. The export
  script gathers them into one `.weights` file but leaves the loose ones behind; step 4 deletes
  them before checking the export.
- **Python 3.14** doesn't have wheels for everything here yet; `pyproject.toml` pins 3.12.
- **Many fine-tunes ship `pytorch_model.bin`**, not safetensors. Both work.
- **A fine-tune with added tokens can't be converted this way.** sherpa-onnx writes `tokens.txt`
  from Whisper's standard vocabulary, so step 2 refuses a model whose vocabulary size differs from
  its base model's.

## Tested with

| Config | Model | Result |
|---|---|---|
| `es-small-hitz.yaml` | `HiTZ/whisper-small-es` (`pytorch_model.bin`, 80 mels, 12 decoder layers) | All seven steps pass. Logits: 0 (step 3), 0.00004 (step 4). int8: 375 MB, 94% agreement with fp32 in cnverc's engine |
| `es-turbo-adriszmar.yaml` | `adriszmar/whisper-large-v3-turbo-es` (safetensors, 128 mels, 4 decoder layers) | All seven steps pass. Logits: 0 (step 3), 0.00008 (step 4). int8: 1,036 MB, 100% agreement with fp32 |

Both ran on the same scripts; only the configs differ. Both are installed in cnverc and
listed as `ok`. Whether either beats the stock Whisper turbo on your own speech is step 7's
listening test, which hasn't been done yet.

## Layout

```
model-converter/
  pyproject.toml        the Python environment (uv)
  whisper-to-onnx/
    configs/            one .yaml per model
    steps/              the seven steps, and common.py
    patches/            the change to sherpa-onnx's export script, kept visible
  test_audio/           test recordings (not committed)
  vendor/               sherpa-onnx at v1.13.8, cloned by step 1 (not committed)
  runs/<run_name>/      everything a run produces (not committed)
  whisper-to-onnx-converter.md   the original instructions this was built from
```
