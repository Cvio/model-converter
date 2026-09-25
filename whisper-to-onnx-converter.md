# Instructions: build a Whisper → ONNX converter for cnverc

For Claude Code. Build a small Python app that takes any fine-tuned Whisper model from Hugging
Face and turns it into a model folder that cnverc detects and runs.

Read this whole file before writing any code.

The converter is the deliverable. No particular model matters. It has to work on a model
downloaded from Hugging Face, on a model produced by the LoRA training app, and on whatever
comes next.

---

## Why this is not a one-line job

There is a general-purpose way to turn a Hugging Face model into ONNX (`optimum-cli export
onnx`). **It does not work here.** cnverc runs its models through sherpa-onnx, and sherpa-onnx
does not load generic ONNX files. It expects its own layout: a separate encoder file, a
separate decoder file, a tokens file, and specific information baked into the ONNX metadata.
Only sherpa-onnx's own export script produces that.

That script has its own requirement: it reads the original OpenAI Whisper checkpoint format,
not the Hugging Face format. So the real work is a format conversion in the middle.

The whole path:

```
Hugging Face safetensors
  → OpenAI Whisper checkpoint format     (rename every weight)
  → sherpa-onnx export script            (produces encoder.onnx, decoder.onnx, tokens.txt)
  → int8 quantization                    (makes it small enough to ship)
  → model folder + engine.toml           (what cnverc reads)
```

---

## What to check in a model repo before converting

Do this for every model, not just the first one. Four things vary and all four break the run
if assumed.

**1. Is it a full fine-tune or a LoRA?** A full fine-tune has one `model.safetensors` roughly
the size of the base model. A LoRA has a small `adapter_model.safetensors` plus an
`adapter_config.json`. A LoRA cannot be converted directly — it has to be merged into its base
model first, which is step 6 of the training app. If the converter sees `adapter_config.json`,
it should stop and say so rather than convert something meaningless.

**2. Are the tokenizer files there?** Many fine-tunes ship only what they changed, so
`tokenizer.json`, `vocab.json` and `merges.txt` are often missing. That is normal. The
converter pulls them from the base model instead. `base_model` in the config exists for this.

**3. What is the architecture?** Read it from `config.json` every time. Never hardcode.
whisper-small has 12 decoder layers and 80 mel bins. whisper-large-v3-turbo has 4 decoder
layers and 128 mel bins. A converter with numbers baked in works once and then produces silent
garbage on the next model.

**4. What is safe to skip downloading?** Training runs leave behind `optimizer.pt`,
`rng_state.pth`, `scheduler.pt`, `trainer_state.json` and `training_args.bin`. These are often
larger than the model itself — `optimizer.pt` is routinely twice the size. None are needed.

A fifth thing worth reading if present: `trainer_state.json` records the word error rate the
model reported during training. That is not a verdict on the model — the test set it was
measured on is usually not named, and for languages without standard spelling the number is
inflated badly. But it is free information. Print it if it is there.

---

## Rules

1. One script per step. Each runs on its own. Each prints what it did.
2. Every setting lives in `config.yaml`. Nothing about a specific model goes in a script.
3. Everything a run produces goes in `runs/<name>/`. Never overwrite an old run.
4. Never skip a check to keep going. A conversion that silently half-works produces a model
   that loads fine and transcribes nonsense.

## Layout

```
whisper-to-onnx/
  config.yaml
  steps/
    1_check_setup.py
    2_download.py
    3_to_openai_format.py
    4_export_onnx.py
    5_quantize.py
    6_assemble.py
    7_verify.py
  runs/<name>/
    hf/          downloaded Hugging Face files
    openai/      the converted checkpoint
    onnx/        exported and quantized files
    out/         the finished model folder for cnverc
  README.md
```

## config.yaml

Everything model-specific lives here. Converting a different model means a new config file,
not new code. The values below are an example.

```yaml
model_id: otozz/whisper-small-dialect_iraqi
base_model: openai/whisper-small     # where the tokenizer comes from
run_name: whisper-small-iraqi

engine:
  folder_name: whisper-small-ar-iq   # the folder name inside cnverc's models/asr/
  display_name: Whisper small Iraqi (int8)
  languages: [ar]

test:
  wav: test_audio/sample.wav         # 16 kHz mono, used by steps 3 and 7
  compare_against: models/asr/whisper-large-v3-turbo   # inside cnverc
```

A local path works in `model_id` too. The model coming out of the LoRA training app is a
folder on disk, not a Hugging Face ID, and the converter must accept both.

---

## Step 1 — check the tools

Script: `steps/1_check_setup.py`

Print the version of: Python, PyTorch, `transformers`, `onnx`, `onnxruntime`.

Clone sherpa-onnx if it is not already present, and **read**
`scripts/whisper/export-onnx.py` before doing anything else. Print two things from it:

- The list of model names it recognises.
- How it loads a checkpoint — by name, or from a local file.

That script is the thing everything else has to satisfy. Its behaviour has changed between
versions, so read the copy that is actually on disk rather than assuming.

**Check:** all versions print, and the export script's loading behaviour is printed.

On PyTorch 2.9 or newer, the export in step 4 needs `dynamo=False` passed to
`torch.onnx.export`. The newer exporter fails on how Whisper looks up positional embeddings.
Note the PyTorch version now so step 4 knows whether it applies.

---

## Step 2 — download

Script: `steps/2_download.py`

From `model_id`, download only:

- `model.safetensors`
- `config.json`
- `generation_config.json`
- `preprocessor_config.json`

Do not download the training leftovers listed above.

**Stop if `adapter_config.json` is present.** That is a LoRA, not a full model. Print that it
needs merging first and where that happens.

From `base_model`, download the tokenizer files.

Then read `config.json` and print the architecture: encoder layers, decoder layers, d_model,
attention heads, vocab size, mel bins. Save those numbers to the run folder. Every later step
reads them from there.

If `trainer_state.json` exists in the repo listing, read its reported error rate and print it
without downloading the whole file.

**Check:** the printed architecture is sensible and matches what the model claims to be.

---

## Step 3 — convert to OpenAI format

Script: `steps/3_to_openai_format.py`

This is the step that breaks. Everything else is plumbing.

Hugging Face and OpenAI store the same Whisper weights under different names.
`model.encoder.layers.0.self_attn.q_proj.weight` in one is `encoder.blocks.0.attn.query.weight`
in the other. Every weight in the model needs renaming, and the output has to be a file
containing exactly two things: a `dims` dictionary and a `model_state_dict` dictionary.

Do not write the mapping from memory. Two reliable sources:

- The `transformers` library ships `convert_openai_to_hf.py`, which does this in the opposite
  direction. Invert it.
- Search for `convert_hf_to_openai.py`, which several projects publish.

Read one of those, then write the mapping. Build `dims` from the `config.json` values saved in
step 2 — never from constants.

Two details that are easy to miss. The output projection in Hugging Face (`proj_out`) shares
its weights with the token embedding, and OpenAI's format does not store it separately, so
drop it. And the attention key projection has no bias in either format, so do not invent one.

**Then verify the conversion before going any further.** This is the important part:

1. Load the original Hugging Face model and transcribe the test wav. Print the text.
2. Load the converted checkpoint with the `openai-whisper` package and transcribe the same
   file. Print the text.
3. Compare.

**Check:** the two transcripts are the same, or near enough that any difference is obviously
just decoding settings. If they differ meaningfully, the mapping is wrong. Find it now. A bad
mapping still exports to ONNX perfectly happily and produces a model that transcribes fluent
nonsense.

After the mapping is written, print any weight name that appeared in the input and was not
consumed by the mapping, and any name the mapping expected and did not find. On a model with a
different layer count or a different base, that list is the first thing that catches it.

---

## Step 4 — export to ONNX

Script: `steps/4_export_onnx.py`

Run sherpa-onnx's `scripts/whisper/export-onnx.py` against the checkpoint from step 3.

Depending on what step 1 found, either name the checkpoint file so the script picks it up, or
patch the script's loading function to take a path. Prefer patching, and keep the patch in
this repo as a `.patch` file so it is visible rather than a silent local edit.

If step 1 reported PyTorch 2.9 or newer, pass `dynamo=False` to `torch.onnx.export`.

Output: `<name>-encoder.onnx`, `<name>-decoder.onnx`, `<name>-tokens.txt`.

**Check:** all three files exist and the encoder is roughly the expected size. Then run
sherpa-onnx's own `scripts/whisper/test.py` against them with the test wav, and confirm the
transcript still matches step 3's.

---

## Step 5 — quantize

Script: `steps/5_quantize.py`

Shrink both ONNX files to int8 using dynamic quantization from `onnxruntime`. This roughly
quarters the size. cnverc's other models are int8, so this keeps it consistent.

**Check:** run `test.py` again on the int8 files. The transcript will not be identical to the
full-size one and does not need to be. It does need to still be sensible. If the int8 output
is garbage where the full-size output was fine, stop and report it — ship the full-size files
instead and note the size cost.

---

## Step 6 — assemble the folder

Script: `steps/6_assemble.py`

Build the folder cnverc expects, named from `engine.folder_name`:

```
<folder_name>/
  engine.toml
  encoder.int8.onnx
  decoder.int8.onnx
  tokens.txt
```

Write `engine.toml` from the config:

```toml
name = "Whisper small Iraqi (int8)"
kind = "segment"
backend = "whisper"
languages = ["ar"]

[files]
encoder = "encoder.int8.onnx"
decoder = "decoder.int8.onnx"
tokens  = "tokens.txt"
```

Three things cnverc's model reader is strict about. The file names in `engine.toml` must match
the files on disk exactly. The only keys allowed are `name`, `kind`, `backend`, `languages`,
`data_dir` and `files` — any other key is rejected outright and the model shows as broken.
And `languages` must contain the language code cnverc will ask for.

**Check:** print the folder contents and the `engine.toml`, and confirm every declared file is
present.

---

## Step 7 — verify inside cnverc

Script: `steps/7_verify.py`, plus two things done by hand.

Copy the folder into cnverc's `models/asr/` and run `cnverc --report`. The new model must show
as `ok`, with a `[+]` beside every file.

Then the comparison. With both the new model and the model named in `test.compare_against`
installed, run `cnverc --listen --compare` and speak — or play — real audio in the target
language. cnverc runs both models on the same audio and prints the transcripts side by side.

**Check:** the converted model loads, runs, and produces text in the right language.

The comparison answers whether that particular model is worth keeping. Either answer is a
successful run of the converter. A converted model that turns out to be worse than what cnverc
already has means the model was weak, not that the conversion failed.

---

## Reusing this

Everything model-specific is in `config.yaml`. Converting a different Whisper model means a
new config file, not new code. If any step needs editing to handle a different model, that is
a bug in the step — fix it so the value comes from the config instead.

Two things genuinely vary per run and are handled deliberately rather than assumed:

- The export script's behaviour in step 4, because sherpa-onnx changes it between versions.
  That is why step 1 reads the script and why the patch is kept as a visible file.
- The weight name mapping in step 3, if a model uses a base Whisper size the mapping has not
  seen. That is why step 3 prints unmatched names on both sides.

## One cnverc fix before testing a new language

`language_name` in cnverc's `translate.rs` only knows a fixed set of languages. Adding a model
for a language it does not know means the translation prompt reads "into ar" instead of "into
Arabic". Fix that first for any new language, or the translation half of the test measures a
bug rather than the model.
