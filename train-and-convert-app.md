# Instructions: turn model-converter into one train-and-convert app

For Claude Code. Extend the existing `model-converter` repo
(`D:\AI_Data\projects\model-converter`) so it does the whole job:

```
safetensors model + training data
  → LoRA training
  → merge the LoRA into the model
  → convert: Whisper → ONNX folder, Qwen → .gguf
  → install into volis and check it runs
```

The user drops a model into one folder and data into another, writes a short job file, and runs
one command.

## How to work on this

- **Work one build-order item at a time** (the list at the end of this file). After each one,
  stop, show its check passing, and wait for the user to say go on. Don't start the next item
  on your own.
- Before item 1, summarise back what you understood: what exists, what changes, what's new. The
  user reads it to confirm no detail was missed.
- There are seven build-order items.
- Item 4 changes the **volis** repo, not this one. When you reach it, stop and tell the user.
  It's done in a separate session there, under volis's own `CLAUDE.md` rules.
- If something in this file turns out to be wrong about the repo, stop and say so. Don't work
  around it.
- The repo has uncommitted work from before this file: edits to `README.md`, one config and
  three step scripts, plus the untracked `convert.ps1` and `test_audio/`. Before item 1, show the
  user `git status` and ask whether to commit it as it is. `test_audio/` stays uncommitted (its
  `.wav` files are ignored on purpose).
- Commit after each item passes.
- Some stages can only be finished by the user: listening, rating translations, logging in to
  Hugging Face. The section "What only the user can do", near the end, lists them. When a stage
  reaches one, stop and say exactly what's needed.

Read this whole file, then the repo's `README.md`, before writing any code. The existing
Whisper → ONNX converter works and has been tested on two models. **Don't rewrite it.** Build the
new parts around it, and change it only where this file says to.

**Don't rename anything that exists.** The repo, its folders, its scripts and its existing
config keys keep their names.

---

## Terms

- **LoRA**: a small file of adjustments trained on top of a locked model. The original model
  never changes during training.
- **Merging**: adding the LoRA's adjustments into the original model's weights and saving the
  result as one ordinary model. After merging, there's no separate LoRA any more, just a normal
  model that behaves like the original plus the training. It's one function call in the `peft`
  library (`merge_and_unload()`). What takes care is checking it worked: see "Merging" below.
- **Safetensors**: the normal format for model weights that training tools read and write.
- **ONNX / GGUF**: run-only formats. ONNX is what volis uses for Whisper. GGUF is what it uses
  for Qwen. Neither can be trained, which is why training happens before conversion.

---

## What's in the repo today, and what has to change

Reviewed on 2026-09-27.

| Found | What it means for this work |
|---|---|
| A tested 7-step Whisper → ONNX converter, run by `convert.ps1 <config>` | Reused as-is for the Whisper conversion stage |
| Step 2 accepts a local folder as `model_id`, and **stops if it finds a LoRA** (`adapter_config.json`) | Good. The new merge stage feeds it a merged folder |
| Step 2 **doesn't support sharded weights** (`model.safetensors.index.json`) | The merge stage must save one weights file (see "Merging") |
| `pyproject.toml` installs **CPU-only PyTorch**, on purpose | Training needs CUDA PyTorch. **Training gets its own environment** (see "Two environments") |
| Step 6 allows only `name, kind, backend, languages, data_dir, files` in `engine.toml` | volis now also accepts `varieties`. **Add it** to `ALLOWED_KEYS`, and write it when the config has `engine.varieties` |
| Step 6 accepts only two- or three-letter language codes | Unchanged: `languages` stays plain (`ar`); the dialect goes in `varieties` (`ar-IQ`) |
| Step 7 looks for **`cnverc.exe`**; `machine.yaml` has `cnverc_path` | The app is now **`volis.exe`**. Make step 7 look for `volis.exe`, falling back to `cnverc.exe`. Keep reading `cnverc_path`, and also accept `volis_path` |
| The README, `machine.example.yaml` and the step scripts' messages say "cnverc" throughout | Say **volis** in text people read. Keep file names, folder names and config keys as they are |
| No Qwen → GGUF converter yet ("next", says the README) | Built here, as `qwen-to-gguf/` |
| volis runs `qwen3-1.7b-q4_k_m.gguf`. Its `models/mt/` must hold **exactly one** `.gguf` | The Qwen install step has to move the current file aside, never delete it |
| volis links `llama-cpp-2` **0.1.156** | The GGUF must be made with the same llama.cpp commit that crate pins (see "Qwen → GGUF") |
| The crate's copy of llama.cpp in the Cargo registry has `convert_hf_to_gguf.py` but **no `gguf-py`**, and its `tools/` has **no `quantize` or `server`** | That copy can't do the conversion. Clone llama.cpp at the pinned commit instead (see "Qwen → GGUF") |
| volis logs to **stdout** as well as its log file (`init_logging` in `src/main.rs`) | The two volis additions must send logs to stderr, or their output is unusable (see "Two small volis additions") |
| volis's prompt is in `src/translate.rs` (`prompt_for`, `system_prompt`), ChatML with an empty `<think></think>` block | Qwen must be trained on exactly that prompt (see "Training Qwen") |

---

## Data format: which one, and why

**Speech (for Whisper): Parquet.** One file holds the audio *and* the text, so a dataset can't
lose track of its wav files. It's the format Hugging Face publishes speech datasets in (CIEMPIESS
and the others we've looked at), so a downloaded dataset drops in unchanged. Expected columns:

| column | required | contents |
|---|---|---|
| `audio` | yes | Hugging Face audio (`{bytes, path}`) or raw file bytes |
| `text` | yes | the transcript |
| `speaker` | no, but strongly recommended | who is speaking. Needed to split test data fairly |
| `split` | no | `train` or `test`, if the data already has a split |

Column names differ between datasets. The job file maps them (`columns: {text: normalized_text}`).

JSONL plus a folder of wavs is also accepted for speech, for recordings the user makes
themselves: `{"audio": "clips/0001.wav", "text": "...", "speaker": "..."}`, with paths relative
to the JSONL file. The prepare step converts it to Parquet, so everything after it sees one
format.

**Text pairs (for Qwen): JSONL.** It's text only, so there's no audio to embed. It's readable
and fixable in Notepad, and one line is one example:

```jsonl
{"source_lang": "es-MX", "target_lang": "en", "source": "¿Quieren ir a comer algo?", "target": "Do you guys want to grab something to eat?"}
{"source_lang": "en", "target_lang": "es-MX", "source": "My phone died.", "target": "Se me murió el celular."}
```

The language tags are the ones volis uses (`src/varieties.rs`). The prepare step refuses a tag
volis doesn't know, since training on a prompt volis will never send is wasted.

Pairs don't have to be written by hand. The **pairs job** (below) makes them from the
transcripts of a speech dataset.

---

## Getting the inputs

The user may drop models and data into `inputs/` by hand. But the common ones come from Hugging
Face, so add `fetch.ps1`, which downloads everything a job file names that isn't already there:

```powershell
.\fetch.ps1 jobs\es-mx-whisper.yaml
```

In a job file, any `base_model` or data entry may be a Hugging Face ID, not a folder:

```yaml
base_model: hf:openai/whisper-large-v3-turbo
data:
  train: hf:ciempiess/ciempiess_light@train
  regression: hf:google/fleurs:es_419@test      # id:config@split
```

`fetch.ps1` downloads each one into `inputs/models/<name>/` or `inputs/data/<name>/`, and
rewrites nothing in the job. The stages resolve `hf:` entries to those folders, and stop if a
folder is missing, printing the `fetch.ps1` command.

Rules for fetching:

- **Models:** only weights, configs, tokenizer and processor files. Never training leftovers
  (`optimizer.pt` and the rest). The existing converter's step 2 has the list: reuse it.
- **Datasets:** Parquet only. Many datasets on Hugging Face also publish an automatic Parquet
  copy (the `refs/convert/parquet` branch). Use that for datasets that are only published as a
  loading script. Newer versions of the `datasets` library refuse to run loading scripts.
  `google/fleurs` is one of these.
- **Gated datasets:** some require accepting terms on the website and logging in first.
  FLORES+ (`openlanguagedata/flores_plus`) is one. If a download is refused for that reason,
  stop and tell the user to accept the terms on the dataset's page and run
  `uv run huggingface-cli login`. Don't try to get around it.
- Print the size before downloading anything over 5 GB, and check there's room on the disk.

What the Mexican Spanish jobs need, as a starting list:

| What | Hugging Face ID | Used by |
|---|---|---|
| Whisper turbo | `openai/whisper-large-v3-turbo` | Whisper job |
| Whisper small | `openai/whisper-small` | the fast rehearsal run |
| Qwen 1.7B | `Qwen/Qwen3-1.7B` | Qwen job; the model volis runs, before compression |
| Mexican speech, train | `ciempiess/ciempiess_light` | Whisper job, pairs job |
| Mexican speech, test | `ciempiess/ciempiess_test` | Whisper job, pairs job |
| General Latin American speech | `google/fleurs`, config `es_419`, test split | Whisper regression check |
| General translation | `openlanguagedata/flores_plus`, Spanish and English, `dev` and `devtest` | Qwen general pairs and regression check (gated) |
| The teacher model | see "The teacher model" | pairs job |

Check column names on download and print them. **Don't assume `ciempiess_test` uses the same
column names as `ciempiess_light`.**

---

## Disk space

A run needs a lot of room: the base models, the downloaded data, checkpoints, a merged model, and
the converted files. `runs/` on this machine already holds 26 GB. Every job's check stage
estimates what the job will need and stops if the disk hasn't got it, with 20% to spare. Print
the estimate either way.

---

## Layout

New parts are marked `NEW`. Everything else exists already.

```
model-converter/
  inputs/                          NEW, not committed
    models/<folder>/               base models in safetensors, dropped in by the user
    data/<folder>/                 .parquet, or .jsonl (+ wavs)
  jobs/                            NEW: one .yaml per job, with the four Mexican Spanish
                                   examples in this file committed as starting points
  train.ps1                        NEW: the one command
  fetch.ps1                        NEW: downloads what a job names into inputs/
  teacher/                         NEW: starts and stops llama-server for the teacher
  pairs/steps/                     NEW: the pairs job
  training/                        NEW: its own uv project (CUDA PyTorch)
    pyproject.toml
    uv.lock
    steps/
  qwen-to-gguf/                    NEW
    steps/
  whisper-to-onnx/                 existing converter, small changes only
  convert.ps1  setup.ps1           existing; setup.ps1 also sets up training/
  runs/<job>/                      existing pattern: everything a job produces
```

---

## The one command

```powershell
.\train.ps1 jobs\es-mx-whisper.yaml
```

It runs every stage in order, stops at the first failed check, and prints the command to resume
from that stage. Same behaviour and wording as `convert.ps1` (`-From N`, `-Force`). A finished
stage isn't repeated.

### A Whisper job

```yaml
# jobs/es-mx-whisper.yaml
kind: whisper
name: es-mx-whisper                      # its folder under runs/
base_model: inputs/models/whisper-large-v3-turbo
language: es                             # what Whisper is told
variety: es-MX                           # written into engine.toml as varieties

data:
  train: inputs/data/ciempiess_light     # a folder of .parquet, or a .jsonl
  test: inputs/data/ciempiess_test       # optional; otherwise held back by speaker
  columns: { audio: audio, text: normalized_text, speaker: speaker_id }
  regression: inputs/data/fleurs-es_419  # optional: ordinary speech in the same language
  min_seconds: 0.5
  max_seconds: 30

lora: { rank: 32, alpha: 64, dropout: 0.05 }
training: { learning_rate: 1.0e-4, warmup_steps: 50, epochs: 3, batch_size: 8,
            gradient_accumulation: 2, eval_every_steps: 200 }

engine:                                  # passed on to the existing converter
  # languages defaults to [language]; varieties to [variety]
  folder_name: whisper-large-v3-turbo-es-mx
  display_name: Whisper large-v3-turbo Mexican Spanish (int8)
test_wav: test_audio/es-16k.wav          # the converter's own check clip
```

### A pairs job

Makes translation pairs from a speech dataset's transcripts, for the Qwen job to train on.

```yaml
# jobs/es-mx-pairs.yaml
kind: pairs
name: es-mx-pairs
variety: es-MX                           # the dialect side's language tag
other: en                                # the other side
sources:
  train: runs/es-mx-whisper/data/train   # transcripts from the Whisper job's prepare stage
  test: runs/es-mx-whisper/data/test     # test speakers only; becomes the pair test set
min_words: 5
teacher_check: hf:openlanguagedata/flores_plus    # reference translations for choosing the teacher
flores_code: spa_Latn                    # the FLORES+ language closest to the dialect (acm_Arab for ar-IQ)
```

Using the Whisper job's prepared data means the punctuation is already restored, and the
speakers split the same way. No test speaker's sentence can end up in translation training.

### A Qwen job

```yaml
# jobs/es-mx-qwen.yaml
kind: qwen
name: es-mx-qwen
base_model: inputs/models/Qwen3-1.7B     # safetensors, the same model volis runs quantized
data:
  train: runs/es-mx-pairs/out/train.jsonl    # made by the pairs job
  test: runs/es-mx-pairs/out/test.jsonl
  general: runs/es-mx-pairs/out/general.jsonl   # FLORES+ dev pairs; keeps general translation from getting worse
lora: { rank: 16, alpha: 32, dropout: 0.05 }
training: { learning_rate: 2.0e-4, epochs: 2, batch_size: 16, gradient_accumulation: 1 }
output:
  file_name: qwen3-1.7b-es-mx-q4_k_m.gguf
  quantization: Q4_K_M
```

Only `kind`, `name`, `base_model` and `data.train` are required. Everything else has the defaults
shown.

---

## Two environments

The converter's environment uses CPU-only PyTorch on purpose. Its README explains why, and it
guards a delicate set of DLLs (onnxruntime 1.28.2 from `sherpa-onnx-core`, checked by
`0_doctor.py`). **Don't add CUDA PyTorch to it.**

Training gets its own uv project in `training/`, with its own lock and its own `.venv`, inside
the repo like the existing one:

- PyTorch **built for CUDA 12.8 or newer**, from PyTorch's CUDA index. The 5090 is a Blackwell
  card, and older builds install without complaint and then fail on it.
- `transformers`, `peft`, `accelerate`, `datasets`, `soundfile`, `librosa`, `jiwer` (error
  rates), `sacrebleu` (chrF translation scores), `pyarrow`.

`train.ps1` runs training stages in `training/`'s environment and conversion stages in the
existing one. They share nothing but files in `runs/<job>/`.

`setup.ps1` sets up both. Keep its existing behaviour (cache and Python inside the repo, the
doctor at the end) and add a GPU check for the training environment: PyTorch version, CUDA
version, GPU name and memory, and a ten-step dummy LoRA run on `whisper-small`.

**Run training on the 5090 desktop.** The laptop's 4070 has 8 GB, which can train Qwen 1.7B but
is tight for Whisper turbo. The training check stage measures peak memory on a few real steps,
and halves the batch size (doubling accumulation) if the card is over 90% full.

The desktop gets the repo the way the README's "On another PC" section already describes: clone,
`setup.ps1`, `machine.yaml`. Keep that section accurate as the training parts are added.

---

## The teacher model

Two stages need a much larger model than anything volis runs, used once and never shipped:

- **Restoring punctuation** in speech transcripts before Whisper training (stage W2).
- **Translating** transcripts into English, to make translation pairs (the pairs job).

This is the **teacher**. It runs locally on the 5090, so the data never leaves the machine.

**How it runs:** a GGUF model through llama.cpp's `llama-server`, built from the same llama.cpp
source used for the Qwen → GGUF stage (see "Qwen → GGUF"). The stages start it, send it batches
over `localhost`, and stop it at the end. It runs several requests at once (`--parallel`), so
tens of thousands of short sentences take hours, not days. This is a training tool, not volis,
so a local server is fine here.

**Which model:** the largest general model that fits in 32 GB at Q4_K_M and is strong in both
Spanish and Arabic. Candidates to try: Qwen3-32B and Gemma 3 27B. **Pick by test, not by name.**
Stage P2 below does that.

**Decoding:** greedy (always the most likely next word), with thinking turned off for Qwen3
models, same as volis. The teacher's output has to be repeatable.

The teacher's model file is named in `machine.yaml`, because it depends on the machine's GPU,
not on the job. `teacher_gguf:` is a **list** of paths, even with one entry:

```yaml
teacher_gguf:
  - D:/models/Qwen3-32B-Q4_K_M.gguf
  - D:/models/gemma-3-27b-it-Q4_K_M.gguf
llama_server: D:/tools/llama.cpp/llama-server.exe
```

**The teacher's llama.cpp doesn't have to match volis.** It never ships; it only writes text. Use
any recent `llama-server` built with CUDA. llama.cpp's GitHub releases include prebuilt Windows
CUDA builds: use one built for CUDA 12.8 or newer, which the 5090 (Blackwell) needs. That usually
avoids installing NVIDIA's CUDA toolkit at all. Build it from source only if no suitable
prebuilt one exists. Check at startup that it really runs on the GPU (its log names the CUDA
device), and stop if it fell back to the CPU.

---

## Stages of a Whisper job

**W1 — check.** The base model folder has `model.safetensors` (or `pytorch_model.bin`),
`config.json`, and the processor and tokenizer files. Print the architecture from `config.json`.
The data opens, the mapped columns exist, and sample rows print.

**W2 — prepare.** Read Parquet or JSONL. Resample to 16 kHz mono. Drop clips outside
`min_seconds`–`max_seconds` and count them. If there's no test data, hold back 5% of *speakers*
(never clips) as test. Also hold back 5% of training speakers for validation. Print hours, clips
and speakers per split, and the **speaker overlap between train and test, which must be 0**.

**Restore punctuation** when the job says `restore_punctuation: true` (the default when the
transcripts have no punctuation at all). CIEMPIESS transcripts are all lowercase with no
punctuation. Trained on those as they are, Whisper would learn to stop punctuating, and volis
already has a problem with missing question marks: in testing, "¿Quieren…?" came out as a
statement and was translated as "they want…". So the teacher adds capital letters and
punctuation, including ¿ and ?, to each transcript once, and saves the result.

The teacher must not change any word. After it runs, compare each line with the original
after removing case and punctuation from both. **Any line whose words changed is thrown away and
counted.** If more than 5% are thrown away, stop: the teacher is rewriting, not punctuating.
Colloquial spellings (*pus*, *namás*) must survive unchanged. Print ten before-and-after
examples.

**W3 — baseline.** Score the base model on test and regression data. Set language and task on
the generation config first, so Whisper doesn't guess the language. Score by word error rate
and character error rate, after one shared text-cleaning function chosen by language. For
Spanish that means lowercasing and removing punctuation. For Arabic it also means removing
diacritics and unifying letter variants.

**W4 — train.** LoRA on the attention layers (`q_proj, k_proj, v_proj, out_proj`) in **both
the encoder and the decoder**. The encoder is where accent is heard, and turbo has only four
decoder layers. Three quiet failure points, all handled:

- With gradient checkpointing on, call `model.enable_input_require_grads()`, or training learns
  nothing.
- Set `forced_decoder_ids = None` and `suppress_tokens = []` before training.
- Pad labels with `-100`, so padding isn't learned as text.

Validate every `eval_every_steps`, and **keep the checkpoint with the lowest validation error**,
not the last one.

**W5 — evaluate.** Base against tuned, on test and regression, plus ten example clips side by
side. **Stop** if the test error didn't go down, or if the regression error rose by more than
about one point.

**W6 — merge.** See "Merging". The output is `runs/<job>/merged/`.

**W7 — convert.** Write a config for the existing converter from the job (`model_id` = the merged
folder, `base_model` = the base model folder, `language`, `engine`, `test.wav`), and run its
seven steps unchanged. `engine.languages` is `[language]` unless the job says otherwise, and
`engine.varieties` is `[variety]` when the job has one. Its step 2 accepts local folders already,
and its step 3 already proves the conversion is exact.

**W8 — score what volis will run.** W5 scores the merged model in full precision. Step 5 of the
converter checks the int8 files on one clip. Nothing yet measures the int8 model on the test
set, and that is the model volis runs. Transcribe the test set (or 300 clips of it, if larger)
with the converted int8 folder, the way volis does (`transcribe_like_cnverc` in the converter's
`common.py`), and compare its WER with W5's tuned WER. **Stop** if it is more than 1.5 points
worse: int8 lost too much, and the job should be re-converted with `use_fp32: true`.

---

## Stages of a pairs job

**P1 — collect.** Read the transcripts from `sources`. Drop lines under `min_words` (fragments
like "para que sea" teach nothing), and remove duplicates. Print counts for train and test.

**P2 — choose the teacher.** If `machine.yaml` names more than one `teacher_gguf`, translate the
FLORES+ `devtest` sentences in the dialect's language into English with each one, and score
chrF against FLORES+'s own English. Use the higher-scoring teacher and record the choice. With
only one teacher named, still score it and print the number.

FLORES+ has no Mexican Spanish: its Spanish (`spa_Latn`) is a neutral written standard, so for
`es-MX` this measures general Spanish ability, which is the best available proxy. For Iraqi
Arabic, FLORES+ has `acm_Arab` (Mesopotamian Arabic), which is much closer than Standard Arabic
(`arb_Arab`): use it for the teacher check and the general pairs. The job file names the FLORES+
language code (`flores_code:`), so no script assumes one.

**P3 — translate.** The teacher translates every transcript into English.

- **Dialect → English pairs:** the transcript as source, the teacher's English as target.
- **English → dialect pairs:** the same pairs reversed, so the dialect side is always **real
  speech from real speakers**, never written by a model. Clean that side first: the teacher
  removes false starts and repeated words (*y y*, *que que*) and **changes nothing else**. Use
  the same word-comparison check as punctuation restoring, allowing only removals.
- **Never have a model write dialect sentences from nothing.** Models asked to write a dialect
  drift towards the standard language or a neighbouring dialect.
- **General pairs:** FLORES+ `dev` (never `devtest`) in both directions, for the Qwen job's mix.

Write `out/train.jsonl`, `out/test.jsonl` and `out/general.jsonl` in the JSONL format above.

**P4 — the fluent-speaker check.** Write `out/review.csv`: 50 random pairs from the test set,
with an empty `ok` column. **Stop and ask the user** to have a fluent speaker of the dialect
mark each row `y` or `n`. When the stage runs again, it reads the file. At 90% `y` or more, the
job passes. Below that, stop: the teacher isn't good enough, and everything the Qwen job learns
would be capped by its mistakes. Say so plainly, with the rows marked `n`.

---

## Stages of a Qwen job

**Q1 — check.** Base model files present, and `config.json` says a Qwen3 architecture. The data
opens, and every line has the four fields and known language tags.

**Q2 — prepare.** Build the exact text volis sends for each pair (see "Training Qwen"). Drop
pairs where either side is empty or the target is more than three times the source's length.
Mix in `general` pairs at about a fifth of the total, if given. Refuse to continue if any test
sentence appears in training.

**Q3 — baseline.** Translate the test set with the base model, using volis's exact prompt and
volis's decoding (greedy: always the most likely next word). Score by chrF, per direction.

**Q4 — train.** LoRA on all linear layers. **Only the answer is trained:** everything up to and
including the empty `<think>\n\n</think>\n\n` block gets label `-100`, because volis sends
all of it. The translation **and the closing `<|im_end|>`** are trained. Without `<|im_end|>`
the model never learns to stop, and volis's length guard throws its output away.

**Q5 — evaluate.** chrF per direction, base against tuned. Print 20 examples side by side.
Stop if either direction got worse.

**Q6 — merge.** See "Merging".

**Q7 — convert to GGUF.** See "Qwen → GGUF".

**Q8 — install and check in volis.** Score both models the way volis will actually run them:

1. **Before swapping,** translate the test set through volis with its **current** `.gguf`
   (`volis --translate`, see "Two small volis additions"). Score chrF per direction. This is the
   fair baseline: the same program, compression and decoding. Q3's score isn't, because it came
   from the full-precision model.
2. Move volis's current `.gguf` into `models/mt-parked/` (create it), and copy the new one into
   `models/mt/`, because volis refuses to run with two. Run `volis --report` and check it lists
   the new file.
3. Translate the test set through volis again, with the new file.
4. **Pass** if the new model beats step 1 in both directions, and lost no more than 3 chrF points
   against Q5 (compression costs a little; more than that means the conversion went wrong).
   Count lines volis refused (empty output) for both models, and **stop** if the new model is
   refused more often.
5. Print both scores, and how to swap back.

---

## Training Qwen on volis's exact prompt

A model trained on one prompt layout and run with another loses most of what it learned. So
the training text is **the exact string volis builds**, from `prompt_for` in
`src/translate.rs`:

```
<|im_start|>system
{system prompt for this source and target}<|im_end|>
<|im_start|>user
{source text}<|im_end|>
<|im_start|>assistant
<think>

</think>

{target text}<|im_end|>
```

The system prompt changes with the source and target (it names the variety, "Mexican
Spanish", and adds a sentence when the target has a dialect). **Don't copy it into Python by
hand.** It would drift the next time `translate.rs` changes. Get it from volis (see "Two small
volis additions").

---

## Merging

The same for both kinds of model:

1. Load the **base model** in full precision (not 8-bit or 4-bit: merging into compressed
   weights loses accuracy).
2. Load the chosen LoRA onto it with `PeftModel.from_pretrained`.
3. Call `merge_and_unload()`. This adds the adjustments into the weights and returns a plain
   model with no LoRA attached.
4. Save with `save_pretrained(..., safe_serialization=True, max_shard_size="20GB")`. The large
   shard size keeps it one `model.safetensors`, which the Whisper converter needs. Save the
   processor or tokenizer and `generation_config.json` beside it.
5. **Check the merge worked.** Reload the merged model from disk **on its own, with no LoRA
   attached**, and re-run the test set with the same precision and decoding the evaluate stage
   used. Its score must be within **0.3 points** (WER or chrF) of the evaluate stage's tuned
   score. If it isn't, stop. A bad merge still produces a model that loads and runs, just not
   the trained one.

Keep the LoRA folder too. It's small, and it's the thing to reuse if a later volis loads
adapters at runtime instead of merged models.

---

## Qwen → GGUF

volis links `llama-cpp-2` 0.1.156, which builds a particular llama.cpp commit into volis. Write
the GGUF with **that commit**, not the newest llama.cpp, so the file is written by the same
llama.cpp that will read it.

**Don't use the copy in the Cargo registry.** It is trimmed for building the library: it has
`convert_hf_to_gguf.py` but not the `gguf-py` package the script imports, and its `tools/` has no
`quantize`. Instead:

- **Find the commit.** `llama-cpp-sys-2` gets llama.cpp as a git submodule of the
  [`llama-cpp-rs`](https://github.com/utilityai/llama-cpp-rs) repository. Look up the submodule
  commit at the tag for 0.1.156 (for example with
  `git ls-tree <tag> llama-cpp-sys-2/llama.cpp` in a clone of that repository). If the tag
  can't be found, stop and say so; don't guess a nearby commit.
- **Clone llama.cpp at that commit** into `qwen-to-gguf/llama.cpp/` (not committed; add it to
  `.gitignore`). Record the commit in `runs/<job>/llama.cpp-commit.txt`.
- **Convert:** its `convert_hf_to_gguf.py`, with its own `gguf-py`, turns the merged model into a
  full-precision GGUF. Run it in the training environment, since it needs `transformers` and
  `torch`.
- **Quantize:** build `llama-quantize` from the same clone with CMake (the MSVC tools that build
  volis are enough, and no CUDA is needed), and quantize to `Q4_K_M`, the level volis's current
  model uses.

The teacher's `llama-server` is separate and needn't match; see "The teacher model".

Keep all of this in `qwen-to-gguf/steps/`, with the same step style as `whisper-to-onnx/`
(`common.py`, `STOP:` messages, `runs/<job>/`). Record the llama.cpp commit used in the run
folder.

**Check:** the `.gguf` file exists and is roughly the size of volis's current one (about 1.1 GB
for 1.7B at Q4_K_M), and Q8 passes.

---

## Two small volis additions

These go in the **volis** repo, as their own small change, before Q2 needs them. Its
`CLAUDE.md` rules apply: no new settings, and `cargo fmt` and `clippy` clean. The instructions
for that session are in a separate file, `volis-additions.md`, next to this one. Hand it to a
Claude Code session opened in the volis repo. This session doesn't edit volis.

Both print **only** their result to stdout. volis's logging also writes to stdout today, so for
these two commands it must go to stderr instead. Otherwise log lines land between the
translations and line N of the output stops matching line N of the input.

1. `volis --print-prompt <source> <target>` prints the exact text `prompt_for` would send, with
   `{text}` left as a placeholder. Q2 builds every training example from it.
2. `volis --translate <source> <target> < file.txt` translates one line per input line, with
   the model in `models/mt/`, and prints one translation per line. That includes the three
   refusal guards: a refused line prints as empty and is reported on stderr. Q8 uses it to score
   the installed model the way volis will actually run it.

Neither touches the network, so both fit volis's rules.

---

## Rules for the whole app

1. Every setting comes from the job file or `machine.yaml`. Nothing about a particular model or
   dataset is written into a script.
2. Everything a job produces goes in `runs/<job>/`. Nothing is overwritten without `-Force`.
3. Every file is opened with `encoding="utf-8"`. On Windows the default corrupts Arabic without
   any error. The same goes for pipes: when a stage feeds text to `volis --translate`, it does so
   from Python with UTF-8 bytes, never through a PowerShell pipe (Windows PowerShell 5.1 re-encodes
   piped text and breaks accents and Arabic).
4. No check is skipped to keep going.
5. Update the README: the new quick start (drop in, write a job, `.\train.ps1`), the data
   formats, and one "things that went wrong" entry per problem found while building this, in the
   style the README already uses.

## Running training on another machine

Training may move to a bigger GPU later: a rented cloud machine, or a Linux box. Those are
almost always Linux. So **the training stages (W1–W6, P1–P4, Q1–Q6) must also run inside a
Linux container.** The teacher runs there too, so the container includes a CUDA build of
`llama-server`. Conversion, scoring and installation (W7, W8, Q7, Q8) stay on the Windows machine
that has volis:
the converter's checks run volis's own engine, and volis is a Windows program there.

The dividing line is the merged model. Training produces `runs/<job>/merged/`. Conversion reads
it. That folder, plus the job file, is everything that crosses between machines.

Build the training code so the container is easy to add:

- Training stages are plain Python run from the command line with the job file as the argument.
  No PowerShell inside them. `train.ps1` only calls them.
- No Windows-only paths or tools in `training/`. Use `pathlib` everywhere.
- The inputs, the job file and `runs/` are passed in as folders. Nothing is copied into the
  container image, so a new dataset never means rebuilding the image.

Then add, as its own step in the build order:

- A `training/Dockerfile`, based on an official NVIDIA CUDA 12.8 (or newer) runtime image, which
  installs the training environment from `training/uv.lock`, the same versions as on Windows.
- **`docs/CONTAINER.md`**, written for the user, not for a developer. It covers:
  1. What's inside the container and what isn't: training yes; conversion and volis no.
  2. Building the image, and saving it to a single file (`docker save`) or pushing it to a
     registry.
  3. Getting it, the job file and the input data onto the other machine.
  4. Running a job with the GPU visible and the three folders attached. One command, with a
     real example.
  5. Checking the GPU is really being used, not silently falling back to the CPU.
  6. Bringing `runs/<job>/merged/` back, and running the Windows side
     (`.\train.ps1 <job> -From W7`, or `-From Q7`).
  7. What to do when it stops: the same `STOP:` messages as on Windows, and how to resume.

Test the container at least once on the Windows machine, through Docker Desktop with its WSL2
GPU support, before calling it done. Run the `whisper-small` rehearsal job through it and
check the merged model converts on the Windows side.

## What only the user can do

Claude Code stops and asks at each of these:

| When | What | Why it can't be automated |
|---|---|---|
| Before item 1 | Decide whether to commit the existing uncommitted work | It's the user's work |
| Item 2 | Accept FLORES+'s terms on Hugging Face and run `uv run huggingface-cli login` | The dataset is gated behind an account |
| Item 2 | Choose and download the teacher GGUF (or approve the download), and set `teacher_gguf` in `machine.yaml` | It's tens of GB, and depends on the machine |
| Item 2 | Install NVIDIA's CUDA toolkit, only if no prebuilt CUDA `llama-server` works | Installs system software |
| Item 3 | Listen to the ten before-and-after clips W5 prints, and try the model in volis with **Compare recognizers** | Numbers can improve while speech gets worse |
| Item 4 | Open a Claude Code session in the volis repo with `volis-additions.md` | This session doesn't edit volis |
| Item 5 | Get a fluent Mexican Spanish speaker to mark `review.csv` | Only a speaker can judge the teacher |
| Item 6 | Read 20 translations after Q5, and use the model in volis after Q8 | A score can rise while output gets stiffer |
| Item 7 | Install Docker Desktop, with WSL2 GPU support turned on | Installs system software |
| Any real run | Run it on the 5090 desktop | The laptop's GPU is too small for turbo |

## Build order

Each stage passes its check before the next is started.

1. The volis-name and `varieties` fixes to the existing converter: step 7, step 6, their
   messages, `machine.example.yaml` and the README. Re-run `es-turbo-adriszmar` through
   `convert.ps1` to prove nothing broke.
2. The training environment and its setup check, `fetch.ps1`, the disk space check, and the
   teacher runner (with a prebuilt CUDA `llama-server`). Check: the teacher punctuates ten
   CIEMPIESS lines, with no words changed, on the GPU.
3. The Whisper job, W1 to W8. First a fast rehearsal with `whisper-small` and about one hour of
   CIEMPIESS, to prove every stage end to end. Then turbo on the full data, on the desktop.
4. The two volis additions, in a separate session in the volis repo (`volis-additions.md`).
5. The pairs job, P1 to P4.
6. The Qwen job, Q1 to Q8.
7. The training container and `docs/CONTAINER.md`.
