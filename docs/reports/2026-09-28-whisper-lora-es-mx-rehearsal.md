# Report: training a Mexican Spanish LoRA for Whisper, and merging it into the base model

*The rehearsal run, 2026-09-28. Job: `jobs/es-mx-whisper-small-rehearsal.yaml`. Machine: laptop,
RTX 4070 (8 GB). Everything below was measured; the numbers come from
`runs/es-mx-whisper-small-rehearsal/stages/` and `scores/`.*

## In short

We taught Whisper (the speech recognizer volis uses) to hear Mexican Spanish better, by training
a small add-on (a **LoRA**) on recordings of Mexican speakers, then folding it into the model
(**merging**) and converting the result into the files volis runs. This first run was a
**rehearsal**: a small model and about an hour and a half of speech, meant to prove every step
works before the real run on the RTX 5090.

- **Every step works end to end**, from downloading the data to the finished model listed as `ok`
  by volis, in one command (`train.ps1`).
- **Mexican Spanish got clearly better:** word error rate on unseen Mexican speakers went from
  **16.5% to 12.7%** (character error rate from 8.5% to 5.2%).
- **Merging is exact:** the merged model scores exactly like the model with the LoRA attached
  (difference 0.00).
- **The conversion to volis's format costs little:** the compressed (int8) model scores 13.3% in
  volis's own engine, 0.6 points above full precision.
- **But ordinary Spanish got worse** (4.98% to 6.64%), and much of the "improvement" is the model
  copying how this dataset's transcribers wrote, not just hearing the dialect better. Both need
  fixing before a model like this goes into daily use. The real job already includes the first
  fix (mixing in ordinary speech).

## Words used here

- **WER (word error rate):** of every 100 words said, how many the model got wrong (a wrong,
  missing or extra word each count). Lower is better. **CER** is the same for letters.
- **LoRA:** a small file of adjustments trained on top of a model that stays locked. Here, 7.1
  million trainable numbers on top of whisper-small's 242 million.
- **Merging:** adding the LoRA's adjustments into the model's own weights, giving one ordinary
  model file with nothing attached.
- **int8:** a compressed form of the model (numbers stored in 8 bits), about a third of the size.
  volis runs its models this way.
- **Test data / regression data:** recordings the model never trained on. *Test* is Mexican
  Spanish (does it hear the dialect better?); *regression* is ordinary Latin American Spanish
  (did it get worse at everything else?).

## What was run

| | |
|---|---|
| Base model | `openai/whisper-small` (12 encoder and 12 decoder layers, 80 mel bins) |
| Training data | CIEMPIESS LIGHT (Mexican broadcast conversation, UNAM radio): 1,312 clips, 1.64 h, 8 speakers |
| Validation | 59 clips, 0.11 h, 1 speaker (held back from training, by speaker) |
| Test | CIEMPIESS TEST: 200 clips spread across all 20 speakers, 0.46 h |
| Regression | Google FLEURS `es_419` test: 200 clips, 0.68 h (read Latin American Spanish) |
| LoRA | rank 32, alpha 64, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `out_proj` in both encoder and decoder |
| Training | learning rate 1e-4, 20 warmup steps, 3 epochs = 246 steps, batch 8 × accumulation 2, bf16 autocast, gradient checkpointing |
| Decoding | greedy, language and task set (`es`, `transcribe`), so Whisper never guesses the language |
| Scoring | lowercase, punctuation and symbols removed, then WER and CER (`training/steps/textnorm.py`, shared by every stage) |

Two settings apply to this rehearsal only, and are marked in its job file:
- **Punctuation restoring off.** The laptop's teacher model (Qwen3-8B) changed words in 7.1% of
  transcripts, over the 5% limit. The real job keeps it on, with a 32B teacher.
- **Regression limit 2.0 points instead of 1.0**, so the rehearsal could continue past the
  evaluation stage and prove merging and conversion.

## Results

### Accuracy

| | Test WER (Mexican) | Test CER | Regression WER (ordinary) | Regression CER |
|---|---|---|---|---|
| Base whisper-small | 16.50 | 8.52 | 4.98 | 1.70 |
| Tuned (base + LoRA) | **12.69** | **5.19** | 6.64 | 2.32 |
| Change | **−3.81** | **−3.33** | +1.67 | +0.62 |

Per clip, on the 200 test clips: **103 improved, 33 got worse, 64 unchanged**.

### Training

Validation WER (the one held-back speaker), every 50 steps:

| Step | 0 | 50 | 100 | 150 | 200 | 246 |
|---|---|---|---|---|---|---|
| Training loss | – | 0.567 | 0.156 | 0.164 | 0.124 | 0.113 |
| Validation WER | 12.78 | **9.50** | 10.77 | 10.03 | 10.35 | 9.82 |

The best point came early, at step 50, and was the one kept; after it, loss kept falling while
validation stopped improving: the model was starting to memorise 8 speakers' 1.6 hours. Keeping
the best checkpoint rather than the last one is what protected the result. Training took 43
minutes; memory peaked at 7.6 of 8.6 GB with batch 8.

### Merging

| | Test WER |
|---|---|
| Base + LoRA attached (W5) | 12.687 |
| Merged model, reloaded from disk on its own (W6) | 12.687 |

Identical to three decimals. The merge was done in full precision (float32), saved as one
`model.safetensors` (967 MB) with the processor beside it.

### Conversion to volis's format

The merged model went through the existing converter unchanged:

| Check | Result |
|---|---|
| Converted checkpoint vs original, same input | largest logit difference 0 |
| ONNX export vs checkpoint, same input | largest logit difference 0.0000445 |
| int8 vs full precision, one test clip, volis's engine | 94% of characters, 80% of words agree; 375 MB (fp32 969 MB) |
| volis `--report` | listed as `ok`, with `varieties = ["es-MX"]` |
| **int8 on the 200 test clips, volis's engine (W8)** | **WER 13.27**, +0.58 against full precision |

## What we learned

### 1. Much of the gain is transcription style, not only dialect

Side by side, the tuned model often wins by writing the way CIEMPIESS's transcribers wrote:

| Reference (CIEMPIESS) | Base | Tuned |
|---|---|---|
| eso es como lo lo lo que tendríamos que cuestionarnos cada vez | Eso es como lo que tendríamos que cuestionarnos cada vez. | eso es como lo lo lo que tendríamos que cuestionarnos cada vez |
| quien te hace esos comentarios no y yo aquí lo que quisiera también agregar es que | quién te hace esos comentarios y yo aquí lo que quisiera también agregar es que | quien te hace esos comentarios no y yo aquí lo que quisiera también agregar es que |
| sí pus es lo que yo veo no lamentablemente es un tema que que pervive | Si, pues es lo que yo, lamentablemente, es un tema que pervive. | si pues es lo que yo no lamentablemente es un tema que que pervive |

CIEMPIESS transcribes word for word, keeping repeated words ("lo lo lo", "que que") and the
filler "no". Base Whisper tidies those away, which counts as errors against this reference;
the tuned model learned to keep them. That is a real change in behaviour, but not the same thing
as understanding the dialect better, and for captions in volis the tidy version may even be
preferable. **The 3.8-point improvement overstates how much better it hears Mexican Spanish.**
Scoring on a test set transcribed in a tidier style (or scoring with fillers and repeats removed
from both sides) would separate the two.

### 2. Training on unpunctuated text teaches the model to stop punctuating

The rehearsal trained with punctuation restoring off, on CIEMPIESS's all-lowercase,
punctuation-free transcripts. The tuned model now writes the same way: no capitals, no commas,
no question marks (last column above). Scoring removes punctuation, so the numbers don't show
it, but volis would caption "quien te hace esos comentarios no" instead of a sentence. This is
exactly why the real job restores punctuation first, and why a strong teacher matters: a
question that loses its "¿…?" is translated as a statement (seen in volis: "¿Quieren…?"
translated as "they want…").

### 3. A little dialect data hurts ordinary speech

Ordinary Spanish got 1.67 WER points worse. The added errors, by kind:

- **Numbers written as words.** CIEMPIESS spells numbers out, and the model copied it: FLEURS's
  "6 30" became "seis y media", "7 30 h" became "siete horas".
- **Accent marks:** "solo" became "sólo" (4 times), "París" "Paris". Scoring counts these as
  wrong words; ignoring accents only brings the rise down to 1.55 points, so they're a small part.
- **Genuine slips:** "Mendoza" as "Mendosa", "en irse" as "enirse", extra "de" and "a".

With 1.6 hours from 8 speakers, the model over-learned. The real job mixes ordinary speech into
training (`data.mix`: FLEURS `es_419` train, a fifth of the hours) and keeps the 1.0-point limit.
Whether that is enough is the first thing to check on the 5090.

### 4. Three things that fail silently, and how the pipeline catches them

- **The encoder's LoRA can learn nothing without any error.** With gradient checkpointing,
  `enable_input_require_grads()` only reaches the decoder. The encoder's input comes from a
  frozen convolution, so nothing entering its checkpointed layers needs a gradient, and its LoRA
  would stay at zero while training looks normal. The encoder is the part that hears the audio,
  including how the dialect sounds. W4 hooks
  the first convolution and checks, before the first step, that every one of the 96 encoder LoRA
  weights receives a gradient.
- **Checking that weights moved after step 1 is a false alarm.** The learning rate warms up from
  zero, so the first optimizer step changes nothing anywhere. The check has to look at gradients.
- **Separate datasets reuse speaker IDs.** CIEMPIESS LIGHT and TEST both number speakers F_01 to
  M_10, and they are different people (the dataset card says so). A bare-ID overlap check reports
  all 20 test speakers as leaking into training. W2 labels each ID with its dataset.

### 5. Restoring punctuation: the teacher needs to be told what "don't change a word" means

The teacher model is asked to add capitals and punctuation and change no word; any line whose
words changed is thrown away. What it changed, over the rehearsal's 1,371 lines:

| Teacher prompt and check | Lines thrown away |
|---|---|
| "Change no word", accents counted | 15.4% |
| Same, accents ignored in the check | 7.4% |
| Prompt names repeats, false starts, hesitations ("e", "este") and misspellings; accents ignored | 7.1% (Qwen3-8B) |

Most rejections at first were the written accent on question words ("que" to "qué", "como" to
"cómo"), which goes with adding ¿?, so the check now ignores accents (but not ñ). What remained
were real rewrites: dropping the hesitation "e", correcting transcripts' misspellings ("echo" to
"hecho"), removing a repeated word. An 8B teacher can't get under 5% here; the real job uses a
32B one.

## What this means for the real run

1. **Run it on the 5090 with punctuation restoring on** and a 32B-class teacher. Check the
   thrown-away share is under 5%, and read the ten before/after examples W2 prints.
2. **Watch the regression number first.** With 18 hours instead of 1.6, more speakers, and a fifth
   of ordinary speech mixed in, it should stay within 1 point. If it doesn't, raise `mix.share`.
3. **Judge the test improvement against lesson 1.** Some of it will be transcription style.
   Listening in volis (Compare recognizers) on real Mexican speakers is the real test.
4. **Expect int8 to cost around half a point**, as here; W8 stops if it costs more than 1.5.

## Where things are

- Stage results: `runs/es-mx-whisper-small-rehearsal/stages/W1.json` … `W8.json`
- Transcripts behind every score: `runs/es-mx-whisper-small-rehearsal/scores/`
- The pipeline: `training/steps/whisper/`, run by `train.ps1`
- The plan it follows: `train-and-convert-app.md`

---

## Part 2: translation pairs for the translator (the pairs job rehearsal)

*Job: `jobs/es-mx-pairs-rehearsal.yaml`. Same laptop, teacher Qwen3-8B. Results from
`runs/es-mx-pairs-rehearsal/`.*

### In short

The translator (Qwen) learns a dialect from **translation pairs**: a Mexican Spanish sentence and
its English, used both ways. The pairs job makes them from the Whisper rehearsal's Mexican
transcripts, so the Mexican side is always something a real person said, and a larger model
(the **teacher**) writes the English. Every stage worked, and the last one did its job: a check
of 50 translations found the rehearsal teacher **not good enough (48% correct, 90% needed)**, so
the job stopped before the translator could learn from them.

### What was run

| | |
|---|---|
| Sentences | 300 training and 100 test transcripts (5 words or more; test speakers kept apart) |
| Teacher | Qwen3-8B (Q4_K_M) on the laptop GPU, greedy decoding, thinking off |
| Teacher check (P2) | 150 FLORES+ devtest sentences, Spanish to English: **chrF 59.5** |
| Pairs made (P3) | 600 training (300 each way), 200 test, 1,994 general (FLORES+ dev, Spanish ↔ English) |
| Tidying | the teacher removes false starts, repeats and "e" hesitations from the Mexican side, and nothing else; it did more than that on 25 of 400 lines, which kept their original wording |
| Review (P4) | 50 random test pairs, marked by Opus 5 standing in for a fluent speaker: **24 y, 26 n (48%)** |

The reviewer here was a model, as a quick rehearsal check. The step exists because only a native
speaker can reliably judge whether a translation of Mexican speech is right and natural; for the
real run, a person should do it.

### Why half the translations failed

| Pattern | Example (Mexican transcript → teacher's English) |
|---|---|
| **Statements read as questions** | "porque no sabían…" → "why didn't they know…" |
| **Hesitation "e" translated as a word or name** | "…fue olvidada dice e en…" → "…says E, in in a lot" |
| **Spelled-out letters not recognised** | "ve i hache" (VIH, Spanish for HIV) → "V and H" |
| **Colloquial spellings misread** | *entos*, *entoces* (entonces), *libertá*, *díctun* |
| **Who's who confused** | "cuando yo la conocí te tenía quince" → "when I met her **you** were fifteen" |
| **Long rambling lines garbled** | "que la vistiera de mujer de hombre perdón" → "dress like a man's woman" |

Two causes explain most of it, both rehearsal shortcuts:

1. **No punctuation.** The rehearsal skipped punctuation restoring, so the teacher got
   all-lowercase text with no full stops or question marks. "porque" (because) and "por qué"
   (why) look alike without the accent and the ¿?, and one thought runs into the next.
2. **A small teacher.** An 8B model stumbles on CIEMPIESS's writing conventions: letters spelled
   out ("ve i hache", "be", "ce"), colloquial spellings, and "e" for a hesitation.

A stronger translation score on FLORES+ (59.5) didn't predict this: FLORES+ is clean, written,
punctuated text. **The teacher check needs spoken, dialect text too**, which is what the person
review provides.

### What the data can teach

CIEMPIESS is university radio talk, mostly law, economics and social issues. It is real Mexican
speech, but everyday dialect words (*carro*, *chamba*, *¿qué onda?*) are rare in it. Pairs made
from it teach Mexican *spoken register* more than everyday *vocabulary*. For a translator that
handles everyday Mexican conversation, more conversational speech datasets are needed; the jobs
take any speech dataset, so adding one is a job-file change.

### What this means for the real run

1. **Punctuation restoring on** (the real Whisper job has it), with the 32B teacher. The same
   review should then be run again.
2. **Tell the teacher CIEMPIESS's conventions** in its translation prompt: spelled-out letters
   are acronyms ("ve i hache" is VIH/HIV), "e" alone is a hesitation, and *entos*/*pus*/*-tá* are
   colloquial spellings. This is a prompt change in `teacher/teacher.py`.
3. **A person reviews the 50 pairs.** Only after 90% does the translator train on them.
