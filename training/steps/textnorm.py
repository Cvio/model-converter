"""One text-cleaning function and the error rates, shared by every scoring stage.

No third-party imports, on purpose: the training stages (W3, W5, W6) run in
the training environment and W8 runs in the converter's, and all of them must
clean and count text identically, or their numbers can't be compared.

Cleaning is chosen by language (train-and-convert-app.md, W3):
- every language: Unicode NFC, lowercase, punctuation removed, spaces collapsed;
- Arabic also: diacritics (harakat, tatweel) removed and letter variants
  unified (alef forms to ا, alef maqsura to ي, taa marbuta to ه).
"""

import re
import unicodedata

ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")
ARABIC_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})


def normalize(text: str, language: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    if language.split("-")[0] == "ar":
        text = ARABIC_DIACRITICS.sub("", text).translate(ARABIC_LETTERS)
    # Punctuation of every script (Unicode category P*), and symbols like ¿ ¡.
    text = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text)
    return " ".join(text.split())


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def error_rates(references: list, hypotheses: list, language: str) -> dict:
    """Corpus WER and CER in percent, after normalize(), plus the counts."""
    word_errors = words = char_errors = chars = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = normalize(ref, language), normalize(hyp, language)
        word_errors += edit_distance(r.split(), h.split())
        words += len(r.split())
        char_errors += edit_distance(list(r.replace(" ", "")), list(h.replace(" ", "")))
        chars += len(r.replace(" ", ""))
    return {
        "wer": 100 * word_errors / max(words, 1),
        "cer": 100 * char_errors / max(chars, 1),
        "clips": len(references),
        "words": words,
    }
