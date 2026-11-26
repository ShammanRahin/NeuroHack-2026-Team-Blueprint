"""Adversarial-text normalisation for the hidden 'robustness' subset (lookalike letters, zero-width / format characters,
fullwidth forms, repeated characters, odd spacing). Pure per-string rules, identical for train and test, no test statistics.

Follows published defences: NFKC + zero-width / format-character removal + a confusables table, each repairing its own
kind of change (see docs/FEATURES.md for references)."""
import re
import unicodedata

# Cyrillic / Greek letters that look like Latin ones -> Latin
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h",
    "ԛ": "q", "ԝ": "w", "ӏ": "l", "ь": "b", "п": "n", "т": "t", "к": "k", "м": "m", "в": "b", "г": "r", "ı": "i",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "У": "Y",
    "І": "I", "Ј": "J", "Ѕ": "S", "Ԁ": "D", "Ԛ": "Q", "Ԝ": "W",
    "ο": "o", "Ο": "O", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ρ": "P",
    "Τ": "T", "Υ": "Y", "Χ": "X", "ν": "v", "α": "a", "ε": "e", "ι": "i", "κ": "k", "ρ": "p", "τ": "t", "υ": "u",
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
})
_REPEAT = re.compile(r"(.)\1{2,}")  # 3+ of the same char -> 2 ("hellooooo" -> "helloo")


def robust(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")  # zero-width, bidi marks, soft hyphen, BOM...
    text = text.translate(_HOMOGLYPHS)
    return " ".join(_REPEAT.sub(r"\1\1", text).split())
