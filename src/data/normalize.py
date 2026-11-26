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
_REPEAT = re.compile(r"([^\W\d_xX])\1{2,}")  # masked numbers like 01XXXXXXXXX are kept intact  # 3+ of the same LETTER -> 2 ("hellooooo" -> "helloo"); digits kept


LEVEL = 1  # set from cfg data.normalize by load_clean; 2 = + leet mapping and spaced-letter rejoin

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_SPACED = re.compile(r"\b(?:[A-Za-z][\s.\-*_]){2,}[A-Za-z]\b")
_LATIN_TOKEN = re.compile(r"\S+")


def _leet_token(m):
    w = m.group(0)
    letters = sum(c.isalpha() and c.isascii() for c in w)
    odd = sum(c in "013457@$" for c in w)
    if letters >= 3 and 0 < odd <= 2 and not re.search(r"01[xX\d]|\d{3,}|[xX]{3,}", w):
        return w.translate(_LEET)
    return w


def _homoglyph_token(m):
    w = m.group(0)
    return w if re.search(r"[\u0980-\u09FF]", w) else w.translate(_HOMOGLYPHS)  # never remap Bengali-script tokens


def robust(text: str, level=None) -> str:
    level = LEVEL if level is None else level
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")  # zero-width, bidi marks, soft hyphen, BOM...
    text = _LATIN_TOKEN.sub(_homoglyph_token, text)
    if level >= 2:
        text = _SPACED.sub(lambda m: re.sub(r"[\s.\-*_]", "", m.group(0)), text)
        text = _LATIN_TOKEN.sub(_leet_token, text)
    return " ".join(_REPEAT.sub(r"\1\1", text).split())
