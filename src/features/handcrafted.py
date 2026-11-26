"""MEMBER B. Hand-written, interpretable scam-tactic features (no learning, so nothing is fit on test text).

Each lexicon mirrors a rule from the competition's labelling guide (Data tab): credential requests (class 1 / T2 / action 1),
prizes (class 2 / T4), fake jobs and investments (class 3 / T6), parcels (class 4), promotions (class 5), urgency and
pressure (T5), claimed organisations (T1), payment requests (T3 / action 2), links (action 3), install/join (action 4),
contact (action 5), plus the hard-legitimate cues (OTP delivery with a do-not-share warning, transaction notices,
messages that warn ABOUT a scam). English, Banglish and Bangla script. Text arrives already de-obfuscated by
src/data/normalize.py; long keywords are also matched with spaces and punctuation removed ("p a s s w o r d").

Vectorised over pandas string ops: ~1 minute for 500,000 rows. The scale (MaxAbsScaler) is fit on train rows only."""
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from src.common.registry import feature

LEX = {
    "cred": ["otp", "pin", "password", "passcode", "cvv", "nid", "card number", "card no", "verification code",
             "6 digit", "6-digit", "secret code", "code ta", "code ti", "pin ta", "pin number", "login", "account number",
             "ওটিপি", "পিন", "পাসওয়ার্ড", "এনআইডি", "কোড", "গোপন"],
    "share_req": ["share", "send", "bolen", "bolun", "bolben", "din", "diben", "dao", "pathan", "pathiye", "provide",
                  "tell me", "give", "enter", "submit", "verify", "দিন", "পাঠান", "বলুন", "শেয়ার", "জানান"],
    "warning": ["do not share", "don't share", "dont share", "never share", "kauke bolben na", "kauke diben na",
                "share korben na", "karo sathe", "will never ask", "never ask", "keu jodi", "কাউকে বলবেন না",
                "কাউকে দিবেন না", "শেয়ার করবেন না", "eta kauke"],
    "mistake": ["bhul", "vul", "mistake", "wrong number", "by mistake", "ferot", "return the money", "send it back",
                "ভুল", "ফেরত"],
    "prize": ["won", "win ", "winner", "jitechen", "jitesen", "jiten", "prize", "lottery", "lucky draw", "cashback",
              "cash back", "gift", "reward", "bonus", "congratulations", "congrats", "ovinondon", "spin", "selected",
              "অভিনন্দন", "জিতেছেন", "পুরস্কার", "উপহার", "ক্যাশব্যাক", "বোনাস", "লটারি"],
    "job": ["job", "salary", "income", "earn", "part-time", "part time", "work from home", "ghore bose", "ghor bose",
            "task", "commission", "invest", "double", "profit", "crypto", "bitcoin", "usdt", "trading", "recruit",
            "chakri", "per day", "daily income", "like korle", "আয়", "চাকরি", "বিনিয়োগ", "লাভ", "কমিশন", "ঘরে বসে"],
    "parcel": ["parcel", "courier", "delivery", "deliver", "package", "customs", "shipment", "tracking", "consignment",
               "rider", "hub", "পার্সেল", "কুরিয়ার", "ডেলিভারি", "কাস্টমস", "প্যাকেজ"],
    "promo": ["offer", "discount", "sale", "% off", "voucher", "coupon", "promo", "buy 1", "free delivery", "admission",
              "order now", "flat ", "upto", "up to", "অফার", "ছাড়", "ডিসকাউন্ট"],
    "urgency": ["urgent", "immediately", "now", "today only", "within", "expire", "last chance", "block", "suspend",
                "band hobe", "bondho", "off hoye", "24 ghonta", "ghontar", "1 hour", "jaldi", "hurry", "asap", "deadline",
                "otherwise", "na hole", "তাড়াতাড়ি", "জরুরি", "বন্ধ", "এখনই", "মেয়াদ"],
    "pay_req": ["pay", "fee", "charge", "deposit", "send money", "transfer", "processing", "security money",
                "registration", "duty", "advance", "taka pathan", "taka din", "টাকা পাঠান", "ফি", "চার্জ", "পরিশোধ"],
    "click": ["click", "tap", "visit", "link", "open", "lingk", "লিংক", "লিঙ্ক", "ক্লিক"],
    "install": ["install", "download", "apk", "app ", "join", "telegram", "whatsapp group", "group e", "subscribe",
                "ইনস্টল", "জয়েন", "গ্রুপ"],
    "contact": ["call", "contact", "reply", "whatsapp", "imo", "message", "inbox", "dial", "যোগাযোগ", "কল"],
    "org": ["bkash", "nagad", "rocket", "upay", "dbbl", "bank", "brac", "gp ", "grameenphone", "robi", "banglalink",
            "airtel", "teletalk", "daraz", "pathao", "foodpanda", "sundarban", "paribahan", "redx", "steadfast",
            "paperfly", "ecourier", "police", "govt", "government", "nbr", "bdjobs", "helpline", "customer care",
            "support", "agent", "officer", "team", "department", "বিকাশ", "নগদ", "ব্যাংক", "সরকার"],
    "legit_txn": ["successful", "trxid", "txnid", "trx id", "balance", "avl bal", "debited", "credited", "received",
                  "receipt", "bill", "due", "your order", "out for delivery", "has been shipped", "reminder",
                  "সফল", "ব্যালেন্স", "জমা হয়েছে"],
    "scam_talk": ["scam", "fraud", "protarona", "prottarona", "beware", "careful", "sabdhan", "fake", "dont trust",
                  "সাবধান", "প্রতারণা", "প্রতারক", "ভুয়া"],
}
_DESPACED_MIN = 5  # long keywords are also searched with spaces/punctuation removed (robust to "p a s s w o r d")

_PATTERNS = {
    "url": r"hxxps?|https?://|www\.|\[\.\]|\.(?:top|xyz|click|link|info|site|online|icu|live|shop|club|vip)\b",
    "apk": r"\.apk\b|\bapk\b",
    "shortcode": r"\*\d{2,4}(?:\*[\d*]+)?#",
    "phone": r"01[x\d]{9}|\+?880|01xx",
    "amount": r"(?:tk|taka|bdt|৳|টাকা)\s?[\d,০-৯.]+|[\d,০-৯.]+\s?(?:tk|taka|bdt|৳|টাকা|lakh|lac|crore)",
    "percent": r"\d+\s?%",
    "otp_digits": r"\b\d{4,8}\b",
    "masked": r"x{3,}\d*",
}


def _contains_any(s, words):
    return s.str.contains("|".join(re.escape(w) for w in words), regex=True)


def _count_any(s, words):
    return s.str.count("|".join(re.escape(w) for w in words))


def _gibberish_frac(s):
    """Share of Latin tokens that look like keyboard noise (e.g. 'wkrjfejejrn'): no vowel or 4+ consonants in a row."""
    def frac(t):
        toks = re.findall(r"[a-z]{5,}", t)
        if not toks:
            return 0.0
        bad = sum(1 for w in toks if not re.search(r"[aeiouy]", w) or re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", w))
        return bad / len(toks)
    return s.map(frac)


def compute(texts: pd.Series) -> np.ndarray:
    low = texts.fillna("").str.lower()
    despaced = low.str.replace(r"[\s\W_]+", "", regex=True)
    cols = {}
    for name, words in LEX.items():
        hit = _contains_any(low, words)
        long_words = [w.replace(" ", "") for w in words if len(w.replace(" ", "")) >= _DESPACED_MIN]
        if long_words:
            hit = hit | _contains_any(despaced, long_words)
        cols[f"{name}_any"] = hit.astype(np.float32)
        cols[f"{name}_n"] = np.log1p(_count_any(low, words)).astype(np.float32)
    for name, pat in _PATTERNS.items():
        cols[f"{name}_n"] = np.log1p(low.str.count(pat)).astype(np.float32)

    # conversation structure: decisive request may come late, after trust-building chatter
    turns = texts.fillna("").str.split(r"\|\|\|")
    n_turns = turns.map(len)
    cols["n_turns"] = np.log1p(n_turns).astype(np.float32)
    cols["is_conv"] = (n_turns > 1).astype(np.float32)
    first = turns.map(lambda t: t[0].lower())
    later = turns.map(lambda t: " ".join(t[1:]).lower())
    for name in ("cred", "pay_req", "click", "install", "prize", "job", "urgency"):
        cols[f"{name}_later_only"] = (_contains_any(later, LEX[name]) & ~_contains_any(first, LEX[name])).astype(np.float32)
    cols["speaker_markers"] = np.log1p(texts.fillna("").str.count(r"(?:^|\|\|\|)\s*(?:\[[^\]]{1,20}\]|[A-Za-z]{1,10}):")).astype(np.float32)

    # surface statistics (formal email / call transcript / group chat registers; noise-only records)
    raw = texts.fillna("")
    n_chars = raw.str.len().clip(lower=1)
    cols["len_log"] = np.log1p(n_chars).astype(np.float32)
    cols["n_tokens_log"] = np.log1p(raw.str.split().map(len)).astype(np.float32)
    cols["digit_ratio"] = (raw.str.count(r"\d") / n_chars).astype(np.float32)
    cols["upper_ratio"] = (raw.str.count(r"[A-Z]") / n_chars).astype(np.float32)
    cols["bangla_ratio"] = (raw.str.count(r"[ঀ-৿]") / n_chars).astype(np.float32)
    cols["latin_ratio"] = (raw.str.count(r"[A-Za-z]") / n_chars).astype(np.float32)
    cols["punct_ratio"] = (raw.str.count(r"[^\w\s]") / n_chars).astype(np.float32)
    cols["emoji_n"] = np.log1p(raw.str.count(r"[\U0001F300-\U0001FAFF☀-➿]")).astype(np.float32)
    cols["exclaim_n"] = np.log1p(raw.str.count("!")).astype(np.float32)
    cols["gibberish_frac"] = _gibberish_frac(low).astype(np.float32)
    any_lex = np.zeros(len(raw), dtype=bool)
    for name in ("cred", "prize", "job", "parcel", "promo", "pay_req", "org", "legit_txn"):
        any_lex |= cols[f"{name}_any"].to_numpy() > 0
    cols["no_signal"] = (~any_lex).astype(np.float32)
    # hard-legitimate interactions from the labelling guide
    cols["otp_with_warning"] = (cols["cred_any"] * cols["warning_any"]).astype(np.float32)
    cols["cred_request"] = (cols["cred_any"] * cols["share_req_any"] * (1 - cols["warning_any"])).astype(np.float32)
    cols["prize_and_cred"] = (cols["prize_any"] * cols["cred_any"]).astype(np.float32)
    cols["prize_and_pay"] = (cols["prize_any"] * cols["pay_req_any"]).astype(np.float32)
    cols["parcel_and_pay_or_link"] = (cols["parcel_any"] * np.maximum(cols["pay_req_any"], cols["url_n"] > 0)).astype(np.float32)
    return pd.DataFrame(cols).to_numpy(dtype=np.float32)


FEATURE_NAMES = None  # filled lazily for inspection


@feature("handcrafted")
def handcrafted(bundle, weight=1.0):
    scaler = MaxAbsScaler().fit(compute(bundle.train[bundle.text_col]))  # train rows only
    return lambda df: (weight * scaler.transform(compute(df[bundle.text_col]))).astype(np.float32)


# ------------------------------------------------------------------ noise-only router + raw obfuscation features
_VOCAB = {w.strip() for words in LEX.values() for w in words for w in w.split()} | set(
    "the a an to you your is are for of and in on with this that it be have has will can please call sms ok hi hello "
    "thanks dear sir madam bhai apu ami apni tumi amar apnar ki na hobe ache ase kore korun din den taka tk "
    "valo bhalo kemon achen ekhon pore abar".split())


def noise_only(text):
    """True only when several signals agree that a record carries no meaningful content (Data tab: such records
    are label 0 / NONE / 0). Conservative on purpose: Banglish must not be swallowed."""
    t = str(text).lower()
    if re.search(r"[ঀ-৿]", t) or re.search(r"\d|hxxp|http|www|\[\.\]|৳", t):
        return False
    toks = re.findall(r"[a-z]+", t)
    if not toks:
        return bool(t.strip())                       # only symbols / emoji
    if any(w in _VOCAB for w in toks):
        return False
    gib = [w for w in toks if len(w) >= 4 and (not re.search(r"[aeiouy]", w) or re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", w))]
    return len(gib) >= max(1, len(toks) // 2) and len(toks) <= 8


_ZW = re.compile("[​‌‍⁠﻿­]")
_CYR = re.compile(r"[Ѐ-ӿͰ-Ͽ]")
_EMO = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def obfuscation(raw: pd.Series) -> np.ndarray:
    raw = raw.fillna("").astype(str)
    n = raw.str.len().clip(lower=1)
    cols = {
        "zw_n": np.log1p(raw.str.count(_ZW)),
        "lookalike_n": np.log1p(raw.str.count(_CYR)),
        "mixed_tok": np.log1p(raw.map(lambda s: sum(1 for w in s.split() if _CYR.search(w) and re.search(r"[A-Za-z]", w)))),
        "upper_frac": raw.str.count(r"[A-Z]") / n,
        "max_run": np.log1p(raw.map(lambda s: max((len(m.group(0)) for m in re.finditer(r"(.)\1*", s)), default=0))),
        "emoji_n": np.log1p(raw.str.count(_EMO)),
        "spaced_letters": np.log1p(raw.str.count(r"\b(?:[A-Za-z][\s.\-*_]){2,}[A-Za-z]\b")),
        "noise_only": raw.map(noise_only).astype(float),
    }
    return pd.DataFrame(cols).to_numpy(dtype=np.float32)


@feature("obfusc")
def obfusc(bundle, weight=1.0):
    col = bundle.text_col + "_raw" if bundle.text_col + "_raw" in bundle.train.columns else bundle.text_col
    scaler = MaxAbsScaler().fit(obfuscation(bundle.train[col]))
    return lambda df: (weight * scaler.transform(obfuscation(df[col if col in df.columns else bundle.text_col]))).astype(np.float32)
