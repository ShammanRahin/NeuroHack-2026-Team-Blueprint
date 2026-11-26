"""MEMBER B. Dense multilingual sentence embeddings (+ optional SetFit-style contrastive fine-tuning of the encoder).

Pretrained encoders (declare every one used in docs/DECLARATION.md):
  intfloat/multilingual-e5-small            118M, MIT, best quality/speed trade-off for Bangla in published benchmarks
  kazalbrur/bangla-embed-e5-small-banglish  118M, e5-small distilled from BGE-M3 with romanized-Bangla (Banglish) views
  BAAI/bge-m3                               568M, MIT, strongest but ~5x slower (500k rows: plan the runtime)
e5 models expect the "query: " prefix.

Rules: pretrained weights are an allowed, declared resource. The optional fine-tuning (finetune_steps > 0) uses
bundle.train only (the current CV fold's training rows); the runner fits features inside every fold, so no
validation label leaks. Test text is only ever encoded, never trained on.

Speed: fp16 on GPU, length-sorted batches, in-process memo for repeated texts (CV folds re-use the same rows),
and a disk cache per 50k-row test chunk under WORK_DIR/cache/emb (so re-runs and other experiments are free)."""
import copy
import hashlib
import random

import numpy as np
from src.common import env
from src.common.registry import feature

_ENCODERS = {}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(model_name, max_seq_length):
    key = (model_name, max_seq_length)
    if key not in _ENCODERS:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name, device=_device())
        m.max_seq_length = max_seq_length
        _ENCODERS[key] = m
    return _ENCODERS[key]


class Encoder:
    def __init__(self, model, prefix, batch_size, cache_tag):
        self.model, self.prefix, self.batch_size, self.cache_tag = model, prefix, batch_size, cache_tag
        self.memo = {}

    def _encode(self, texts):
        import torch
        with torch.autocast("cuda", dtype=torch.float16, enabled=_device() == "cuda"):
            v = self.model.encode([self.prefix + t for t in texts], batch_size=self.batch_size,
                                  normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return v.astype(np.float32)

    def __call__(self, texts):
        texts = [str(t) for t in texts]
        if len(texts) >= 20_000 and self.cache_tag:  # big test chunk: disk cache
            h = hashlib.sha1(("\x00".join(texts) + self.cache_tag).encode()).hexdigest()[:20]
            p = env.work_dir() / "cache" / "emb" / f"{h}.npy"
            if p.exists():
                return np.load(p)
            v = self._encode(texts)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, v.astype(np.float16))
            return v
        todo = sorted({t for t in texts if t not in self.memo})
        if todo:
            for t, v in zip(todo, self._encode(todo)):
                self.memo[t] = v
        return np.stack([self.memo[t] for t in texts]) if texts else np.zeros((0, self.model.get_sentence_embedding_dimension()), np.float32)


def _setfit_finetune(model, texts, labels, prefix, steps, batch_size, lr, seed):
    """SetFit body training: contrastive pairs (same label -> cosine 1, different -> 0), CosineSimilarityLoss.
    Plain torch loop so it does not depend on a particular sentence-transformers trainer API."""
    import torch
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    model.train()
    by_label = {}
    for t, y in zip(texts, labels):
        by_label.setdefault(y, []).append(prefix + t)
    labs = [y for y in by_label if by_label[y]]
    if len(labs) < 2:
        return model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    def emb(batch):
        feats = model.tokenize(batch)
        feats = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in feats.items()}  # v6 adds non-tensor keys
        return torch.nn.functional.normalize(model(feats)["sentence_embedding"], dim=-1)

    for _ in range(steps):
        a, b, target = [], [], []
        for _ in range(batch_size):
            y = rng.choice(labs)
            if rng.random() < 0.5 and len(by_label[y]) > 1:
                x1, x2 = rng.sample(by_label[y], 2)
                target.append(1.0)
            else:
                y2 = rng.choice([l for l in labs if l != y])
                x1, x2 = rng.choice(by_label[y]), rng.choice(by_label[y2])
                target.append(0.0)
            a.append(x1)
            b.append(x2)
        sim = (emb(a) * emb(b)).sum(-1)
        loss = loss_fn(sim, torch.tensor(target, device=sim.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@feature("sbert")
def sbert(bundle, model="intfloat/multilingual-e5-small", prefix="query: ", max_seq_length=256, batch_size=128,
          weight=1.0, finetune_steps=0, finetune_batch=16, finetune_lr=2e-5, seed=42):
    base = _load(model, max_seq_length)
    if finetune_steps > 0:
        tuned = _setfit_finetune(base, bundle.train[bundle.text_col].tolist(), bundle.train[bundle.target_col].tolist(),
                                 prefix, finetune_steps, finetune_batch, finetune_lr, seed)
        enc = Encoder(tuned, prefix, batch_size, cache_tag=None)  # fold-specific weights: never share the disk cache
    else:
        enc = Encoder(base, prefix, batch_size, cache_tag=f"{model}|{max_seq_length}|{prefix}")
        enc.memo = _SHARED_MEMO.setdefault((model, max_seq_length, prefix), {})
    return lambda df: weight * enc(df[bundle.text_col].tolist())


_SHARED_MEMO = {}


# label-description similarity: same encoder + cache as `sbert`, so it costs ~nothing extra
# each prototype = mean of an English, a Banglish and a Bangla-script description (written from the labelling guide)
DESCRIPTIONS = [
    ["Legitimate genuine transaction, OTP delivery, bill, delivery update, service or personal message",
     "apnar taka joma hoyeche, bill er reminder, order deliver hobe, OTP kauke bolben na, bondhur sathe kotha",
     "আপনার লেনদেন সফল হয়েছে, বিলের রিমাইন্ডার, অর্ডার ডেলিভারি হবে, ব্যক্তিগত কথা"],
    ["Scam asking for PIN, OTP, password, NID or card details, or asking to return money sent by mistake",
     "apnar PIN ba OTP ta din, NID pathan, bhul kore taka chole gese ferot din",
     "আপনার পিন বা ওটিপি দিন, এনআইডি পাঠান, ভুল করে টাকা চলে গেছে ফেরত দিন"],
    ["Prize or lottery scam: you won a prize, gift or cashback, pay a fee or click to claim it",
     "apni lottery te jitechen, gift ba cashback pete fee din ba link e click korun",
     "আপনি লটারি জিতেছেন, উপহার বা ক্যাশব্যাক পেতে ফি দিন বা লিংকে ক্লিক করুন"],
    ["Fake job or investment: earn money from home, task income, registration fee, double your money, crypto",
     "ghore bose income korun, task kore taka, registration fee din, taka double, invest korun",
     "ঘরে বসে আয় করুন, কাজ করে টাকা, রেজিস্ট্রেশন ফি দিন, টাকা দ্বিগুণ, বিনিয়োগ করুন"],
    ["Parcel or delivery phishing: parcel held, pay delivery or customs fee, confirm address at a link",
     "apnar parcel atke ase, delivery charge ba customs fee din, link e address confirm korun",
     "আপনার পার্সেল আটকে আছে, ডেলিভারি চার্জ বা কাস্টমস ফি দিন, লিংকে ঠিকানা নিশ্চিত করুন"],
    ["Promotional advertisement: sale, discount, offer, new product or app promotion",
     "offer! sob item e discount, notun product, app install kore chhar pan",
     "অফার! সব পণ্যে ছাড়, নতুন পণ্য, অ্যাপে বিশেষ ছাড়"],
    ["Sender claims to be a bank, mobile wallet, courier, telecom, employer or government",
     "ami bKash ba bank theke bolchi, courier office, sarkari office",
     "আমি বিকাশ বা ব্যাংক থেকে বলছি, কুরিয়ার অফিস, সরকারি অফিস"],
    ["Request for credentials, OTP, PIN or sensitive personal information",
     "OTP, PIN, password ba NID number din",
     "ওটিপি, পিন, পাসওয়ার্ড বা এনআইডি নম্বর দিন"],
    ["Request to pay an advance fee or send money first",
     "age fee ba charge pathan, tarpor paben",
     "আগে ফি বা চার্জ পাঠান, তারপর পাবেন"],
    ["Urgency, threat or pressure: act now or the account will be blocked",
     "ekhoni korun, na hole account band hoye jabe, 1 ghontar moddhe",
     "এখনই করুন, না হলে অ্যাকাউন্ট বন্ধ হয়ে যাবে, ১ ঘণ্টার মধ্যে"],
    ["Request to click or open a link or website",
     "ei link e click korun, website e jan",
     "এই লিংকে ক্লিক করুন, ওয়েবসাইটে যান"],
    ["Request to install an app, join a group or do a task",
     "app download korun, telegram ba whatsapp group e join korun",
     "অ্যাপ ডাউনলোড করুন, টেলিগ্রাম বা হোয়াটসঅ্যাপ গ্রুপে যোগ দিন"],
    ["Request to call, reply to or contact the sender",
     "ei number e call korun, reply din, inbox korun",
     "এই নম্বরে কল করুন, উত্তর দিন, যোগাযোগ করুন"],
]


@feature("labeldesc")
def labeldesc(bundle, model="intfloat/multilingual-e5-small", prefix="query: ", max_seq_length=256, batch_size=128,
              weight=1.0):
    base = _load(model, max_seq_length)
    enc = Encoder(base, prefix, batch_size, cache_tag=f"{model}|{max_seq_length}|{prefix}")
    enc.memo = _SHARED_MEMO.setdefault((model, max_seq_length, prefix), {})
    flat = [t for group in DESCRIPTIONS for t in group]
    E = Encoder(base, prefix, batch_size, cache_tag=None)(flat).reshape(len(DESCRIPTIONS), -1, base.get_sentence_embedding_dimension())
    D = E.mean(1)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    return lambda df: weight * (enc(df[bundle.text_col].tolist()) @ D.T)
