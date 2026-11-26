"""MEMBER B. Zero-shot features that need NO training data, so they carry over to unseen scenarios and wording.

1. `nli`: a multilingual NLI model scores every message (premise) against hypotheses written from the competition's
   labelling guide (class / technique / action definitions). Feature = P(entailment) per hypothesis.
   Default model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT; NLI data in 26 languages incl. Bengali).
2. `labeldesc` (in embeddings.py): cosine similarity between the message embedding and each guide description.

Nothing is fitted here (no test text is learned from). Speed for 500k rows: length-sorted dynamic padding, fp16 on GPU,
de-duplication, in-process memo and a disk cache per large chunk under WORK_DIR/cache/nli."""
import hashlib

import numpy as np
from src.common import env
from src.common.registry import feature

HYPOTHESES = [
    # classes (labelling guide)
    "This message asks the recipient to share an OTP, PIN, password, card number or NID.",
    "This message claims money was sent by mistake and asks for it to be sent back.",
    "This message says the recipient won a prize, gift, lottery, bonus or cashback.",
    "This message offers a job, task-based income, money doubling or an investment.",
    "This message is about a parcel or delivery that needs a fee, customs payment, address confirmation or a link.",
    "This message is an advertisement or a promotional offer.",
    "This message is a genuine transaction, payment, OTP or account notification.",
    "This message is a normal personal conversation.",
    "This message warns someone about a scam or fraud.",
    # techniques
    "The sender claims to be from a bank, mobile wallet, courier, telecom company, employer or government office.",
    "The message asks the recipient to pay money or a fee.",
    "The message pressures the recipient with urgency, threats or a deadline.",
    # actions
    "The message asks the recipient to click or open a link.",
    "The message asks the recipient to install an app or join a group.",
    "The message asks the recipient to call, reply to or contact someone.",
    # noise-only records
    "This message is meaningless random text.",
]

_MODELS = {}


class NLIScorer:
    def __init__(self, model_name, max_length, batch_size):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ent = next(i for name, i in labels.items() if name.startswith("entail"))
        self.max_length, self.batch_size = max_length, batch_size
        self.name = model_name
        self.memo = {}

    def _score(self, texts, hypotheses):
        import torch
        pairs = [(t, h) for t in texts for h in hypotheses]
        order = np.argsort([len(p[0]) for p in pairs], kind="stable")  # length-sorted -> short padding
        out = np.zeros(len(pairs), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="only_first",
                               max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                out[idx] = torch.softmax(logits, -1)[:, self.ent].cpu().numpy()
        return out.reshape(len(texts), len(hypotheses))

    def __call__(self, texts, hypotheses):
        texts = [str(t) for t in texts]
        key = "\x00".join(hypotheses)
        if len(texts) >= 20_000:
            h = hashlib.sha1(("\x00".join(texts) + key + self.name + str(self.max_length)).encode()).hexdigest()[:20]
            p = env.work_dir() / "cache" / "nli" / f"{h}.npy"
            if p.exists():
                return np.load(p)
        uniq = sorted({t for t in texts if (t, key) not in self.memo})
        if uniq:
            for t, v in zip(uniq, self._score(uniq, hypotheses)):
                self.memo[(t, key)] = v
        res = np.stack([self.memo[(t, key)] for t in texts]) if texts else np.zeros((0, len(hypotheses)), np.float32)
        if len(texts) >= 20_000:
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, res)
            for t in set(texts):              # don't keep 500k entries in RAM
                self.memo.pop((t, key), None)
        return res


@feature("nli")
def nli(bundle, model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", max_length=128, batch_size=256,
        weight=1.0, hypotheses=None):
    hyps = hypotheses or HYPOTHESES
    key = (model, max_length, batch_size)
    if key not in _MODELS:
        _MODELS[key] = NLIScorer(model, max_length, batch_size)
    scorer = _MODELS[key]
    return lambda df: weight * scorer(df[bundle.text_col].tolist(), hyps)
