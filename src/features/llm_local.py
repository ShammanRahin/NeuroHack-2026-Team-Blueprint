"""MEMBER B. Zero-shot judgements from a LOCAL open-source LLM on the notebook's GPU (no API, no key, no cost).

Default model Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0). The model reads the competition's labelling guide + one message
and we read its next-token probabilities for the answer digits 0-5 directly (one forward pass, no generation), for:
  label  (Task A classes)      and optionally   action  (Task C)
-> 6 features per question. The classifier trained on the labelled rows learns how much to trust them.
Nothing is trained or fitted on test text (inference only).

Speed: the long guide prefix is encoded ONCE and its KV-cache is reused for every message; only the short message
suffix is computed per row. Length-sorted batches, fp16, truncation, disk cache per large chunk (WORK_DIR/cache/llmloc)."""
import copy
import hashlib

import numpy as np
from src.common import env
from src.common.registry import feature

GUIDE_LABEL = """You classify SMS / chat messages from Bangladesh (English, Bangla, Banglish or mixed; may be noisy or obfuscated).
Judge the intent of the whole message or conversation (turns are separated by |||), not single keywords.
Labels, checked in this priority order (stop at the first that applies):
1 = MFS/OTP scam: asks for PIN, OTP, password, NID, card details or other credentials; or claims money was sent by mistake
    and asks for it back; or impersonates support to get account access. A genuine OTP delivery that says not to share
    the code is 0.
2 = Prize/lottery scam: unearned prize, gift, cashback or reward that needs a payment, click, details or action to claim.
4 = Delivery/parcel phishing: deceptive parcel or delivery claim asking for a fee, customs payment, address confirmation
    or a link click.
3 = Fake job/investment: unrealistic income, recruitment fee, task-based earning, money doubling, crypto or investment.
5 = Promotional spam: ordinary marketing, sales, discounts, product or app promotion, no fraud.
0 = Legitimate: genuine transaction, OTP delivery, bill, delivery update, service or personal message; a message warning
    about a scam; meaningless noise text.
Reply with one digit only."""

GUIDE_ACTION = """You read SMS / chat messages from Bangladesh (English, Bangla, Banglish or mixed; may be noisy).
What is the main action the sender wants the recipient to take? If several, choose the most consequential.
0 = nothing suspicious / informational only (also for normal promotions and genuine messages)
1 = share or enter a PIN, OTP, password, NID, card details or other confidential information (even through a link)
2 = pay, transfer or send money or a fee
3 = only click or open a link or website
4 = install an app, join a group or complete a task
5 = call, reply to, contact or message someone
Reply with one digit only."""

QUESTIONS = {"label": GUIDE_LABEL, "action": GUIDE_ACTION}
_RUNNERS = {}


class LocalLLM:
    def __init__(self, model_name, max_msg_tokens, batch_size):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.max_msg_tokens, self.batch_size, self.name = max_msg_tokens, batch_size, model_name
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        self.digits = [self.tok.encode(str(d), add_special_tokens=False)[0] for d in range(6)]
        self.chatml = "<|im_start|>" in self.tok.get_vocab()
        self.prefix = {}
        self.memo = {}

    def _parts(self, guide):
        if self.chatml:
            return (f"<|im_start|>system\n{guide}<|im_end|>\n<|im_start|>user\nMessage:\n",
                    "<|im_end|>\n<|im_start|>assistant\n")
        return f"{guide}\n\nMessage:\n", "\nAnswer:"

    def _prefix_cache(self, q):
        if q not in self.prefix:
            pre, _ = self._parts(QUESTIONS[q])
            ids = self.tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            with self.torch.inference_mode():
                out = self.model(input_ids=ids, use_cache=True)
            self.prefix[q] = (out.past_key_values, ids.shape[1])
        return self.prefix[q]

    def _score(self, texts, q):
        torch = self.torch
        cache0, P = self._prefix_cache(q)
        _, post = self._parts(QUESTIONS[q])
        post_ids = self.tok.encode(post, add_special_tokens=False)
        seqs = []
        for t in texts:
            ids = self.tok.encode(str(t), add_special_tokens=False)[: self.max_msg_tokens]
            seqs.append(ids + post_ids)
        order = np.argsort([len(s) for s in seqs], kind="stable")
        out = np.zeros((len(texts), 6), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                L = max(len(seqs[i]) for i in idx)
                inp = torch.full((len(idx), L), self.pad, dtype=torch.long)
                mask = torch.zeros((len(idx), P + L), dtype=torch.long)
                mask[:, :P] = 1
                last = []
                for r, i in enumerate(idx):
                    n = len(seqs[i])
                    inp[r, :n] = torch.tensor(seqs[i])
                    mask[r, P:P + n] = 1
                    last.append(n - 1)
                cache = copy.deepcopy(cache0)
                cache.batch_repeat_interleave(len(idx))
                pos = (P + torch.arange(L)).unsqueeze(0).expand(len(idx), L)
                logits = self.model(input_ids=inp.to(self.device), attention_mask=mask.to(self.device),
                                    position_ids=pos.to(self.device), past_key_values=cache, use_cache=True).logits
                sel = logits[torch.arange(len(idx)), torch.tensor(last)][:, self.digits].float()
                out[idx] = torch.softmax(sel, -1).cpu().numpy()
        return out

    def __call__(self, texts, q):
        texts = [str(t) for t in texts]
        big = len(texts) >= 20_000
        if big:
            h = hashlib.sha1(("\x00".join(texts) + q + self.name + str(self.max_msg_tokens)).encode()).hexdigest()[:20]
            p = env.work_dir() / "cache" / "llmloc" / f"{h}.npy"
            if p.exists():
                return np.load(p)
            res = self._score(texts, q)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, res)
            return res
        todo = sorted({t for t in texts if (t, q) not in self.memo})
        if todo:
            for t, v in zip(todo, self._score(todo, q)):
                self.memo[(t, q)] = v
        return np.stack([self.memo[(t, q)] for t in texts]) if texts else np.zeros((0, 6), np.float32)


@feature("llm_local")
def llm_local(bundle, model="Qwen/Qwen2.5-1.5B-Instruct", questions=("label",), max_msg_tokens=192, batch_size=64,
              weight=1.0):
    key = (model, max_msg_tokens, batch_size)
    if key not in _RUNNERS:
        _RUNNERS[key] = LocalLLM(model, max_msg_tokens, batch_size)
    r = _RUNNERS[key]
    return lambda df: weight * np.hstack([r(df[bundle.text_col].tolist(), q) for q in questions])
