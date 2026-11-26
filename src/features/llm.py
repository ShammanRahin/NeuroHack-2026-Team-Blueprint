"""MEMBER B. Zero-shot LLM judgements as features (Kaggle Rules s.4/s.6: hosted LLM APIs are allowed when called by code in
the notebook and declared; the key lives only in .env).

The LLM reads the competition's labelling guide and returns, per message, label / techniques / action. These become
18 one-hot features; the classifier trained on the labelled rows learns how much to trust them. Nothing is trained on
test text: the LLM is used for inference only.

Config via environment (.env):  LLM_API_KEY, LLM_MODEL, LLM_BASE_URL (any OpenAI-compatible /chat/completions endpoint,
e.g. OpenAI, Gemini's OpenAI-compatible endpoint, Groq, OpenRouter, DeepSeek).
Cost control: many messages per request, temperature 0, persistent JSONL cache keyed by text hash (re-runs are free),
LLM_MAX_CALLS hard cap. Every call is logged to WORK_DIR/llm_usage.csv (tokens) for the declaration."""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import threading
import time
import urllib.request

import numpy as np
from src.common import env
from src.common.registry import feature

GUIDE = """You label SMS / chat messages from Bangladesh (English, Bangla, Banglish, mixed; may be obfuscated or noisy).
Judge the intent of the WHOLE message or conversation (turns separated by |||), not keywords.

label (first match in this priority order):
1 MFS/OTP scam: asks for PIN, OTP, password, NID, card details or other credentials; claims a mistaken transfer and asks
  for money back; impersonates support to get account access. (A genuine OTP delivery that warns not to share is 0.)
2 Prize/lottery scam: unearned prize, gift, cashback or reward that needs payment, a click, details or an action to claim.
4 Delivery/parcel phishing: deceptive or unexpected parcel/delivery claim to get payment, credentials, address
  confirmation or a link click.
3 Fake job/investment: unrealistic income, upfront recruitment fee, task-based earning, money doubling, crypto/investment.
5 Promotional spam: marketing, sales, discounts, product/app promotion; no fraud.
0 Legitimate: genuine transactional, informational, personal or service message; also messages warning ABOUT a scam;
  also meaningless noise-only text.
Examples of priority: prize asking for OTP -> 1; "gift" parcel demanding customs payment -> 2; job asking for OTP -> 1.

techniques (all that apply; [] for labels 0 and 5): T1 authority/organisation impersonation (a bank, MFS, courier,
government, telecom, employer is claimed), T2 credential/OTP/PIN/sensitive-info harvesting, T3 advance-fee or payment
deception, T4 prize/reward/lottery/cashback bait, T5 urgency/threat/pressure/trust-building, T6 task earning/fake
job/investment/financial manipulation.

action (the most consequential one; 0 for labels 0 and 5): 0 none, 1 share credentials/confidential info (even via a
link), 2 pay/transfer money, 3 click/open a link only, 4 install app/join group/do a task, 5 contact/call/reply.

Answer ONLY with a JSON array, one object per input message, in order:
[{"i": <index>, "label": <0-5>, "techniques": ["T1", ...], "action": <0-5>}, ...]"""

_LOCK = threading.Lock()


class LLMClient:
    def __init__(self, model, base_url, api_key, batch, workers, max_calls, timeout):
        self.model, self.base_url, self.key = model, base_url.rstrip("/"), api_key
        self.batch, self.workers, self.max_calls, self.timeout = batch, workers, max_calls, timeout
        self.calls = 0
        self.cache_path = env.work_dir() / "cache" / "llm" / f"{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.cache[r["h"]] = r
                except Exception:
                    pass

    @staticmethod
    def h(text):
        return hashlib.sha1(text.encode()).hexdigest()

    def _post(self, msgs):
        body = json.dumps({"model": self.model, "temperature": 0, "messages": [
            {"role": "system", "content": GUIDE},
            {"role": "user", "content": "\n".join(f"[{i}] {m}" for i, m in enumerate(msgs))}]}).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        txt = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})
        with _LOCK:
            with open(env.work_dir() / "llm_usage.csv", "a") as f:
                f.write(f"{time.time():.0f},{self.model},{usage.get('prompt_tokens', '')},{usage.get('completion_tokens', '')}\n")
        m = re.search(r"\[.*\]", txt, re.S)
        return json.loads(m.group(0)) if m else []

    def _run_batch(self, msgs):
        for attempt in range(4):
            with _LOCK:
                if self.calls >= self.max_calls:
                    return {}
                self.calls += 1
            try:
                res = {int(r["i"]): r for r in self._post(msgs) if isinstance(r, dict) and "i" in r}
                return res
            except Exception as e:  # rate limit / transient: back off and retry
                time.sleep(2 ** attempt + 1)
                if attempt == 3:
                    print(f"[llm] batch failed: {e}")
        return {}

    def label(self, texts):
        todo = sorted({t for t in texts if self.h(t) not in self.cache})
        batches = [todo[i:i + self.batch] for i in range(0, len(todo), self.batch)]
        if batches:
            print(f"[llm] {len(todo)} new texts in {len(batches)} requests ({self.model})")
        with cf.ThreadPoolExecutor(self.workers) as ex, open(self.cache_path, "a", encoding="utf-8") as f:
            for msgs, res in zip(batches, ex.map(self._run_batch, batches)):
                for i, t in enumerate(msgs):
                    r = res.get(i)
                    if r is None:
                        continue
                    rec = {"h": self.h(t), "label": r.get("label"), "techniques": r.get("techniques", []),
                           "action": r.get("action")}
                    self.cache[rec["h"]] = rec
                    with _LOCK:
                        f.write(json.dumps(rec) + "\n")
        return [self.cache.get(self.h(t)) for t in texts]


def _onehot(recs):
    X = np.zeros((len(recs), 19), dtype=np.float32)  # 6 label + 6 techniques + 6 action + 1 missing
    for n, r in enumerate(recs):
        try:
            X[n, int(r["label"])] = 1
            for t in r.get("techniques") or []:
                X[n, 6 + int(str(t).strip().upper().lstrip("T")) - 1] = 1
            X[n, 12 + int(r["action"])] = 1
        except Exception:
            X[n, :] = 0
            X[n, 18] = 1  # no / unparseable answer
    return X


_CLIENTS = {}


@feature("llm")
def llm(bundle, model=None, base_url=None, batch=25, workers=8, max_calls=None, timeout=120, weight=1.0):
    model = model or os.environ.get("LLM_MODEL")
    base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get("LLM_API_KEY")
    if not (model and key):
        raise RuntimeError("llm feature needs LLM_API_KEY and LLM_MODEL in .env (never hardcode keys)")
    max_calls = int(max_calls or os.environ.get("LLM_MAX_CALLS", 50_000))
    ck = (model, base_url)
    if ck not in _CLIENTS:
        _CLIENTS[ck] = LLMClient(model, base_url, key, batch, workers, max_calls, timeout)
    client = _CLIENTS[ck]
    return lambda df: weight * _onehot(client.label(df[bundle.text_col].astype(str).tolist()))
