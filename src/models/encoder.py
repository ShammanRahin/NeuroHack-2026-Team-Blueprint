"""Multi-task transformer fine-tuned on the 1M-row synthetic set (+ augmented real rows): label / action / techniques.

Kept deliberately hard to overfit:
  * the backbone is a small pretrained multilingual encoder (Banglish e5-small, MIT), fine-tuned for one pass only;
  * label smoothing, dropout, weight decay, linear warm-up/decay;
  * ~8% of the generator's TEMPLATES are held out from training entirely. Their rows (unseen phrasings) and the 86
    real rows are the validation sets; training stops early (keeps the best checkpoint) if the held-out score falls;
  * the temperature for Task D is fitted on the held-out templates, so confidence is calibrated for NOVEL text.
Test text is only ever passed through the trained model (prediction); nothing is fitted on it.
"""
import hashlib
import math
import random
import time

import numpy as np

N_CLS, N_TECH = 6, 6


def _hash01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def template_holdout(templates, share=0.08, salt="enc"):
    """True for rows whose generator template (or composer hook x request pair) is held out."""
    return np.array([t != "noise" and _hash01(f"{salt}:{t}") < share for t in templates])


class _Tok:
    """Pre-tokenises into one flat int32 buffer + offsets (1M lists of python ints would cost ~2 GB)."""

    def __init__(self, tok, texts, max_len, prefix, chunk=20_000):
        ids, lens = [], []
        for s in range(0, len(texts), chunk):
            enc = tok([prefix + t for t in texts[s:s + chunk]], truncation=True, max_length=max_len,
                      add_special_tokens=True)["input_ids"]
            for e in enc:
                ids.append(np.asarray(e, dtype=np.int32))
                lens.append(len(e))
        self.lens = np.asarray(lens, dtype=np.int32)
        self.off = np.concatenate([[0], np.cumsum(self.lens)]).astype(np.int64)
        self.buf = np.concatenate(ids) if ids else np.zeros(0, np.int32)
        self.pad = tok.pad_token_id

    def batch(self, idx):
        L = int(self.lens[idx].max())
        x = np.full((len(idx), L), self.pad, dtype=np.int64)
        m = np.zeros((len(idx), L), dtype=np.int64)
        for r, i in enumerate(idx):
            n = self.lens[i]
            x[r, :n] = self.buf[self.off[i]:self.off[i] + n]
            m[r, :n] = 1
        return x, m


def _batches(lens, bs, rng, shuffle=True):
    """Length-bucketed batches (sort within shuffled mega-batches) -> little padding, random order."""
    idx = np.arange(len(lens))
    if shuffle:
        rng.shuffle(idx)
    out, mega = [], bs * 64
    for s in range(0, len(idx), mega):
        chunk = idx[s:s + mega]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        out += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
    if shuffle:
        rng.shuffle(out)
    return out


def _build(model_name, dropout):
    import torch
    from transformers import AutoModel

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(model_name)
            h = self.enc.config.hidden_size
            self.drop = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(h, 2 * N_CLS + N_TECH)

        def forward(self, x, m):
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):  # inside: DataParallel replicas are threads
                hs = self.enc(input_ids=x, attention_mask=m).last_hidden_state
            hs = hs.float()
            mf = m.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mf).sum(1) / mf.sum(1).clamp(min=1.0)  # mean pooling, as e5 was trained
            return self.head(self.drop(pooled))

    return Net()


def _predict(net, tk, dev, bs):
    import torch
    order = np.argsort(tk.lens, kind="stable")
    out = np.zeros((len(tk.lens), 2 * N_CLS + N_TECH), dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            x, m = tk.batch(idx)
            out[idx] = net(torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)).cpu().numpy()
    net.train()
    return out


def _probs(logits, T=1.0):
    def sm(z):
        z = z / T
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    return {"label": sm(logits[:, :N_CLS]), "action": sm(logits[:, N_CLS:2 * N_CLS]),
            "techniques": 1.0 / (1.0 + np.exp(-logits[:, 2 * N_CLS:]))}


def train_predict(train_texts, Y, valid, predict, model_name, max_len=160, prefix="query: ", batch_size=64,
                  lr=5e-5, epochs=1.0, warmup=0.06, weight_decay=0.01, label_smoothing=0.1, dropout=0.1,
                  w_action=0.5, w_tech=0.5, evals=4, seed=42, score_fn=None, max_minutes=50, log=print):
    """Train on (train_texts, Y={label, action, techniques}); `valid` = {name: (texts, Y)} scored `evals` times
    (best checkpoint by the mean of score_fn over the valid sets is kept); returns ({name: probs} for `predict`
    = {name: texts}, info). Temperature for Task D is fitted on valid set 'templates' (unseen templates)."""
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    tr = _Tok(tok, list(train_texts), max_len, prefix)
    vk = {k: _Tok(tok, list(v[0]), max_len, prefix) for k, v in valid.items()}
    log(f"[encoder] tokenised {len(tr.lens):,} train rows in {time.time() - t0:.0f}s "
        f"(mean {tr.lens.mean():.0f} tokens, {np.mean(tr.lens >= max_len):.1%} truncated at {max_len})")

    net = _build(model_name, dropout).to(dev)
    model = torch.nn.DataParallel(net) if dev == "cuda" and torch.cuda.device_count() > 1 else net
    n_gpu = torch.cuda.device_count() if dev == "cuda" else 1
    bs = batch_size * max(1, n_gpu)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
    groups = [{"params": [p for n, p in net.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": weight_decay},
              {"params": [p for n, p in net.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps_per_epoch = math.ceil(len(tr.lens) / bs)
    total = max(1, int(steps_per_epoch * epochs))
    sched = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    bce = torch.nn.BCEWithLogitsLoss()
    yl = torch.as_tensor(Y["label"], dtype=torch.long)
    ya = torch.as_tensor(Y["action"], dtype=torch.long)
    yt = torch.as_tensor(np.asarray(Y["techniques"], dtype=np.float32))
    yt_s = yt * (1 - label_smoothing) + 0.5 * label_smoothing

    eval_at = {int(total * (k + 1) / evals) for k in range(evals)}
    best, best_state, hist, step, run_loss = -1.0, None, [], 0, 0.0
    log(f"[encoder] {model_name} on {dev} x{n_gpu}: {total:,} steps, batch {bs}, lr {lr}, {epochs} epoch(s)")
    model.train()
    while step < total:
        for idx in _batches(tr.lens, bs, rng):
            if step >= total:
                break
            x, m = tr.batch(idx)
            x, m = torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)
            ti = torch.as_tensor(idx)
            z = model(x, m)
            loss = (ce(z[:, :N_CLS], yl[ti].to(dev)) + w_action * ce(z[:, N_CLS:2 * N_CLS], ya[ti].to(dev))
                    + w_tech * bce(z[:, 2 * N_CLS:], yt_s[ti].to(dev)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * float(loss) if step > 1 else float(loss)
            if step % 500 == 0:
                log(f"[encoder] step {step:,}/{total:,} loss {run_loss:.4f} ({time.time() - t0:.0f}s)")
            late = time.time() - t0 > 60 * max_minutes
            if late:
                log(f"[encoder] time cap {max_minutes} min reached at step {step:,} -> final eval and stop")
                total = step
            if (step in eval_at or late) and score_fn is not None:
                scores = {k: score_fn(_probs(_predict(model, vk[k], dev, 512 * n_gpu)), valid[k][1]) for k in valid}
                s = float(np.mean(list(scores.values())))
                hist.append((step, scores))
                log(f"[encoder] eval @ {step:,}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items())
                    + f" | mean {s:.4f}" + (" (best)" if s > best else ""))
                if s > best:
                    best = s
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
            if late:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    T, info = 1.0, {"history": hist, "best": best, "seconds": time.time() - t0}
    if "templates" in vk:  # Task D: temperature on UNSEEN templates (novel phrasing), label head
        lz = _predict(model, vk["templates"], dev, 512 * n_gpu)
        y = np.asarray(valid["templates"][1]["label"])
        grid = np.exp(np.linspace(np.log(0.5), np.log(4.0), 36))

        def brier(Tc):
            p = _probs(lz, Tc)["label"]
            return np.mean((p.max(1) - (p.argmax(1) == y)) ** 2)
        T = float(min(grid, key=brier))
        info["T"] = T
        log(f"[encoder] temperature on held-out templates T={T:.2f} (Brier-style {brier(1.0):.4f} -> {brier(T):.4f})")
    out = {}
    for k, texts in predict.items():
        tk = _Tok(tok, list(texts), max_len, prefix)
        out[k] = _probs(_predict(model, tk, dev, 512 * n_gpu), T)
        log(f"[encoder] predicted {k}: {len(texts):,} rows ({time.time() - t0:.0f}s total)")
    del model, net, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out, info
