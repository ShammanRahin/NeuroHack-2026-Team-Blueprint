"""Quick check of zero-shot signals on the labelled rows (before spending GPU hours on 500k test rows).
  python tools/eval_zeroshot.py [--llm Qwen/Qwen2.5-1.5B-Instruct] [--nli] [--data-dir DIR]
Prints zero-shot macro-F1 (argmax, no training) for label/action and throughput -> estimated time for 500k rows."""
import argparse, pathlib, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import env, metrics
from src.data.normalize import robust

a = argparse.ArgumentParser()
a.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
a.add_argument("--no-llm", action="store_true")
a.add_argument("--nli", action="store_true")
a.add_argument("--data-dir", default=None)
a.add_argument("--batch", type=int, default=64)
a = a.parse_args()
d = pathlib.Path(a.data_dir) if a.data_dir else env.data_dir({"data": {"dir": None, "test": "test.csv"}})
s = pd.read_csv(d / "sample.csv")
texts = s["text"].map(robust).tolist()
test = pd.read_csv(d / "test.csv", nrows=2000)["text"].map(robust).tolist()

if not a.no_llm:
    from src.features.llm_local import LocalLLM
    r = LocalLLM(a.llm, 192, a.batch)
    for q, col in (("label", "label"), ("action", "action")):
        P = r._score(texts, q)
        print(f"[llm {a.llm}] zero-shot {q}: macro-F1 {metrics.macro_f1(s[col].to_numpy(), P.argmax(1)):.3f}")
    t0 = time.time(); r._score(test, "label"); dt = time.time() - t0
    print(f"[llm] {len(test) / dt:.0f} rows/s per question -> 500k rows ~ {500_000 / (len(test) / dt) / 3600:.2f} h per question")
if a.nli:
    from src.features.zeroshot import NLIScorer, HYPOTHESES
    n = NLIScorer("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", 128, 256)
    t0 = time.time(); n._score(test[:500], HYPOTHESES); dt = time.time() - t0
    print(f"[nli] {500 / dt:.0f} rows/s ({len(HYPOTHESES)} hypotheses) -> 500k rows ~ {500_000 / (500 / dt) / 3600:.2f} h")
