"""Same code on Kaggle / Colab / laptop: only paths differ."""
import glob, os, pathlib, sys


def platform():
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        return "colab"
    return "local"


def data_dir(cfg):
    d = cfg["data"].get("dir") or os.environ.get("DATA_DIR")
    if d:
        return pathlib.Path(d)
    p = platform()
    if p == "kaggle":
        hits = sorted(glob.glob(f"/kaggle/input/**/{cfg['data']['test']}", recursive=True), key=len)  # any mount depth
        if not hits:
            raise FileNotFoundError("Attach the competition data to this notebook")
        return pathlib.Path(hits[0]).parent
    return pathlib.Path("/content/data" if p == "colab" else "data")


def work_dir():
    """Cache + outputs. On Colab set WORK_DIR to a Drive folder to survive restarts."""
    d = os.environ.get("WORK_DIR") or {"kaggle": "/kaggle/working", "colab": "/content/work"}.get(platform(), "work")
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_dotenv(path=".env"):
    """Minimal KEY=VALUE loader so the notebook reads API keys from .env (never hardcoded). Real env vars win."""
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
