"""Build the final NeuraHack ZIP (Kaggle Rules tab s.7/s.8): exactly submission.csv, notebook.ipynb, README.md, .env.

The notebook is rebuilt from src/ + configs/ (one %%writefile cell per file, so it stays readable), starts with the
declaration section, and runs the same pipeline. Fails loudly instead of producing a ZIP that breaks a rule.

  python tools/build_submission.py --exp preli_full --team-name <Team_Name> \
      --submission <selected kaggle submission.csv> [--env <.env>] [--zip-name NeuraHack_Preli_<Team_Name>.zip]
      [--competition nsucec-neurahack-2026-preliminary] [--runtime "~25 min on a T4"]
  ZIP name: the Kaggle Rules tab says NeuraHack_Preli_<Team_Name>.zip, the PDF says NeuraHack2026_<Team>_<Code>.zip.
  Default follows the Rules tab; pass --zip-name if the organisers' email says otherwise.
  No --env: an EMPTY .env is included (rules: "If you use no API keys, include an empty .env file").
"""
import argparse, importlib.metadata as md, json, pathlib, re, sys, zipfile
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config  # noqa: E402


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}


def config_chain(exp):
    names, seen = [], set()
    while exp and exp not in seen:
        seen.add(exp)
        names.append(exp)
        exp = yaml.safe_load((ROOT / "configs" / f"{exp}.yaml").read_text()).get("base")
    return names[::-1]


def libs_table(req_file):
    rows = []
    for line in req_file.read_text().splitlines():
        name = re.split(r"[<>=!~\[ ]", line.strip(), maxsplit=1)[0]
        if name:
            try:
                rows.append(f"| {name} | {md.version(name)} |")
            except md.PackageNotFoundError:
                rows.append(f"| {name} | (not installed here) |")
    return "| Library | Version used when this notebook was built |\n|---|---|\n" + "\n".join(rows)


TECH = {f"T{i}" for i in range(1, 7)}


def check_csv(path, n_classes):
    """Competition format: id,label,techniques,action,confidence (legacy id,label also accepted)."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    if cols not in (["id", "label"], ["id", "label", "techniques", "action", "confidence"]):
        sys.exit(f"CSV columns must be id,label,techniques,action,confidence (got {cols})")
    if df["id"].duplicated().any() or df.isna().any().any():
        sys.exit("CSV has duplicate ids or blank fields")
    for c in [c for c in ("label", "action") if c in df]:
        if not pd.api.types.is_integer_dtype(df[c]) or not df[c].between(0, n_classes - 1).all():
            sys.exit(f"{c} must be integers 0..{n_classes - 1}")
    if "techniques" in df:
        for v in df["techniques"].astype(str).unique():
            codes = v.split("|")
            if v != "NONE" and (not set(codes) <= TECH):
                sys.exit(f"invalid techniques value {v!r}")
    if "confidence" in df and not pd.to_numeric(df["confidence"], errors="coerce").between(0, 1).all():
        sys.exit("confidence must be numbers in [0, 1]")
    return len(df)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--exp", required=True)
    a.add_argument("--team-name", required=True)
    a.add_argument("--team-code", default="")
    a.add_argument("--zip-name", default=None, help="default NeuraHack_Preli_<team-name>.zip (Kaggle Rules tab)")
    a.add_argument("--submission", required=True, type=pathlib.Path)
    a.add_argument("--env", default=None, type=pathlib.Path, help="omit -> an empty .env is included")
    a.add_argument("--competition", default=None, help="Kaggle slug; enables an optional data-download cell for Colab")
    a.add_argument("--runtime", default="see notebook")
    a.add_argument("--api-notes", default="See the declaration section of the notebook.")
    a.add_argument("--n-classes", type=int, default=6)
    a.add_argument("--out", type=pathlib.Path, default=ROOT / "dist")
    a.add_argument("--declaration", type=pathlib.Path, default=ROOT / "docs" / "DECLARATION.md")
    a.add_argument("--allow-todo", action="store_true", help="tests only")
    a.add_argument("--skip-install-cell", action="store_true", help="tests only")
    a = a.parse_args()

    for label, v in (("team name", a.team_name), ("team code", a.team_code)):
        if v and not re.fullmatch(r"[A-Za-z0-9_-]+", v):
            sys.exit(f"{label} must be letters, digits, _ or - only (it goes into the ZIP name)")
    decl = a.declaration.read_text()
    if "TODO" in decl and not a.allow_todo:
        sys.exit(f"{a.declaration} still contains TODO: the declaration section is mandatory (rulebook s.6)")
    if a.env is not None and not a.env.exists():
        sys.exit(f".env not found at {a.env}")
    env_text = a.env.read_text() if a.env is not None else ""
    n_rows = check_csv(a.submission, a.n_classes)

    cfg = config.load([f"exp={a.exp}"])
    cfg.pop("_hash", None)
    chain = config_chain(a.exp)
    req = ROOT / "requirements.txt"
    src_files = sorted(p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    cfg_files = [ROOT / "configs" / f"{n}.yaml" for n in chain]

    cells = [md_cell(f"# NeuraHack 2026: {a.team_name} ({a.team_code})\n\n{decl}\n\n### 1. Libraries and versions\n\n{libs_table(req)}\n\n"
                     f"### Configuration used (seeds are here)\n\n```json\n{json.dumps(cfg, indent=2)}\n```\n")]
    if not a.skip_install_cell:
        pkgs = [l.strip() for l in req.read_text().splitlines() if l.strip()]
        if any(f.get("name") == "sbert" for f in cfg.get("features", [])):
            pkgs.append("sentence-transformers")
        cells.append(code_cell("%pip install -q " + " ".join(pkgs)))
    dirs = sorted({str(p.relative_to(ROOT).parent) for p in src_files + cfg_files})
    cells.append(md_cell("## Pipeline code\nThe cells below write the project files (`src/`, `configs/`) exactly as developed, then run them."))
    cells.append(code_cell("import os\nfor d in " + repr(dirs) + ":\n    os.makedirs(d, exist_ok=True)"))
    for p in src_files + cfg_files:
        body = p.read_text() or "# package marker\n"   # %%writefile rejects an empty body
        cells.append(code_cell(f"%%writefile {p.relative_to(ROOT)}\n{body}"))
    if a.competition:
        cells.append(code_cell(
            "# Colab only: download the competition data with the Kaggle API (KAGGLE_USERNAME / KAGGLE_KEY come from .env).\n"
            "import os, subprocess, sys\nsys.path.insert(0, '.')\nfrom src.common import env\nenv.load_dotenv()\n"
            "if env.platform() != 'kaggle' and not os.path.exists('/content/data/test.csv') and os.environ.get('KAGGLE_USERNAME'):\n"
            f"    subprocess.run('pip -q install kaggle && kaggle competitions download -c {a.competition} -p /content/data "
            "&& cd /content/data && unzip -oq \"*.zip\"', shell=True, check=True)"))
    cells.append(md_cell("## Run"))
    cells.append(code_cell(
        "import sys, shutil\nsys.path.insert(0, '.')\nfrom src import run\nfrom src.common import config, env\n"
        f"run.main(['exp={a.exp}'])\n"
        f"cfg = config.load(['exp={a.exp}'])\n"
        "shutil.copy(env.work_dir() / 'outputs' / cfg['name'] / 'submission.csv', 'submission.csv')\n"
        "print('wrote submission.csv')"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 4}
    nb_text = json.dumps(nb, indent=1)

    feats = ", ".join(f["name"] + (f" ({f['params']['model']})" if f.get("params", {}).get("model") else "")
                      for f in cfg.get("features", []))
    readme = f"""# {a.team_name}: NeuraHack 2026 preliminary (ScamShield BD)

## Approach
{cfg.get("note", "")}
Features: {feats}. Model: {cfg["model"]["name"]} {cfg["model"].get("params", {})}, one head per task
(label, action, techniques T1..T6); confidence = probability of the predicted label. Labelling-guide conventions are
applied when decoding (classes 0/5 -> NONE and action 0). Text is de-obfuscated (NFKC, format characters, homoglyphs).
Training data: sample.csv plus code-generated synthetic templates and seeded adversarial augmentation; nothing is
fitted or trained on test text. Validation: repeated stratified group k-fold over near-duplicate groups.

## External models, datasets and resources
See the declaration section at the top of `notebook.ipynb` (libraries with versions, every pretrained model / dataset /
API with source and licence).

## Random seeds
seed = {cfg["seed"]} (CV folds, augmentation, synthetic data, models). Full configuration is printed in the notebook.

## How to run
1. Open `notebook.ipynb` in a clean Kaggle notebook (competition data attached, GPU + internet on) or Colab.
2. Put `.env` next to the notebook (empty if no API keys are used). Keys are read from the environment, never hardcoded.
   On Colab, upload the competition CSVs to `/content/data`, or put KAGGLE_USERNAME / KAGGLE_KEY in `.env` to download them.
3. Run all cells. The last cell writes `submission.csv` ({n_rows} rows).

- Approximate runtime: {a.runtime}
- API usage: {a.api_notes}
"""
    secrets = [v for l in env_text.splitlines() if "=" in l and not l.strip().startswith("#")
               for v in [l.split("=", 1)[1].strip().strip("'\"")] if len(v) >= 8]
    for v in secrets:
        if v in nb_text or v in readme:
            sys.exit("a value from .env appears inside the notebook/README: remove the hardcoded key")

    a.out.mkdir(parents=True, exist_ok=True)
    zpath = a.out / (a.zip_name or f"NeuraHack_Preli_{a.team_name}.zip")
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(a.submission, "submission.csv")
        z.writestr("notebook.ipynb", nb_text)
        z.writestr(".env", env_text)
        z.writestr("README.md", readme)
    names = sorted(zipfile.ZipFile(zpath).namelist())
    assert names == sorted([".env", "README.md", "notebook.ipynb", "submission.csv"]), names
    print(f"OK {zpath}\n   contains exactly: {names}\n   notebook cells: {len(cells)}, csv rows: {n_rows}")
    print("Reminder: this CSV must be one of your 2 SELECTED Kaggle submissions.")


if __name__ == "__main__":
    main()
