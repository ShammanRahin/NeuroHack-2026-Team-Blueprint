"""End-to-end: build the ZIP, then EXECUTE the generated notebook in a clean temp dir and check its output.
Needs: nbconvert, ipykernel.   python tests/package_test.py"""
import os, pathlib, subprocess, sys, tempfile, zipfile
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from smoke_test import make

with tempfile.TemporaryDirectory() as t:
    t = pathlib.Path(t)
    data = t / "data"; data.mkdir(); make(data)
    pd.DataFrame({"id": range(100, 130), "label": 0}).to_csv(t / "sel.csv", index=False)
    (t / "key.env").write_text("OPENAI_API_KEY=sk-test-1234567890abcdef\n")
    cmd = [sys.executable, str(ROOT / "tools/build_submission.py"), "--exp", "baseline", "--team-name", "TeamX", "--team-code", "NH1",
           "--submission", str(t / "sel.csv"), "--env", str(t / "key.env"), "--out", str(t / "dist"), "--n-classes", "3",
           "--allow-todo", "--skip-install-cell"]
    # 1) a TODO declaration must be refused without --allow-todo
    r = subprocess.run([c for c in cmd if c not in ("--allow-todo",)], capture_output=True, text=True)
    assert r.returncode != 0 and "TODO" in r.stderr + r.stdout, "should refuse an unfinished declaration"
    # 2) real build
    print(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    z = t / "dist" / "NeuraHack2026_TeamX_NH1.zip"
    names = sorted(zipfile.ZipFile(z).namelist()); assert names == [".env", "README.md", "notebook.ipynb", "submission.csv"], names
    # 3) run the notebook from scratch
    run = t / "run"; run.mkdir(); zipfile.ZipFile(z).extractall(run)
    envv = dict(os.environ, DATA_DIR=str(data), WORK_DIR=str(t / "work"), RESULTS_LOG=str(t / "log.csv"), EXP_USER="ci")
    out = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "notebook.ipynb",
                          "--output", "executed.ipynb"], cwd=run, env=envv, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    sub = pd.read_csv(run / "submission.csv")
    assert list(sub.columns) == ["id", "label"] and len(sub) == 30
    # 4) a hardcoded key must be caught
    (t / "bad.env").write_text("K=" + "baseline" + "\n")  # value 'baseline' appears in the notebook
    r = subprocess.run([c if c != str(t / "key.env") else str(t / "bad.env") for c in cmd], capture_output=True, text=True)
    assert r.returncode != 0 and ".env" in r.stderr + r.stdout
print("PACKAGE TEST OK")
