"""Run training on Kaggle GPUs from the command line (private dataset + private notebooks).

    python tools/kaggle_run.py bundle                       # upload/refresh code as private dataset
    python tools/kaggle_run.py launch e01_baseline -- --fold 0 --epochs 40
    python tools/kaggle_run.py status e01_baseline
    python tools/kaggle_run.py fetch  e01_baseline         # download outputs into runs/
    python tools/kaggle_run.py queue E06 E07 E08          # run experiments (max 2 at once), auto-fetch
    python tools/kaggle_run.py batch batch_a E05 E04      # one kernel runs them back to back (laptop can be off)
    python tools/kaggle_run.py fetch batch_a              # later: download every run it produced

The code dataset contains src/, data/folds.csv and data/processed (disk profiles + label maps), so
Kaggle runs need no preprocessing. Each experiment is its own private kernel `filament-<name>`;
it trains, saves OOF probabilities, tunes post-processing, and leaves everything in /kaggle/working.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "mkamangu"
CODE_DATASET = f"{USER}/filament-code"
COMPETITION = "filament-segmentation-2026"
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
WORK = ROOT / ".kaggle_build"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

RUNNER = r'''
import os, subprocess, sys, shutil, glob
import torch
assert torch.cuda.is_available(), "no GPU attached (Kaggle account needs phone verification for GPUs)"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
code_src = glob.glob("/kaggle/input/**/filament-code", recursive=True) + glob.glob("/kaggle/input/filament-code")
src = next(p for p in code_src if os.path.isdir(os.path.join(p, "src")))
shutil.copytree(src, "/kaggle/working/code", dirs_exist_ok=True)
# offline install: no internet needed (wheel + pretrained encoder weights ship in the code dataset)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                *glob.glob("/kaggle/working/code/wheels/*.whl")], check=True)
os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_CACHE="/kaggle/working/code/hf_cache")
data = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026", recursive=True)[0]
env = dict(os.environ, FILAMENT_DATA=data, PYTHONPATH="/kaggle/working/code/src")
out = "/kaggle/working/runs"
jobs = {jobs!r}  # [(name, train_args), ...] run back to back in this one GPU session
for name, train_args in jobs:
    print(f"=== {{name}} {{train_args}}", flush=True)
    r = subprocess.run([sys.executable, "-m", "filaments.train", "--name", name, "--out-dir", out,
                        "--workers", "4", "--save-probs", *train_args], env=env, cwd="/kaggle/working/code")
    if r.returncode != 0:  # keep going: one failed experiment must not cost the rest
        print(f"=== {{name}} FAILED (exit {{r.returncode}})", flush=True)
        continue
    for run in sorted(glob.glob(f"{{out}}/{{name}}/fold*")):
        subprocess.run([sys.executable, "-m", "filaments.tune_post", run, "--workers", "4",
                        "--thresholds", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
                        "--min-areas", "0", "100", "200", "400", "800", "1600"], env=env,
                       cwd="/kaggle/working/code")
    print(f"=== {{name}} done", flush=True)
shutil.rmtree("/kaggle/working/code")
'''


def sh(*cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout


def bundle():
    d = WORK / "dataset"
    if d.exists():
        shutil.rmtree(d)
    (d / "filament-code").mkdir(parents=True)
    pkg = d / "filament-code"
    shutil.copytree(ROOT / "src", pkg / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (pkg / "data").mkdir()
    shutil.copy(ROOT / "data" / "folds.csv", pkg / "data" / "folds.csv")
    shutil.copytree(ROOT / "data" / "processed", pkg / "data" / "processed")
    shutil.copy(ROOT / "requirements.txt", pkg / "requirements.txt")
    # Kaggle kernels may have no internet: ship the smp wheel and the pretrained encoder weights.
    sh(str(ROOT / ".venv" / "bin" / "pip"), "download", "-q", "--no-deps", "-d", str(pkg / "wheels"),
       "segmentation-models-pytorch==0.5.0")
    for m in HF_CACHE.glob("models--smp-hub--*"):
        shutil.copytree(m, pkg / "hf_cache" / m.name, symlinks=False,  # resolve blob symlinks
                        ignore=shutil.ignore_patterns("blobs"))
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "filament-code", "id": CODE_DATASET, "licenses": [{"name": "CC0-1.0"}]}))
    exists = "filament-code" in sh(KAGGLE, "datasets", "list", "--mine")
    if exists:
        print(sh(KAGGLE, "datasets", "version", "-p", str(d), "-m", "update", "--dir-mode", "zip", "-q"))
    else:
        print(sh(KAGGLE, "datasets", "create", "-p", str(d), "--dir-mode", "zip", "-q"))  # private by default


def launch(name: str, args: list[str], jobs: list | None = None):
    """Push private GPU kernel `filament-<name>` running `jobs` (default: the single job name/args)."""
    jobs = jobs or [(name, args)]
    d = WORK / "kernels" / name
    d.mkdir(parents=True, exist_ok=True)
    slug = f"filament-{name.replace('_', '-')}"
    (d / "run.py").write_text(RUNNER.format(jobs=jobs))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [CODE_DATASET], "competition_sources": [COMPETITION], "kernel_sources": [],
    }, indent=2))
    out = sh(KAGGLE, "kernels", "push", "-p", str(d))
    print(out)
    if "error" in out.lower():  # the CLI exits 0 on e.g. "Maximum batch GPU session count reached"
        raise RuntimeError(out.strip())


def status(name: str) -> str:
    out = sh(KAGGLE, "kernels", "status", f"{USER}/filament-{name.replace('_', '-')}")
    print(out)
    return out


# Methodology experiments (EXPERIMENTS.md, notebook section 2.6).
COMMON = ["--fold", "0", "--epochs", "20", "--samples-per-epoch", "1000", "--val-every", "2"]
EXPERIMENTS = {
    "E01": ("e01_baseline", []),
    "E02": ("e02_scale1", ["--scale", "1.0"]),
    "E03": ("e03_scale025", ["--scale", "0.25"]),
    "E04": ("e04_focal_tversky", ["--loss", "focal:1,tversky:1"]),
    "E05": ("e05_soft", ["--label-mode", "soft"]),
    "E06": ("e06_noaug", ["--no-rotate", "--no-photometric"]),
    "E07": ("e07_scratch", ["--weights", "none"]),
    "E08": ("e08_unetpp", ["--arch", "unetplusplus"]),
    "E09": ("e09_effb3", ["--encoder", "efficientnet-b3"]),
    "E10": ("e10_pos04", ["--pos-frac", "0.4"]),
    # seed repeats of the baseline: run-to-run noise, needed to judge every difference above
    "E01S1": ("e01_baseline_s1", ["--seed", "1"]),
    "E01S2": ("e01_baseline_s2", ["--seed", "2"]),
    # optimisation hyperparameters
    "E11": ("e11_lr1e-4", ["--lr", "1e-4"]),
    "E12": ("e12_lr1e-3", ["--lr", "1e-3"]),
    "E13": ("e13_epochs40", ["--epochs", "40"]),
}


def batch(name: str, ids: list[str]):
    """Launch ONE kernel that runs several experiments sequentially (no local polling needed)."""
    launch(name, [], jobs=[(EXPERIMENTS[e][0], COMMON + EXPERIMENTS[e][1]) for e in ids])
MAX_GPU_SESSIONS = 2  # Kaggle limit per account


def queue(ids: list[str], poll: int = 60):
    """Run experiments keeping at most MAX_GPU_SESSIONS kernels busy; fetch each when it ends."""
    def log(msg):
        print(f"{time.strftime('%d %H:%M')} {msg}", flush=True)

    def done_locally(name):
        return (ROOT / "runs" / name / "fold0" / "summary.json").exists()

    pending = [e for e in ids if not done_locally(EXPERIMENTS[e][0])]
    running: dict[str, str] = {}
    while pending or running:
        # Kaggle API calls fail transiently; never let one failed call kill the queue.
        for eid, name in list(running.items()):
            try:
                st = status(name).lower()
                if any(k in st for k in ("complete", "error", "cancel")):
                    fetch(name)
                    log(f"{eid} {name} finished: {st.strip()}")
                    del running[eid]
            except (subprocess.CalledProcessError, OSError) as e:
                log(f"{eid} status/fetch failed, will retry: {e}")
        while pending and len(running) < MAX_GPU_SESSIONS:
            eid = pending[0]
            name, args = EXPERIMENTS[eid]
            try:  # a kernel from an earlier queue may be running or finished already
                st = status(name).lower()
                if any(k in st for k in ("running", "queued")):
                    pending.pop(0)
                    running[eid] = name
                    log(f"{eid} {name} attached")
                    continue
                if "complete" in st:
                    fetch(name)
                    if done_locally(name):
                        pending.pop(0)
                        log(f"{eid} {name} already complete, fetched")
                        continue
            except (subprocess.CalledProcessError, OSError):
                pass  # kernel does not exist yet (or status unavailable): launch it
            try:
                launch(name, COMMON + args)
            except (RuntimeError, subprocess.CalledProcessError) as e:
                log(f"launch of {eid} deferred: {e}")
                break
            pending.pop(0)
            running[eid] = name
            log(f"{eid} {name} launched")
        time.sleep(poll)
    log("queue empty")


def fetch(name: str):
    with tempfile.TemporaryDirectory() as tmp:
        print(sh(KAGGLE, "kernels", "output", f"{USER}/filament-{name.replace('_', '-')}", "-p", tmp))
        for src in sorted((Path(tmp) / "runs").glob("*")):  # one kernel may hold several runs
            shutil.copytree(src, ROOT / "runs" / src.name, dirs_exist_ok=True)
            print("fetched ->", ROOT / "runs" / src.name)
        log = next(Path(tmp).glob("*.log"), None)
        if log:
            (ROOT / "runs" / "kaggle_logs").mkdir(parents=True, exist_ok=True)
            shutil.copy(log, ROOT / "runs" / "kaggle_logs" / f"{name}.log")


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "bundle":
        bundle()
    elif cmd == "launch":
        name, extra = rest[0], rest[1:]
        launch(name, extra[1:] if extra[:1] == ["--"] else extra)
    elif cmd == "status":
        status(rest[0])
    elif cmd == "fetch":
        fetch(rest[0])
    elif cmd == "queue":
        queue(rest)
    elif cmd == "batch":
        batch(rest[0], rest[1:])
    else:
        sys.exit(__doc__)
