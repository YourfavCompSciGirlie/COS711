"""Generate the team's Colab notebook (COS_711_Assignment_2.ipynb) from the repository.

    .venv/bin/python tools/build_colab.py   ->  notebooks/COS_711_Assignment_2.ipynb

Follows the section structure the team agreed on (0 Setup ... 8 Conclusions). Every module in
src/filaments is embedded as a `%%writefile` cell under the section it belongs to, so the whole
pipeline is readable in one notebook while the code is still imported as one package (no copy-paste
drift). Re-run after changing src/ and paste the regenerated module cells into the shared notebook.
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "filaments"
OUT = ROOT / "notebooks" / "COS_711_Assignment_2.ipynb"

cells = []
written = set()


def deps(name: str) -> set[str]:
    """`name` and every package module it imports, transitively."""
    out, todo = set(), [name]
    while todo:
        m = todo.pop()
        if m not in out:
            out.add(m)
            todo += re.findall(r"^from \.(\w+) import", (SRC / f"{m}.py").read_text(), re.M)
    return out


def code(s: str):
    # On a fresh Colab runtime a module exists only once its %%writefile cell has run, so every import
    # (with its own imports) must come after those cells.
    if not s.startswith("%%writefile"):
        used = re.findall(r"^\s*from filaments\.(\w+) import", s, re.M)
        used += re.findall(r"^\s*!.*-m filaments\.(\w+)", s, re.M)
        used += [m for grp in re.findall(r"from filaments import ([\w ,]+)", s) for m in re.findall(r"(\w+)(?: as \w+)?", grp)
             if (SRC / f"{m}.py").exists()]
        missing = set().union(*[deps(m) for m in used]) - written
        assert not missing, f"cell imports {sorted(missing)} before they are written:\n{s[:200]}"
    cells.append(nbf.v4.new_code_cell(s.strip()))


md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))


def module(name: str):
    body = (SRC / name).read_text() or '"""Solar filament segmentation package."""\n'  # %%writefile needs a body
    written.add(name.removesuffix(".py"))
    code(f"%%writefile src/filaments/{name}\n{body}")


# ============================================================================ title
md("""
# COS 711 Assignment 2 · Segmenting solar filaments with deep learning

Kaggle: [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026),
ranked by **Panoptic Quality (PQ)**. Data: MAGFiLO (GONG H-alpha, 2048×2048, expert polygons).

## Group members and responsibilities

| Member | Student no. | Responsibility (marking category) |
|---|---|---|
| Malaika Kamangu | u23618583 | Format · Data preparation · Experimental setup & methodology |
| Stefan Muller | u22498622 | *Segmentation results* or *Segmentation enhancement* — fill in |
| Karabo | u23543087 | *Segmentation results* or *Segmentation enhancement* — fill in |

## Where each marking category is covered

| Rubric category | Marks | Notebook sections |
|---|---|---|
| Data preparation | 10 | 1 (EDA, preprocessing, augmentation, grouped split) |
| Experimental setup & methodology | 20 | 0.2–0.3, 2 (PQ, model, losses, training, ablations, tuning, 5-fold final model) |
| Segmentation results | 30 | 3 |
| Segmentation enhancement | 30 | 4 |
| Conclusions | 5 | 5 |
| Format | 5 | slides + video (outside this notebook) |

Sections follow the order of the presentation (spec §4): data → methodology → results →
enhancement → conclusions → bibliography. This notebook is the team's single source of code: every
module of the `filaments` package is written by a `%%writefile` cell in the section it belongs to.
""")

# ============================================================================ 0 setup
md("""
# 0. Setup & Imports

**How to use this notebook.** It runs in two places with the same code:

* **Google Colab** (Runtime → Change runtime type → **T4 GPU**): downloads the competition data with
  your Kaggle token and keeps the cache and runs on Google Drive.
* **Locally** (from the repository root or `notebooks/`): uses `data/MAGFiLO_1.0_Kaggle_2026` and
  `runs/`.

Two modes: `TRAIN = False` (default) analyses the team's trained runs in `RUNS_DIR` without a GPU;
`TRAIN = True` retrains (≈ 25 min per run on a T4). All outputs shown were produced by this notebook.

## 0.1 Imports
""")
code("""
import os, sys, json, time, random
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !pip install -q segmentation-models-pytorch==0.5.0 pycocotools kagglehub
import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2, torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
""")

md("""
## 0.2 Config / Constants

Global configuration. These are the defaults of `python -m filaments.train` and the protocol used
for every methodology experiment (one change per experiment, section 2.6). `scale = 0.5` means
the 2048² disk is seen by the network at 1024², chosen because the median filament is ≈ 8 px wide
(section 1.2): at 512² a third of filaments would be ≤ 2 px wide.
""")
code("""
CONFIG = dict(
    arch="unet", encoder="resnet34", weights="imagenet",   # model
    scale=0.5, crop=512, pos_frac=0.7,                     # data: resolution, crop size, filament-centred crops
    label_mode="separate",                                 # each annotator's set is its own sample
    loss="bce:1,dice:1",                                   # imbalance-robust combination
    epochs=20, samples_per_epoch=1000, batch_size=8,       # methodology budget (20 x 1000 crops)
    lr=3e-4, weight_decay=1e-4, warmup_epochs=1.0,         # AdamW + warm-up + cosine decay
    threshold=0.5, min_area=100,                           # post-processing during validation
    fold=0, seed=0,
)
TRAIN = False          # True: retrain runs here; False: analyse the team's trained runs in RUNS_DIR
def train_cmd(name, **overrides):
    c = CONFIG | overrides
    args = [f"--{k.replace('_', '-')} {v}" for k, v in c.items()]
    return f"python -m filaments.train --name {name} --workers 2 --save-probs --val-every 2 " + " ".join(args)
print(train_cmd("example"))
""")

md("""
## 0.3 Reproducibility (seeds)

Python, NumPy and PyTorch are seeded from `--seed` (`filaments.train.seed_everything`). Each
DataLoader worker draws crops from its own RNG derived from the seed, sample index and worker seed,
so crops differ across workers and epochs but a run is repeatable. Run-to-run variation is measured
explicitly by repeating the baseline with seeds 0, 1, 2 (section 2.6); the folds are deterministic
(seed 0, section 1.5).
""")
code("""
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
seed_everything(CONFIG["seed"])
""")

md("""
## 0.4 Device setup, data access & paths

* **Colab, Kaggle data**: add your Kaggle API token in Colab 🔑 *Secrets* as `KAGGLE_API_TOKEN` (the
  `KGAT_...` value) and enable notebook access. You must have joined the competition on Kaggle.
* **Colab, Google Drive** keeps the cache and training runs across sessions (`MyDrive/COS711_A2/`).
* **RUNS_DIR**: trained runs (copy the team's `runs/` folder there to analyse results without retraining).
""")
code("""
if IN_COLAB:
    from google.colab import userdata, drive
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    import kagglehub
    path = kagglehub.competition_download("filament-segmentation-2026")
    os.environ["FILAMENT_DATA"] = os.path.join(path, "MAGFiLO_1.0_Kaggle_2026")
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/COS711_A2"
else:   # local: repository root (data in data/MAGFiLO_1.0_Kaggle_2026)
    WORK = str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
for d in ["src/filaments", "tools", "data", "runs", "submissions", "reports/figures", "reports/eda", "reports/methodology"]:
    os.makedirs(f"{WORK}/{d}", exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, f"{WORK}/src")
os.environ["PYTHONPATH"] = f"{WORK}/src"
# Python caches each folder's file list and refreshes it only when the folder's mtime changes; Google
# Drive does not reliably update it, so modules written by later %%writefile cells would stay invisible.
import importlib
get_ipython().events.register("pre_run_cell", lambda info: importlib.invalidate_caches())
RUNS_DIR = f"{WORK}/runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("work dir:", WORK, "| device:", DEVICE)
""")

# ============================================================================ 1 data
md("""
# 1. Dataset Preparation
## 1.1 Data loading & inspection

Annotations are COCO-style polygons. Image ids look like `010401-20160920230134Lh`
(annotator batch + observation `YYYYMMDDHHMMSS` + GONG site). One observation can have up to three
independent **annotation sets**; Kaggle suggests treating them as different images, and every
split below is made over *observations* so that sets of one image never straddle train/validation.
""")
module("__init__.py")
module("data.py")
module("rle.py")
module("metrics.py")   # PQ implementation, explained in 2.1; the annotator-agreement analysis (1.2) uses it
module("analysis.py")
module("viz.py")
code("""
from filaments.data import (load_annotation_sets, parse_observation, make_folds, load_image, image_path,
                            fit_disk, TEST_IMAGES, SITES)
from filaments import analysis as A, viz
viz.setup()
EDA = Path("reports/eda")
summary = {}       # key EDA numbers, saved to reports/eda/summary.json at the end of section 1

sets = load_annotation_sets()
train_obs = sorted({s.observation for s in sets})
test_obs = sorted(p.stem for p in TEST_IMAGES.glob("*.jpeg"))
sets_per_obs = pd.Series([s.observation for s in sets]).value_counts()
cats = pd.Series([c for s in sets for c in s.categories]).map({1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"})
summary.update(train_observations=len(train_obs), test_observations=len(test_obs), annotation_sets=len(sets),
               filaments=sum(len(s.polygons) for s in sets),
               sets_per_observation=sets_per_obs.value_counts().sort_index().to_dict(),
               category_counts=cats.value_counts().to_dict())
summary
""")
code("""
meta = pd.DataFrame([{"observation": o, "split": sp, "site": o[14:16], "year": int(o[:4]),
                      "time": parse_observation(o)[0]}
                     for sp, lst in [("train", train_obs), ("test", test_obs)] for o in lst])
t_train = np.array([t.timestamp() for t in meta.query("split == 'train'").time])
gap_h = np.array([np.abs(t_train - t.timestamp()).min() / 3600 for t in meta.query("split == 'test'").time])
summary["test_to_nearest_train_hours"] = {"median": float(np.median(gap_h)), "lt_24h": int((gap_h < 24).sum()),
                                          "lt_7d": int((gap_h < 168).sum())}

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [2.2, 1.2, 1.4]})
for ax, key in zip(axes[:2], ["year", "site"]):
    tab = meta.pivot_table(index=key, columns="split", values="observation", aggfunc="count").fillna(0)
    x = np.arange(len(tab)); w = 0.4
    ax.bar(x - w/2, tab["train"], w * 0.92, color=viz.SEQ, label="train")
    ax.bar(x + w/2, tab["test"], w * 0.92, color=viz.ACCENT, label="test")
    ax.set_xticks(x, tab.index); ax.set_title(f"Observations per {key}"); ax.grid(axis="x", visible=False)
axes[0].legend()
axes[2].hist(gap_h / 24, bins=np.arange(0, 30, 1), color=viz.ACCENT, rwidth=0.9)
axes[2].set_title("Test image → nearest train image"); axes[2].set_xlabel("days")
fig.tight_layout(); viz.save(fig, "eda_inventory"); plt.show()
summary["test_to_nearest_train_hours"]
""")
md("""
**Reading.** Train and test cover the same years and all six GONG sites in similar proportions. The
test set is *interleaved in time* with train (many test images are within a day of a train image),
so the public leaderboard rewards near-duplicate days. Our validation must not: nearby observations
are not independent, so folds are built from 27-day blocks (one solar rotation, section 1.5). Expect
local PQ to be somewhat *pessimistic* relative to the leaderboard. The `category_id` labels
(Left/Right/Unidentifiable) describe chirality and are not needed: all filaments are one class.
""")

md("""
## 1.2 Data analysis / EDA

The per-image and per-filament tables are computed once and cached in `reports/eda/`
(≈ 10 min on Colab the first time; instant afterwards).
""")
code("""
def cached(name, fn):
    p = EDA / f"{name}.csv"
    if p.exists():
        return pd.read_csv(p)
    df = fn(); df.to_csv(p, index=False); return df

quality = cached("image_quality", lambda: A.image_quality_table(train_obs + test_obs, workers=os.cpu_count()))
quality["split"] = np.where(quality.observation.isin(train_obs), "train", "test")
disks = {r.observation: (r.cx, r.cy, r.r) for r in quality.itertuples()}
def _fil():
    fil, per_set = A.filament_tables(sets, disks)
    per_set.to_csv(EDA / "annotation_sets.csv", index=False)
    return fil
fil = cached("filaments", _fil)
per_set = pd.read_csv(EDA / "annotation_sets.csv")
agree = cached("annotator_agreement", lambda: A.agreement_table(sets))
print(f"{len(quality)} images | {len(fil)} filaments | {len(per_set)} annotation sets | {len(agree)} annotator pairs")
""")
md("### What the images and labels look like")
code("""
rng = np.random.default_rng(3)
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
for ax, site in zip(axes.flat, viz.SITE_ORDER):
    cand = [s for s in sets if s.site == site and len(s.polygons) >= 6]
    s = cand[rng.integers(len(cand))]
    img = load_image(image_path(s.observation))
    ov = A.overlay(img, s.instance_masks(), [viz.hex_to_rgb(viz.ACCENT)], thickness=7)
    ax.imshow(cv2.resize(ov, (768, 768), interpolation=cv2.INTER_AREA))
    ax.set_title(f"{SITES[site]} ({site}) · {s.observation[:8]} · {len(s.polygons)} filaments", fontsize=9)
    viz.no_axes(ax)
fig.tight_layout(); viz.save(fig, "eda_samples_per_site"); plt.show()
summary["disk"] = quality[["cx", "cy", "r"]].describe().loc[["mean", "std", "min", "max"]].round(2).to_dict()
quality[["cx", "cy", "r"]].describe().round(2)
""")
md("""
Filaments are small, dark, elongated features scattered over the disk; most of the image is
background. Each frame also contains a grey **off-limb halo** (with bright prominences) whose outer
edge is irregular and often off-centre: a naive threshold finds the halo, not the Sun. Fitting the
limb itself (`fit_disk`, section 1.3) shows GONG frames are **already centred and scaled**: centre
within ±2 px of (1023, 1024), radius 901–906 px across all 887 images. No re-centring or rescaling
is needed, and a fixed disk mask suppresses the halo (filaments are on-disk features).
""")

md("### Annotator agreement: the realistic ceiling")
code("""
summary["annotator_agreement"] = {
    "pairs": len(agree), "observations": int(agree.observation.nunique()),
    "pq_mean": float(agree.pq.mean()), "pq_median": float(agree.pq.median()),
    "dice_mean": float(agree.dice.mean()), "sq_mean": float(agree.sq.mean()), "rq_mean": float(agree.rq.mean()),
    "count_mismatch_mean": float((agree.n_a - agree.n_b).abs().mean()),
}
# An observation with 3 annotation sets whose agreement is close to the median (representative).
tri = agree.groupby("observation").filter(lambda g: len(g) == 3)
med = tri.groupby("observation").pq.mean()
obs = (med - agree.pq.median()).abs().idxmin()
group = [s for s in sets if s.observation == obs]
img = load_image(image_path(obs))
masks = {s.set_id: s.instance_masks() for s in group}
union = np.any([m for ms in masks.values() for m in ms], axis=0)
ys, xs = np.nonzero(union); cy, cx = int(np.median(ys)), int(np.median(xs)); H = 380
y0, x0 = max(cy - H, 0), max(cx - H, 0); crop = np.s_[y0:y0 + 2 * H, x0:x0 + 2 * H]

fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})
axes[0].imshow(img[crop], cmap="gray"); axes[0].set_title("H-alpha crop"); viz.no_axes(axes[0])
for ax, s, col in zip(axes[1:4], group, viz.SERIES):
    ax.imshow(A.overlay(img[crop], [m[crop] for m in masks[s.set_id]], [viz.hex_to_rgb(col)], thickness=2))
    ax.set_title(f"annotator {s.set_id.split('-')[0]} · {len(s.polygons)} filaments", fontsize=9); viz.no_axes(ax)
ax = axes[4]
ax.hist(agree.pq, bins=np.linspace(0, 1, 21), color=viz.SEQ, rwidth=0.9, label=f"PQ (median {agree.pq.median():.2f})")
ax.hist(agree.dice, bins=np.linspace(0, 1, 21), histtype="step", color=viz.ACCENT, linewidth=2,
        label=f"Dice, semantic (median {agree.dice.median():.2f})")
ax.set_title(f"Annotator vs annotator ({len(agree)} pairs)"); ax.set_xlabel("score"); ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); viz.save(fig, "eda_annotator_agreement"); plt.show()
agree[["pq", "sq", "rq", "dice"]].describe().T[["mean", "50%", "std"]].round(3)
""")
md("""
**Reading.** Two independent experts score a mean PQ of only ≈ 0.34 against each other (matching the
public Kaggle notebook *"Filament PQ: Two Experts Agree at 0.34"*, which also validates our PQ
implementation), while semantic Dice is ≈ 0.61. Annotators agree *where* filament material is but
disagree on *how to cut it into instances* (one long filament vs several pieces; which faint segments
count). Consequences: labels are **noisy at the instance level**, which caps achievable PQ; all
annotation sets are kept as **separate training samples** (label-noise augmentation; all sets of an
observation share a fold); post-processing is tuned on validation PQ, because the instance
granularity is a convention learnt from the annotators.
""")

md("### Filament geometry: scale, thinness, crowding, class imbalance")
code("""
f = fil.query("area > 0")
summary["filament_geometry"] = {
    "area_px_median": float(f.area.median()),
    "width_median_px_median": float(f.width_median.median()),
    "width_median_px_p10_p90": f.width_median.quantile([.1, .9]).tolist(),
    "frac_width_lt_8px": float((f.width_median < 8).mean()),
    "solidity_median": float(f.solidity.median()),
    "frac_gap_lt_10px": float((f.gap_to_nearest < 10).mean()),
    "frac_gap_lt_25px": float((f.gap_to_nearest < 25).mean()),
    "frac_touching_other": float(f.touches_other.mean()),
    "filaments_per_set_median": float(per_set.n_filaments.median()),
    "filament_pixel_frac_of_disk_median": float(per_set.filament_frac.median()),
    "filament_pixel_frac_of_disk_p90": float(per_set.filament_frac.quantile(.9)),
}
fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
ax = axes[0]
ax.hist(np.log10(f.area), bins=40, color=viz.SEQ, rwidth=0.9)
ax.set_title("Filament area"); ax.set_xlabel("pixels (log10) at 2048²")
ax = axes[1]
ax.hist(f.width_median, bins=np.arange(0, 32, 1), color=viz.SEQ, rwidth=0.9)
for d, ls, lab in [(2, "--", "2 px wide after ÷2 (1024²)"), (4, ":", "2 px wide after ÷4 (512²)")]:
    ax.axvline(2 * d, color=viz.TEXT, linestyle=ls, linewidth=1.2, label=lab)
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Median filament width"); ax.set_xlabel("pixels at 2048²")
ax = axes[2]
d = np.arange(0, A.GAP_CAP + 1)
ax.plot(d, [(f.gap_to_nearest < x).mean() * 100 for x in d], color=viz.SEQ)
ax.set_ylim(0, None)
ax.set_title("Filaments with a neighbour closer than d"); ax.set_xlabel("d (pixels at 2048²)"); ax.set_ylabel("% of filaments")
ax = axes[3]
ax.hist(per_set.filament_frac * 100, bins=40, color=viz.ACCENT, rwidth=0.9)
ax.set_title("Filament pixels per image"); ax.set_xlabel("% of solar disk")
fig.tight_layout(); viz.save(fig, "eda_filament_geometry"); plt.show()
pd.Series(summary["filament_geometry"])
""")
md("""
**Reading.**
* **Thin structures.** Width = 2 × distance-transform value along the skeleton. The median filament
  is ≈ 8 px wide (10–90 %: ≈ 6–16 px) at full resolution and a third are under 8 px. At 1024² those
  become < 4 px; at 512² they shrink to ≈ 2 px and fine barbs vanish → **input ≥ 1024²** (tested in E02/E03).
* **Crowding is rare; fragmentation is the instance problem.** Only ≈ 3 % of filaments have a
  neighbour within 10 px and touching polygons are ≈ 0.1 %, so connected components rarely merge two
  distinct filaments. The bigger risk is one filament predicted as a chain of separate blobs
  (**one-to-many** errors) → connectivity-aware losses and fragment merging (section 4).
* **Low solidity** (median ≈ 0.58): elongated, curved, branched shapes that bounding boxes fit poorly,
  so a box-based instance model (Mask R-CNN) is a poor fit; we use semantic segmentation + instance extraction.
* **Severe class imbalance.** Filament pixels are a median ≈ 0.5 % of the disk (≈ 1:200) → overlap
  losses (Dice/Tversky) or focal BCE and crops oversampled around filaments.
""")

md("### Image quality across sites (noisy samples)")
code("""
prof_cols = [c for c in quality.columns if c.startswith("prof_")]
rr = (np.arange(len(prof_cols)) + 0.5) / len(prof_cols)
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
ax = axes[0]
for site in viz.SITE_ORDER:
    p = quality.loc[quality.site == site, prof_cols].median()
    ax.plot(rr, p.values, color=viz.SITE_COLOR[site], label=f"{site} {SITES[site]}")
ax.set_title("Limb darkening (median radial profile)"); ax.set_xlabel("r / R_sun"); ax.set_ylabel("intensity"); ax.legend(fontsize=8)
for ax, col, title in [(axes[1], "sharpness", "Sharpness (Laplacian variance)"),
                       (axes[2], "nonuniformity", "Large-scale non-uniformity (clouds/haze)")]:
    data = [quality.loc[quality.site == s, col] for s in viz.SITE_ORDER]
    bp = ax.boxplot(data, tick_labels=viz.SITE_ORDER, patch_artist=True, widths=0.55, showfliers=True,
                    medianprops={"color": viz.TEXT}, flierprops={"markersize": 3, "markeredgecolor": viz.TEXT_2})
    for patch, s in zip(bp["boxes"], viz.SITE_ORDER):
        patch.set_facecolor(viz.SITE_COLOR[s]); patch.set_edgecolor(viz.SURFACE)
    ax.set_title(title); ax.grid(axis="x", visible=False)
axes[1].set_yscale("log")
fig.tight_layout(); viz.save(fig, "eda_quality_by_site"); plt.show()
quality.groupby("site")[["median", "iqr", "sharpness", "nonuniformity", "r"]].median().round(3)
""")
code("""
# Outliers: robust z-score per metric over all images; show the worst offenders.
def robust_z(x):
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / mad
quality["z_blur"] = -robust_z(np.log(quality.sharpness))
quality["z_cloud"] = robust_z(quality.nonuniformity)
flags = quality[(quality.z_blur > 3) | (quality.z_cloud > 3)]
summary["quality_outliers"] = {"blurry": int((quality.z_blur > 3).sum()), "nonuniform": int((quality.z_cloud > 3).sum()),
                               "train": int((flags.split == "train").sum()), "test": int((flags.split == "test").sum())}
worst = pd.concat([quality.nlargest(3, "z_blur").assign(why="blurry"),
                   quality.nlargest(3, "z_cloud").assign(why="non-uniform")])
fig, axes = plt.subplots(1, 6, figsize=(18, 3.4))
for ax, r in zip(axes, worst.itertuples()):
    ax.imshow(cv2.resize(load_image(image_path(r.observation)), (512, 512), interpolation=cv2.INTER_AREA), cmap="gray")
    ax.set_title(f"{r.why} · {r.split}\\n{r.observation}", fontsize=8); viz.no_axes(ax)
fig.tight_layout(); viz.save(fig, "eda_quality_outliers"); plt.show()
summary["quality_outliers"]
""")
md("""
**Reading.** Median disk brightness is similar across sites, but sharpness and especially
large-scale non-uniformity (haze, cloud gradients) differ by site: El Teide (Th) is the most uniform,
Cerro Tololo (Ch) the sharpest. Robust outlier flags (|z| > 3) mark a few dozen images as unusually
blurry or non-uniform, in **both train and test**, so they are **kept**, and the model is made robust
to them instead: limb-darkening flattening + disk-wise normalisation (1.3) remove the site-to-site
brightness and centre-to-limb trend, and photometric augmentation (contrast, brightness, blur, noise;
1.4) covers the rest.

### Summary → decisions

| Finding | Decision |
|---|---|
| Disk is fixed (centre ±2 px, r ≈ 903 px); off-limb halo is irregular | edge-based disk fit, pixels beyond 0.99 R set to 0 |
| Median filament ≈ 8 px wide, ⅓ < 8 px | network input ≥ 1024² (`scale ≥ 0.5`), never whole-disk 512² |
| Filaments ≈ 0.5 % of disk pixels (≈ 1:200) | Dice/Tversky/focal losses; 70 % of crops centred on a filament |
| Neighbours rarely touch; experts split/merge differently | instance = connected component; tune post-processing on PQ; fragment merging (section 4) |
| Expert-vs-expert PQ ≈ 0.34 | realistic ceiling; noisy instance labels → keep all annotation sets |
| Site-dependent haze/blur/contrast; outliers in train and test | limb flattening + robust normalisation; photometric augmentation |
| Test interleaved in time with train | validation by 27-day blocks (honest, slightly pessimistic) |
""")

md("""
## 1.3 Preprocessing

1. **Disk fit** (`fit_disk`): steepest radial intensity drop along 360 rays within ±6 % of the
   nominal GONG limb + robust circle fit (3 refits dropping outlier rays). A naive threshold finds the
   off-limb halo instead of the limb (it gave a wrong radius on 160 images in our first version).
2. **Limb-darkening flattening**: divide by the image's median radial profile (256 radial bins).
3. **Normalisation**: `x = (flat − 1) / robust std` → filaments negative, plage positive, one scale for
   all sites; pixels beyond 0.99 R (halo + JPEG limb ring) set to 0, values clipped to ±8.
4. **Cache**: disk geometry, radial profiles and rasterised instance label maps (`data/processed/`,
   21 MB), so training reads only the JPEG and the cache.
""")
module("preprocess.py")
code("""
from filaments.preprocess import build_cache, CACHE_DIR, ProfileStore, full_image
build_cache(workers=os.cpu_count())          # ~5 min once; skipped when the cache exists
ps = ProfileStore()
obs = train_obs[0]
raw = load_image(image_path(obs)); disk, prof, spread = ps(obs)
x = full_image(raw, disk, prof, spread)
cx, cy, r = disk[:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(raw, cmap="gray"); axes[0].add_patch(plt.Circle((cx, cy), r, fill=False, color=viz.ACCENT, lw=1.5))
axes[0].set_title(f"raw JPEG + fitted limb (r = {r:.1f} px)")
axes[1].plot(np.linspace(0, 1, len(prof)), prof, color=viz.SEQ); axes[1].set_title("radial profile (limb darkening)"); axes[1].set_xlabel("r / R")
axes[2].imshow(x, cmap="gray", vmin=-4, vmax=4); axes[2].set_title("normalised network input")
for a in (axes[0], axes[2]): viz.no_axes(a)
fig.tight_layout(); viz.save(fig, "prep_pipeline"); plt.show()
""")

md("""
## 1.4 Augmentation

One affine warp per crop combines the resolution change, free rotation (a solar disk has no preferred
orientation), flip, ±10 % zoom and translation; normalisation is evaluated only at the sampled pixels
(fast, no 2048² intermediate). Photometric augmentation (contrast, brightness, blur ≈ poor seeing,
Gaussian noise) covers the site-to-site variation found in 1.2. 70 % of crops are centred on a
random filament (class imbalance). Whether augmentation helps is tested in E06.
""")
module("dataset.py")
code("""
from filaments.dataset import CropDataset, CropConfig
ds = CropDataset(sets, CropConfig(scale=CONFIG["scale"], crop=CONFIG["crop"]), samples_per_epoch=8)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i in range(4):
    b = ds[i]
    axes[0, i].imshow(b["image"][0], cmap="gray", vmin=-4, vmax=4); viz.no_axes(axes[0, i])
    axes[1, i].imshow(b["mask"][0], cmap="gray"); viz.no_axes(axes[1, i])
axes[0, 0].set_title("augmented crops (512², input scale 0.5)"); axes[1, 0].set_title("targets")
fig.tight_layout(); viz.save(fig, "prep_augmented_crops"); plt.show()
""")

md("""
## 1.5 Train/validation split

5 folds over **observations**, grouped into **27-day blocks** (≈ one solar rotation): all annotation
sets of an observation, and the same filament seen on consecutive days, always land in the same fold.
Blocks are distributed greedily to balance fold sizes. Deterministic (seed 0). All methodology
comparisons use fold 0; the final configuration is trained on all 5 folds (section 2.9).
""")
code("""
fold_of = make_folds(train_obs, n_folds=5, days=27, seed=0)
folds = meta.query("split == 'train'")[["observation", "site", "year", "time"]].copy()
folds["fold"] = folds.observation.map(fold_of)
folds["n_sets"] = folds.observation.map(sets_per_obs)
folds.to_csv("data/folds.csv", index=False)

# Sanity: every observation in exactly one fold; how many same-day neighbours fall across a block edge.
assert folds.observation.is_unique
t_sorted = folds.sort_values("time")
close = (t_sorted.time.diff().dt.total_seconds() < 86400)
summary["folds"] = {
    "observations_per_fold": folds.fold.value_counts().sort_index().tolist(),
    "annotation_sets_per_fold": folds.groupby("fold").n_sets.sum().tolist(),
    "consecutive_same_day_pairs_split_across_folds": int((close & (t_sorted.fold.diff() != 0)).sum()),
}
tab = folds.pivot_table(index="fold", columns="site", values="observation", aggfunc="count").reindex(columns=viz.SITE_ORDER).fillna(0)
fig, ax = plt.subplots(figsize=(7, 3.6))
bottom = np.zeros(len(tab))
for s in viz.SITE_ORDER:
    ax.bar(tab.index, tab[s], 0.6, bottom=bottom, color=viz.SITE_COLOR[s], label=s, edgecolor=viz.SURFACE, linewidth=2)
    bottom += tab[s].values
ax.set_title("Observations per fold, by site"); ax.set_xlabel("fold"); ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax.grid(axis="x", visible=False)
fig.tight_layout(); viz.save(fig, "eda_folds"); plt.show()
(EDA / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
summary["folds"]
""")
md("""
A handful of same-day pairs fall either side of a block boundary (count above): a small, accepted
leak. Site mix per fold stays close to the overall mix.
""")

# ============================================================================ 2 methodology
md("""
# 2. Experimental Setup & Methodology

**Tools.** PyTorch 2, segmentation_models_pytorch 0.5 (U-Net/U-Net++ with ImageNet encoders),
pycocotools (RLE, IoU), OpenCV, NumPy/pandas, Matplotlib.
**Hardware.** All reported runs: Kaggle NVIDIA T4 (16 GB) through private GPU kernels driven by
`tools/kaggle_run.py` (2.5); analysis on Colab / a laptop (Apple M-series, MPS).

**Protocol** (every comparison): validation fold 0 (141 observations, 235 annotation sets), 20 epochs
× 1000 random 512² crops, batch 8, checkpoint selected by validation PQ, post-processing tuned on the
saved out-of-fold probabilities, **one change per experiment**, baseline repeated with 3 seeds.

## 2.1 Evaluation metric: local Panoptic Quality
### IoU and matching

PQ (Kirillov et al., 2019) matches a predicted and a ground-truth filament when IoU > 0.5. Above 0.5
a segment can overlap at most one segment of the other set, so matching is unique: no Hungarian
assignment needed. IoUs are computed on COCO RLEs with `pycocotools` (fast at 2048²).
""")
md("The implementation is `src/filaments/metrics.py`, written in section 1.1 because the annotator-agreement analysis (1.2) already uses it.")
md("""
### PQ computation

`PQ = Σ_TP IoU / (|TP| + ½|FP| + ½|FN|) = SQ × RQ`. Reported both dataset-level (`pq_global`,
headline: sums of TP/FP/FN/IoU over all validation images) and as the mean of per-image PQ, plus
one-to-many (fragmentation) and many-to-one (over-merging) counts, which the competition also
inspects. Each annotation set of a validation image is scored as its own ground truth.
""")
code("""
from filaments.metrics import image_stats, aggregate, dice
def box(y0, y1, x0, x1, n=64):
    m = np.zeros((n, n), bool); m[y0:y1, x0:x1] = True; return m
s = image_stats([box(0, 10, 0, 10)], [box(0, 10, 0, 8), box(50, 60, 50, 60)])
print(f"one match (IoU 0.8) + one FP -> PQ {s.pq:.3f} (expected {0.8 / 1.5:.3f}), SQ {s.sq:.2f}, RQ {s.rq:.3f}")
""")
md("""
### Sanity tests

Unit tests of the edge cases PQ must punish, plus a check on real data: our implementation gives
expert-vs-expert PQ ≈ 0.34 (section 1.2), matching the public Kaggle notebook *"Filament PQ: Two
Experts Agree at 0.34"*.
""")
code("""
from filaments import rle
g = [box(0, 10, 0, 10), box(20, 40, 20, 40)]
assert np.isclose(image_stats(g, g).pq, 1.0), "perfect prediction"
s = image_stats([box(0, 10, 0, 40)], [box(0, 10, 0, 20), box(0, 10, 20, 40)])
assert s.tp == 0 and s.fp == 2 and s.fn == 1 and s.one_to_many == 1, "fragmentation is penalised"
s = image_stats([box(0, 10, 0, 10), box(0, 10, 30, 40)], [box(0, 10, 0, 40)])
assert s.tp == 0 and s.many_to_one == 1, "over-merging is penalised"
assert image_stats([], []).fn == 0 and image_stats(g, []).fn == 2, "empty cases"
m = np.zeros((2048, 2048), bool); m[100:300, 50:900] = True
assert (rle.decode(rle.encode(m)) == m).all(), "RLE round trip (submission format)"
assert abs(agree.pq.mean() - 0.34) < 0.02, "expert-vs-expert PQ matches the public reference"
print("all PQ / RLE sanity tests passed")
""")

md("""
## 2.2 Problem formulation & architecture

**Formulation:** semantic segmentation (filament probability per pixel) followed by **instance
extraction** (threshold → 8-connected components → minimum area). Section 1.2 motivates it: filaments
rarely touch (components separate them), and their elongated, branched shapes fit bounding boxes
poorly (box-based Mask R-CNN). The remaining instance errors are handled in section 4.

**Baseline:** U-Net (Ronneberger et al., 2015) with an ImageNet-pretrained ResNet-34 encoder
(segmentation_models_pytorch). The single grayscale channel is handled by summing the pretrained RGB
filters of the first convolution (no fake RGB). Compared alternatives: U-Net++, EfficientNet-B3
encoder, no pre-training (2.6).
""")
module("models.py")
code("""
from filaments.models import build_model, count_params, size_mb
m = build_model(CONFIG["arch"], CONFIG["encoder"], CONFIG["weights"])
print(f"{CONFIG['arch']} / {CONFIG['encoder']}: {count_params(m) / 1e6:.1f} M parameters, {size_mb(m):.1f} MB")
""")
md("""
## 2.3 Loss functions

Combined as a weighted-sum string, e.g. `"bce:1,dice:1"`. Overlap losses are computed over the whole
batch so crops without filaments are well defined. BCE gives stable per-pixel gradients, Dice
counters the 1:200 imbalance; focal + Tversky (β > α, favours recall of faint/thin structure) is the
imbalance-focused alternative (E04); clDice (soft skeletons) rewards connectivity (section 4).
""")
module("losses.py")
code("""
from filaments.losses import Combined
logits = torch.randn(2, 1, 64, 64); target = (torch.rand(2, 1, 64, 64) > 0.95).float()
for spec in ["bce:1,dice:1", "focal:1,tversky:1", "bce:1,dice:1,cldice:0.5"]:
    total, parts = Combined(spec)(logits, target)
    print(f"{spec:28s} total {total.item():.3f}  {parts}")
""")
md("""
## 2.4 Optimiser, scheduler, training loop

AdamW (lr 3e-4, weight decay 1e-4), linear warm-up for 1 epoch then cosine decay, mixed precision on
CUDA. Validation every 2 epochs on full disks of the validation fold with PQ; the checkpoint with the
best validation PQ is kept. Each run records parameters, size, training time and inference time, and
saves out-of-fold probability maps so post-processing can be tuned without retraining.
""")
module("postprocess.py")
module("evaluate.py")
module("train.py")

md("""
## 2.5 Running the experiments

Each experiment is one call of `filaments.train` (command built from `CONFIG`, 0.2). Colab's free GPU
time is too short for 15+ runs, so the runs were executed as **private Kaggle GPU kernels**:
`tools/kaggle_run.py` uploads this code + the preprocessing cache as a private dataset, and launches
kernels that train, save out-of-fold probabilities and tune post-processing (`batch` runs several
experiments back to back; at most 2 GPU sessions at a time). `EXPERIMENTS` below is the full
registry of what was run.
""")
code(f"%%writefile tools/kaggle_run.py\n{(ROOT / 'tools' / 'kaggle_run.py').read_text()}")
code("""
RUN = "e01_baseline"
if TRAIN:   # same run on the current GPU (≈ 25 min on a T4)
    !{train_cmd(RUN)} --out-dir "{RUNS_DIR}"
""")

md("""
## 2.6 Hyperparameter tuning & ablations

Two levels:
1. **Post-processing** (threshold, minimum instance area): grid-searched on out-of-fold
   probabilities (`filaments.tune_post`), never on the leaderboard.
2. **Model / data / optimisation**: one change per experiment against the baseline, same budget and
   fold. The baseline is repeated with 3 seeds; its standard deviation is the noise floor every other
   difference is judged against.

| ID | Change vs baseline | Question (motivated by) |
|---|---|---|
| E01 (+S1, S2) | none (seeds 0, 1, 2) | baseline and run-to-run noise |
| E02 / E03 | scale 1.0 / 0.25 | resolution vs thin filaments (width ≈ 8 px) |
| E04 | focal + Tversky | imbalance-aware loss (≈ 1:200) |
| E05 | soft consensus labels | annotator noise (expert PQ ≈ 0.34) |
| E06 | no augmentation | value of augmentation (site variation) |
| E07 | no ImageNet pre-training | transfer learning with 707 images |
| E08 / E09 | U-Net++ / EfficientNet-B3 | architecture / encoder |
| E10 | 40 % instead of 70 % filament-centred crops | sampling vs imbalance |
| E11 / E12 | lr 1e-4 / 1e-3 | learning rate |
| E13 | 40 epochs | training length |
""")
module("tune_post.py")
code("""
if TRAIN:
    !python -m filaments.tune_post "{RUNS_DIR}/{RUN}/fold0" --workers 2 --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --min-areas 0 100 200 400 800 1600
""")
code("""
CHANGE = {
    "e01": "Baseline: U-Net R34, 1024² disk, BCE+Dice", "e02": "Full resolution (2048²)",
    "e03": "Low resolution (512²)", "e04": "Focal + Tversky loss", "e05": "Soft consensus labels",
    "e06": "No augmentation", "e07": "No ImageNet pre-training", "e08": "U-Net++",
    "e09": "EfficientNet-B3 encoder", "e10": "Fewer filament-centred crops (40 %)",
    "e11": "Learning rate 1e-4", "e12": "Learning rate 1e-3", "e13": "40 epochs (2x budget)",
}
METH = Path("reports/methodology")
rows, hist = [], {}
for run in sorted(Path(RUNS_DIR).glob("e[0-9][0-9]_*/fold0")):
    if not (run / "summary.json").exists():
        continue
    eid = run.parent.name[:3]; tail = run.parent.name.rsplit("_s", 1)
    seed = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 0
    s = json.loads((run / "summary.json").read_text()); c = json.loads((run / "config.json").read_text())
    pb = json.loads((run / "post_best.json").read_text())
    best = pd.read_csv(run / "post_grid.csv").iloc[0]
    rows.append({"id": eid.upper(), "seed": seed, "run": run.parent.name, "change": CHANGE[eid],
                 "pq_default": s["val_pq_global"], "pq_tuned": pb["pq_global"],
                 "threshold": pb["threshold"], "min_area": pb["min_area"],
                 "sq_tuned": best.sq, "rq_tuned": best.rq, "dice": s["val_dice_mean"],
                 "one_to_many": best.one_to_many, "many_to_one": best.many_to_one, "best_epoch": s["best_epoch"],
                 "params_M": s["params"] / 1e6, "size_MB": s["size_mb"], "train_min": s["train_sec"] / 60,
                 "sec_per_img": s["val_sec_per_image_model"] + s["val_sec_per_image_post"], "device": c["device"]})
    if seed == 0:
        hist[eid.upper()] = pd.read_csv(run / "history.csv")
res = pd.DataFrame(rows)
res.to_csv(METH / "results_fold0.csv", index=False)

# One row per experiment: mean ± std over seeds (only the baseline is repeated; its std is the noise floor).
metrics = ["pq_default", "pq_tuned", "sq_tuned", "rq_tuned", "dice", "one_to_many", "many_to_one", "train_min", "sec_per_img"]
agg = res.groupby(["id", "change"])[metrics].agg(["mean", "std"])
agg.columns = [f"{m}_{s}" for m, s in agg.columns]
agg = agg.join(res.groupby("id").size().rename("n_seeds"), on="id").reset_index()
agg.to_csv(METH / "results_fold0_summary.csv", index=False)
NOISE = float(agg.loc[agg.id == "E01", "pq_tuned_std"].iloc[0])
print(f"{len(res)} runs | baseline run-to-run std of tuned PQ: {NOISE:.4f}")
(agg.sort_values("pq_tuned_mean", ascending=False)
    [["id", "change", "n_seeds", "pq_tuned_mean", "pq_tuned_std", "pq_default_mean", "dice_mean",
      "one_to_many_mean", "many_to_one_mean", "train_min_mean"]].round(3))
""")
md("### Effect of each change (baseline ± 1 std band = run-to-run noise)")
code("""
# Dot plot (not bars): the x-axis is truncated, and bars from a non-zero baseline exaggerate gaps.
order = agg.sort_values("pq_tuned_mean").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(order) + 1.3))
b = float(order.loc[order.id == "E01", "pq_tuned_mean"].iloc[0])
ax.axvspan(b - NOISE, b + NOISE, color=viz.ACCENT, alpha=0.12, linewidth=0, label="baseline ± 1 std (3 seeds)")
ax.axvline(b, color=viz.TEXT_2, linestyle="--", linewidth=1)
for y, r in order.iterrows():
    col = viz.ACCENT if r.id == "E01" else viz.SEQ
    err = r.pq_tuned_std if r.n_seeds > 1 else 0
    if err:
        ax.errorbar(r.pq_tuned_mean, y, xerr=err, fmt="none", ecolor=col, elinewidth=1.5, capsize=4)
    ax.plot(r.pq_tuned_mean, y, "o", color=col, markersize=8, markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    ax.text(r.pq_tuned_mean + err + 0.003, y, f"{r.pq_tuned_mean:.3f}", va="center", fontsize=8, color=viz.TEXT_2)
ax.set_yticks(range(len(order)), order.id + "  " + order.change)
ax.set_xlim(max(0, (order.pq_tuned_mean - order.pq_tuned_std.fillna(0)).min() - 0.02),
            (order.pq_tuned_mean + order.pq_tuned_std.fillna(0)).max() + 0.025)
ax.set_xlabel("validation PQ (fold 0, tuned post-processing)")
ax.set_title("Methodology ablations: one change vs the baseline")
ax.legend(loc="lower right", fontsize=8); ax.grid(axis="y", visible=False)
fig.tight_layout(); viz.save(fig, "meth_ablation_pq"); plt.show()
""")
md("### Post-processing sensitivity (baseline, fold 0)")
code("""
grid = pd.read_csv(Path(RUNS_DIR) / "e01_baseline" / "fold0" / "post_grid.csv").pivot(index="min_area", columns="threshold", values="pq_global")
fig, ax = plt.subplots(figsize=(7, 3.6))
im = ax.imshow(grid.values, cmap="Blues", aspect="auto", origin="lower")
ax.set_xticks(range(grid.shape[1]), [f"{t:.1f}" for t in grid.columns]); ax.set_yticks(range(grid.shape[0]), grid.index)
for (r, c), v in np.ndenumerate(grid.values):
    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
            color="white" if v > np.nanmax(grid.values) * 0.75 else viz.TEXT)
ax.set_xlabel("probability threshold"); ax.set_ylabel("min instance area (px)"); ax.grid(False)
ax.set_title("Baseline PQ vs post-processing parameters")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); viz.save(fig, "meth_post_grid"); plt.show()
""")
md("""
**Conclusions of the ablations**
* **Run-to-run noise** (baseline, 3 seeds) is ±0.010 PQ. Nine of twelve changes fall inside it: at
  this budget the network is not the bottleneck.
* **Significant effects**: input resolution (512² loses 0.036 PQ, as predicted by the filament-width
  analysis in 1.2), ImageNet pre-training (+0.018), learning rate (1e-4 under-trains in 20 epochs:
  0.344). Full resolution (E02) brings no gain at 4× the input pixels, so 1024² is kept.
* **Post-processing tuning adds ≈ +0.03 PQ in every run**; a minimum instance area of 400 px is always
  optimal, while the best threshold varies (0.4–0.8) with the loss and calibration.
* **Fragmentation dominates the errors**: one-to-many errors outnumber many-to-one errors 4–5× in
  every run. The next gains come from instance extraction (section 4), not the network.
* **EfficientNet-B3** matches the baseline with 54 % of the parameters (13.2 M, 51 MB vs 93 MB): input
  for the compression work (section 4.4).
* **Decision:** keep the simplest configuration, the E01 baseline, as the final model.
""")

md("## 2.7 Training curves")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for i, (eid, h) in enumerate(sorted(hist.items())):
    col = viz.SERIES[i % len(viz.SERIES)]
    ls = "-" if i < len(viz.SERIES) else "--"
    axes[0].plot(h.epoch, h.train_loss, color=col, linestyle=ls, label=eid)
    v = h.dropna(subset=["val_pq_global"])
    axes[1].plot(v.epoch, v.val_pq_global, color=col, linestyle=ls, marker="o", markersize=4, label=eid)
axes[0].set_title("Training loss (different losses are not comparable)"); axes[0].set_xlabel("epoch")
axes[1].set_title("Validation PQ (threshold 0.5, min area 100)"); axes[1].set_xlabel("epoch")
axes[1].legend(ncol=2, fontsize=8)
fig.tight_layout(); viz.save(fig, "meth_learning_curves"); plt.show()
""")
md("""
Validation PQ still rises at epoch 20 for most runs, but doubling the budget (E13, 40 epochs) does not
improve the tuned PQ (0.393 vs 0.397 ± 0.010): the 20 × 1000-crop budget is sufficient.
""")

md("## 2.8 Model size, training time & inference time")
code("""
res.query("seed == 0")[["id", "change", "params_M", "size_MB", "train_min", "sec_per_img", "device"]].round(2)
""")
md("""
`sec_per_img` = model (whole disk at the input scale, upsampled to 2048²) + post-processing, on the
T4 during validation. Training time is wall-clock for 20 epochs including validation every 2 epochs.
""")

md("""
## 2.9 Final configuration: 5-fold cross-validation

The final model (E01 baseline) is trained on **all 5 folds** with the same protocol: one model per
fold, each validated on its held-out fold. This gives (i) mean ± std over 5 independent runs, (ii)
out-of-fold probabilities for all 707 training observations and (iii) 5 models for an ensemble
submission. Post-processing is then tuned **once on the pooled out-of-fold predictions** (sums of
TP/FP/FN/IoU over all folds), which is less optimistic than tuning on each fold.
""")
code("""
FINAL = "e01_baseline"
if TRAIN:
    for f in range(5):
        !{train_cmd(FINAL, fold=f)} --out-dir "{RUNS_DIR}"
        !python -m filaments.tune_post "{RUNS_DIR}/{FINAL}/fold{f}" --workers 2 --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --min-areas 0 100 200 400 800 1600
# Kaggle equivalent (what was run): kaggle_run.launch("final_a", [], jobs=[(FINAL, COMMON + ["--fold", str(f)]) for f in (1, 2)])

# Pooled post-processing tuning: PQ from TP/FP/FN/IoU summed over the grids of all 5 folds.
grids = []
for f in range(5):
    g = pd.read_csv(Path(RUNS_DIR) / FINAL / f"fold{f}" / "post_grid.csv").assign(fold=f)
    grids.append(g.assign(iou_sum=g.sq * g.tp))
grids = pd.concat(grids)
pool = grids.groupby(["threshold", "min_area"])[["iou_sum", "tp", "fp", "fn", "one_to_many", "many_to_one"]].sum()
pool["pq"] = pool.iou_sum / (pool.tp + 0.5 * pool.fp + 0.5 * pool.fn)
thr, area = pool.pq.idxmax()
POST = {"threshold": float(thr), "min_area": int(area), "pq_global_pooled": float(pool.pq.max())}
(Path(RUNS_DIR) / FINAL / "post_pooled.json").write_text(json.dumps(POST, indent=1))
print("pooled best post-processing:", POST)
pool.pq.unstack("threshold").round(3)
""")
code("""
final = []
for f in range(5):
    run = Path(RUNS_DIR) / FINAL / f"fold{f}"
    s = json.loads((run / "summary.json").read_text())
    g = grids.query("fold == @f and threshold == @thr and min_area == @area").iloc[0]
    final.append({"fold": f, "pq_default": s["val_pq_global"], "pq_tuned": g.pq_global, "sq": g.sq, "rq": g.rq,
                  "dice": s["val_dice_mean"], "one_to_many": g.one_to_many, "many_to_one": g.many_to_one,
                  "train_min": s["train_sec"] / 60,
                  "sec_per_img": s["val_sec_per_image_model"] + s["val_sec_per_image_post"]})
final = pd.DataFrame(final)
final.to_csv(METH / "final_5fold.csv", index=False)
stats = final.drop(columns="fold").agg(["mean", "std"])
display(final.round(3))
print("\\n".join(f"{k:12s} {stats.loc['mean', k]:.3f} ± {stats.loc['std', k]:.3f}" for k in stats.columns))
""")
code("""
fig, ax = plt.subplots(figsize=(7, 3.4))
x = np.arange(5)
ax.plot(x, final.pq_default, "o", color=viz.SEQ, ms=8, label="threshold 0.5 / min area 100")
ax.plot(x, final.pq_tuned, "o", color=viz.ACCENT, ms=8, label=f"pooled-tuned ({thr:.1f} / {area})")
for col, c in [("pq_default", viz.SEQ), ("pq_tuned", viz.ACCENT)]:
    ax.axhline(final[col].mean(), color=c, linestyle="--", linewidth=1)
ax.set_xticks(x, [f"fold {i}" for i in x]); ax.set_ylabel("validation PQ")
ax.set_title(f"Final model, 5-fold CV: PQ {final.pq_tuned.mean():.3f} ± {final.pq_tuned.std():.3f}")
ax.legend(fontsize=8); ax.grid(axis="x", visible=False)
fig.tight_layout(); viz.save(fig, "meth_final_5fold"); plt.show()
""")
md("""
**Final model** (U-Net ResNet-34, 1024² disk, BCE + Dice, full augmentation, AdamW 3e-4, 20 × 1000
crops; threshold 0.5, min area 400): **5-fold PQ 0.378 ± 0.014** (default post-processing
0.349 ± 0.010), Dice 0.658 ± 0.016, 24.4 M parameters / 93 MB, ≈ 26 min training per fold on a T4,
≈ 0.46 s per image (model + post-processing). Folds 0–1 score ≈ 0.39, folds 2–4 0.365–0.374, so
the fold-0 ablation numbers are slightly optimistic in absolute terms (comparisons between
experiments, all on fold 0, are unaffected). The pooled-tuned setting (0.5 / 400) is within 0.001 of each
fold's own optimum, so the post-processing choice is stable across folds. Fragmentation remains the
dominant error (one-to-many 89–131 vs many-to-one 19–41 per fold).
""")


# ============================================================================ 5 results
md("""
# 3. Segmentation Results

*(Segmentation results owner.)* Report means ± standard deviations over multiple runs (5 folds and/or
seeds), local vs leaderboard, the kinds of errors, and best/worst predictions on test images.

## 3.1 Local PQ (mean ± std over folds / seeds)
""")
code("""
# 5-fold results of the final configuration (computed in 2.9, post-processing tuned on pooled OOF)
display(final.round(3)); print(final.drop(columns="fold").agg(["mean", "std"]).round(3))
""")
md("""
## 3.2 Error analysis

Per site and per error type (FP, FN, one-to-many = fragmentation, many-to-one = over-merging).
""")
code("""
# per-site PQ and error types on the validation fold
p = Path(RUNS_DIR) / FINAL / "fold0" / "val_per_image.csv"
if p.exists():
    v = pd.read_csv(p)
    display(v.groupby("site")[["pq", "dice", "fp", "fn", "one_to_many", "many_to_one"]].mean().round(3))
""")
md("""
## 3.3 Qualitative results: best and worst cases

Ground truth (left) vs prediction (right) for the best and worst validation images; repeat on test
images with `predict.py` outputs (no ground truth).
""")
code("""
from filaments.evaluate import to_native, polygon_rles
from filaments.postprocess import PostConfig, instances_from_prob
def show_case(run_dir, set_id, post=None, crop=None):
    run_dir = Path(run_dir); s = next(x for x in sets if x.set_id == set_id)
    post = post or PostConfig(**{k: v for k, v in json.loads((run_dir / "post_best.json").read_text()).items()
                                 if k in ("threshold", "min_area")})
    prob = to_native(cv2.imread(str(run_dir / "oof_probs" / f"{s.observation}.png"), 0).astype(np.float32) / 255)
    pred = instances_from_prob(prob, post); img = load_image(image_path(s.observation))
    cols = [viz.hex_to_rgb(c) for c in viz.SERIES]
    gt_ov = A.overlay(img, s.instance_masks(), cols, 4); pr_ov = A.overlay(img, [pred == i for i in np.unique(pred[pred > 0])], cols, 4)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for a, im, t in zip(axes, [gt_ov, pr_ov], ["ground truth", "prediction"]):
        a.imshow(im if crop is None else im[crop]); a.set_title(f"{t} · {s.observation}"); a.axis("off")
    plt.show()
p = Path(RUNS_DIR) / FINAL / "fold0"
if (p / "val_per_image.csv").exists():
    v = pd.read_csv(p / "val_per_image.csv").sort_values("pq")
    for sid in [v.set_id.iloc[-1], v.set_id.iloc[0]]:   # best, worst
        show_case(p, sid)
""")
md("## 3.4 Kaggle leaderboard submission")
module("predict.py")
code("""
CKPTS = " ".join(f'"{p}"' for p in sorted(Path(RUNS_DIR).glob(f"{FINAL}/fold*/best.pt")))
!python -m filaments.predict --ckpts {CKPTS} --out submissions/{FINAL}.csv --threshold {POST["threshold"]} --min-area {POST["min_area"]}
# Submit (uses one of 5 daily submissions):
# !kaggle competitions submit -c filament-segmentation-2026 -f submissions/{FINAL}.csv -m "{FINAL}"
""")
md("""
| Submission | Local PQ (fold 0) | Public LB |
|---|---|---|
| E01 baseline, fold-0 model, thr 0.6 / min area 400 | 0.389 | 0.32 |
| E01 baseline, 5-fold ensemble, thr 0.5 / min area 400 | 0.378 ± 0.014 (5-fold CV) | *to submit* |

The local→public gap (≈ 0.07) is also reported by other teams (public notebook *"OOF 0.45 vs Public
0.39"*): methods are compared locally; the leaderboard is used for final confirmation only.
""")
# ============================================================================ 7 inference
md("""
## 3.5 End-to-end inference & timing

`predict.py`: JPEG → disk normalisation (cached profile) → model at 1024² (optionally TTA, several
fold models averaged) → upsample to 2048² → threshold + connected components + min area → COCO RLE
→ `submission.csv`. The whole run is timed, including model loading and file writing.
""")
code("""
p = Path(f"submissions/{FINAL}.json")
print(p.read_text() if p.exists() else "run 3.4 first")
""")
md("""
### Model size and timing

Report for the final model (and each compressed variant): parameters, size on disk, training time
and hardware, inference seconds per image (model, post-processing, end-to-end).
""")
code("""
res[["run", "params_M", "size_MB", "train_min", "sec_per_img"]].round(3)
""")

# ============================================================================ 6 enhancement
md("""
# 4. Segmentation Enhancements

*(Segmentation enhancement owner.)* Every idea is measured against the same baseline on the same
out-of-fold probabilities, and reported as a PQ gain (+ cost). The baseline's errors are dominated by
**fragmentation** (one filament → several pieces: ≈ 5× more one-to-many than many-to-one errors).

## 4.1 Instance separation
""")
code("# e.g. boundary/distance-map head + watershed; compare many_to_one before/after")
md("## 4.2 Post-processing")
code("""
# Starting point: baseline post-processing = threshold + connected components + min area
# (filaments.postprocess.instances_from_prob). Ideas: merge fragments closer than d px along a common
# direction, hysteresis thresholding, morphological closing, hole filling. Tune on
# RUNS_DIR/<run>/fold0/oof_probs with the same scoring as tune_post (no retraining needed).
""")
md("## 4.3 Fine-scale structure recovery")
code("# e.g. full-resolution model (E02), clDice loss (--loss bce:1,dice:1,cldice:0.5), TTA (predict --tta)")
md("## 4.4 Compression / distillation / pruning")
md("""
Available hooks: `count_params`, `size_mb` (2.2);
`--arch/--encoder` for a smaller student (e.g. `resnet18`, `mobilenet_v2`); out-of-fold
probabilities of the teacher in `runs/<run>/fold*/oof_probs/` as soft targets; `predict.py` times
end-to-end inference.
""")
code("# e.g. distil into resnet18 / mobilenet student, prune, FP16/INT8; report size_mb, params and seconds per image")

# ============================================================================ 8 conclusions
md("""
# 5. Conclusions & Bibliography

* **Main take-aways:** *(fill in from sections 2–4)*
* **Biggest roadblocks:** off-limb halo misleading disk detection; annotator disagreement on instance
  granularity (expert PQ ≈ 0.34); local → leaderboard gap; Kaggle GPU limits (2 concurrent sessions).
* **What we would do differently:** *(fill in)*

## Bibliography

1. A. Ahmadzadeh et al. MAGFiLO: Manually Annotated GONG Filaments from H-Alpha Observations. *Scientific Data* 11:1031, 2024.
2. A. Kirillov, K. He, R. Girshick, C. Rother, P. Dollár. Panoptic Segmentation. *CVPR* 2019.
3. O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI* 2015.
4. K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. *CVPR* 2016.
5. Z. Zhou et al. UNet++: A Nested U-Net Architecture for Medical Image Segmentation. *DLMIA* 2018.
6. M. Tan, Q. Le. EfficientNet: Rethinking Model Scaling for CNNs. *ICML* 2019.
7. T.-Y. Lin et al. Focal Loss for Dense Object Detection. *ICCV* 2017.
8. S. S. M. Salehi et al. Tversky Loss Function for Image Segmentation. *MLMI* 2017.
9. S. Shit et al. clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. *CVPR* 2021.
10. I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization (AdamW). *ICLR* 2019.
11. P. Iakubovskii. Segmentation Models PyTorch. GitHub, 2019.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT, f"({len(cells)} cells)")
