# COS711 Assignment 2 — Solar filament segmentation

Deep-learning instance segmentation of solar filaments in GONG H-alpha images
([Kaggle: Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)),
scored by Panoptic Quality (PQ).

## The notebook

`src/` is the source; the notebook embeds every module as a `%%writefile` cell. After changing
`src/`, regenerate and re-run it so the saved outputs match the code:

```bash
.venv/bin/python tools/build_colab.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/COS_711_Assignment_2.ipynb
```

On Colab it works in `MyDrive/COS711_A2/` (needs `runs/`, `data/processed/`, `data/folds.csv` there,
and a `KAGGLE_API_TOKEN` secret); see notebook section 0.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
# data: kagglehub.competition_download('filament-segmentation-2026'), then link it to
#   data/MAGFiLO_1.0_Kaggle_2026   (or set FILAMENT_DATA=/path/to/MAGFiLO_1.0_Kaggle_2026)
PYTHONPATH=src .venv/bin/python -m filaments.preprocess    # disk profiles + label maps -> data/processed (~2 min)
```

All commands below assume `PYTHONPATH=src`.

## Layout

| Path | What |
|---|---|
| `notebooks/COS_711_Assignment_2.ipynb` | **The submitted notebook**: every module below, EDA, methodology, results, enhancement, conclusions (with outputs) |
| `EXPERIMENTS.md` | Experiment protocol, all results and the final-model decision |
| `data/folds.csv` | **The** 5-fold split (by observation, 27-day time blocks). Everyone uses it. |
| `src/filaments/data.py` | Annotation loading, disk fitting, fold construction |
| `src/filaments/preprocess.py` | Limb-darkening flattening + normalisation, cache building |
| `src/filaments/dataset.py` | Random-crop training dataset (affine aug, filament-centred sampling, soft/separate labels) |
| `src/filaments/models.py` | Model factory (segmentation_models_pytorch), size/params |
| `src/filaments/losses.py` | BCE, focal, Dice, Tversky, clDice; combine as `"bce:1,dice:1"` |
| `src/filaments/postprocess.py` | Probability → instances (baseline: threshold + connected components) |
| `src/filaments/metrics.py` | PQ / SQ / RQ, Dice, one-to-many & many-to-one counts |
| `src/filaments/evaluate.py` | Full-disk inference + local PQ on held-out observations |
| `src/filaments/train.py` | Training CLI (`--help` for all options) |
| `src/filaments/tune_post.py` | Grid-search post-processing on saved out-of-fold probabilities |
| `src/filaments/predict.py` | Test-set inference → `submission.csv` (timed end-to-end) |
| `src/filaments/analysis.py`, `viz.py` | Data-analysis tables, figure style |
| `tools/kaggle_run.py` | Bundle code to a private Kaggle dataset, launch/fetch GPU runs |
| `tools/build_colab.py` | Regenerates the notebook from `src/` (keeps both identical) |
| `tests/test_core.py` | PQ, RLE and data unit tests (`pytest`) |

## Typical workflow

```bash
python -m filaments.train --name unet_r34 --fold 0 --save-probs     # train + validate on fold 0
python -m filaments.tune_post runs/unet_r34/fold0                     # tune threshold / min area on OOF
python -m filaments.predict --ckpts runs/unet_r34/fold*/best.pt --out submission.csv
```

On Kaggle GPUs (private kernels):

```bash
python tools/kaggle_run.py bundle                                   # after any code change
python tools/kaggle_run.py launch unet_r34 -- --fold 0 --epochs 40
python tools/kaggle_run.py status unet_r34
python tools/kaggle_run.py fetch unet_r34                            # -> runs/unet_r34/
```

Each run directory holds `config.json`, `history.csv`, `best.pt`, `summary.json` (PQ, Dice, params,
size, train time, inference time), `val_per_image.csv`, `oof_probs/` and `post_grid.csv`.

## Conventions (so results are comparable)

* Validation = one fold of `data/folds.csv`; every annotation set of a val observation is scored as
  its own ground truth. Report `pq_global` (dataset-level PQ) as the headline number.
* Post-processing parameters are tuned on out-of-fold probabilities, never on the leaderboard.
* The model outputs a probability map at model resolution; post-processing works at 2048².
