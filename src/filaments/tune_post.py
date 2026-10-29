"""Grid-search post-processing parameters on saved out-of-fold probability maps.

Post-processing hyperparameters are tuned on validation PQ, never on the leaderboard:
    python -m filaments.tune_post runs/<name>/fold0 [--thresholds 0.3 0.4 0.5] [--min-areas 0 100 400]
Writes <run>/post_grid.csv (one row per setting, sorted by PQ).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .data import load_annotation_sets
from .evaluate import polygon_rles, score, to_native
from .metrics import aggregate


def _one(obs: str, gts: list[list[dict]], prob_dir: Path, thresholds, min_areas):
    prob = to_native(cv2.imread(str(prob_dir / f"{obs}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255)
    out = {}
    for t in thresholds:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((prob >= t).astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        rles = [mask_utils.encode(np.asfortranarray((lab == i + 1).astype(np.uint8))) for i in range(n - 1)]
        for a in min_areas:
            pred = [r for r, ar in zip(rles, areas) if ar >= a]
            out[(t, a)] = [score(gt, pred) for gt in gts]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--min-areas", type=int, nargs="+", default=[0, 50, 100, 200, 400, 800])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    run = Path(args.run)
    prob_dir = run / "oof_probs"
    observations = sorted(f.stem for f in prob_dir.glob("*.png"))
    by_obs: dict[str, list] = {}
    for s in load_annotation_sets():
        if s.observation in observations:
            by_obs.setdefault(s.observation, []).append(polygon_rles(s))
    fn = partial(_one, prob_dir=prob_dir, thresholds=args.thresholds, min_areas=args.min_areas)
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(fn, observations, [by_obs[o] for o in observations]))

    rows = []
    for key in results[0]:
        pairs = [sd for r in results for sd in r[key]]
        summ = aggregate([st for st, _ in pairs])
        rows.append({"threshold": key[0], "min_area": key[1], **summ,
                     "dice_mean": float(np.mean([d for _, d in pairs]))})
    grid = pd.DataFrame(rows).sort_values("pq_global", ascending=False)
    grid.to_csv(run / "post_grid.csv", index=False)
    print(grid.head(10).to_string(index=False))
    best = grid.iloc[0]
    (run / "post_best.json").write_text(json.dumps({"threshold": float(best.threshold),
                                                    "min_area": int(best.min_area),
                                                    "pq_global": float(best.pq_global)}, indent=2))


if __name__ == "__main__":
    main()
