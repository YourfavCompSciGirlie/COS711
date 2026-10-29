"""Predict the test set and write a Kaggle submission (end-to-end timed).

    python -m filaments.predict --ckpts runs/<name>/fold*/best.pt --out submission.csv [--tta]

Several checkpoints (e.g. the 5 fold models) are averaged at the probability level.
Post-processing defaults to the tuned values in <run>/post_best.json of the first checkpoint, if present.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import TEST_IMAGES
from .dataset import full_disk_input
from .evaluate import predict_prob, to_native
from .models import build_model, pick_device
from .postprocess import PostConfig, instances_from_prob
from .preprocess import ProfileStore
from .rle import label_map_to_rows, read_submission, write_submission


def load_models(paths, device):
    models, scale = [], None
    for p in paths:
        ckpt = torch.load(p, map_location=device)
        cfg = ckpt["config"]
        m = build_model(cfg["arch"], cfg["encoder"], weights=None).to(device).eval()
        m.load_state_dict(ckpt["model"])
        if scale is not None and cfg["scale"] != scale:
            raise ValueError("all checkpoints must share the same input scale")
        scale = cfg["scale"]
        models.append(m)
    return models, scale


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--threshold", type=float)
    p.add_argument("--min-area", type=int)
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)

    t_start = time.perf_counter()
    device = pick_device()
    models, scale = load_models(args.ckpts, device)
    tuned = Path(args.ckpts[0]).parent / "post_best.json"
    post = PostConfig(**{k: v for k, v in json.loads(tuned.read_text()).items() if k in ("threshold", "min_area")}) \
        if tuned.exists() else PostConfig()
    if args.threshold is not None:
        post.threshold = args.threshold
    if args.min_area is not None:
        post.min_area = args.min_area
    amp = device.type == "cuda"

    observations = sorted(p.stem for p in TEST_IMAGES.glob("*.jpeg"))[: args.limit]
    profiles, rows = ProfileStore(), []
    for obs in observations:
        x = full_disk_input(obs, scale, profiles)
        prob = np.mean([predict_prob(m, x, device, tta=args.tta, amp=amp) for m in models], axis=0)
        rows += label_map_to_rows(obs, instances_from_prob(to_native(prob), post))
    write_submission(rows, args.out)
    total = time.perf_counter() - t_start

    check = read_submission(args.out)
    report = {"images": len(observations), "images_with_predictions": len(check), "rows": len(rows),
              "post": vars(post), "tta": args.tta, "n_models": len(models),
              "total_sec": total, "sec_per_image": total / max(len(observations), 1), "device": str(device)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
