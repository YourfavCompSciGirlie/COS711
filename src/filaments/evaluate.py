"""Full-disk inference and local PQ evaluation on held-out observations.

Every annotation set of a validation observation is scored as its own ground truth (the same
convention Kaggle uses for training data: "treat those images simply as different images").
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

from .data import AnnotationSet
from .dataset import full_disk_input
from .metrics import aggregate, stats_from_iou
from .postprocess import PostConfig, instances_from_prob
from .preprocess import ProfileStore

NATIVE = 2048


@torch.no_grad()
def predict_prob(model, x: np.ndarray, device, tta: bool = False, amp: bool = False) -> np.ndarray:
    """x: float32 (H, W) normalised disk at model resolution -> sigmoid probabilities (H, W)."""
    t = torch.from_numpy(x)[None, None].to(device)
    views = [lambda a: a, lambda a: a.flip(-1), lambda a: a.flip(-2), lambda a: a.flip(-1).flip(-2)] if tta else [lambda a: a]
    acc = 0
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        for v in views:
            acc = acc + v(torch.sigmoid(model(v(t)).float()))  # each view is its own inverse
    return (acc / len(views))[0, 0].cpu().numpy()


def to_native(prob: np.ndarray) -> np.ndarray:
    if prob.shape[0] == NATIVE:
        return prob
    return cv2.resize(prob, (NATIVE, NATIVE), interpolation=cv2.INTER_LINEAR)


def label_rles(labels: np.ndarray) -> list[dict]:
    ids = np.unique(labels[labels > 0])
    return [mask_utils.encode(np.asfortranarray((labels == i).astype(np.uint8))) for i in ids]


def polygon_rles(s: AnnotationSet) -> list[dict]:
    return [mask_utils.merge(mask_utils.frPyObjects([p], s.height, s.width)) for p in s.polygons]


def score(gt: list[dict], pred: list[dict]):
    iou = np.asarray(mask_utils.iou(pred, gt, [0] * len(gt))).T if gt and pred else np.zeros((len(gt), len(pred)))
    st = stats_from_iou(iou, len(gt), len(pred))
    # semantic Dice of the unions
    if gt and pred:
        ug, up = mask_utils.merge(gt), mask_utils.merge(pred)
        A, B = mask_utils.area(ug), mask_utils.area(up)
        u = float(mask_utils.iou([up], [ug], [0])[0][0])
        dice = 2 * (u * (A + B) / (1 + u)) / (A + B)
    else:
        dice = float(not gt and not pred)
    return st, dice


def evaluate(model, sets: list[AnnotationSet], scale: float, device, post: PostConfig = PostConfig(),
             tta: bool = False, amp: bool = False, prob_dir: Path | None = None,
             limit: int | None = None) -> tuple[dict, pd.DataFrame]:
    """Predict each validation observation once, score against each of its annotation sets."""
    model.eval()
    profiles = ProfileStore()
    by_obs: dict[str, list[AnnotationSet]] = {}
    for s in sets:
        by_obs.setdefault(s.observation, []).append(s)
    observations = sorted(by_obs)[:limit]
    rows, stats = [], []
    t_model = t_post = 0.0
    for obs in observations:
        t0 = time.perf_counter()
        x = full_disk_input(obs, scale, profiles)
        prob = predict_prob(model, x, device, tta=tta, amp=amp)
        t1 = time.perf_counter()
        if prob_dir is not None:
            prob_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(prob_dir / f"{obs}.png"), np.round(prob * 255).astype(np.uint8))
        labels = instances_from_prob(to_native(prob), post)
        pred = label_rles(labels)
        t2 = time.perf_counter()
        t_model += t1 - t0
        t_post += t2 - t1
        for s in by_obs[obs]:
            st, d = score(polygon_rles(s), pred)
            stats.append(st)
            rows.append({"observation": obs, "set_id": s.set_id, "site": s.site, "pq": st.pq,
                         "sq": st.sq, "rq": st.rq, "tp": st.tp, "fp": st.fp, "fn": st.fn,
                         "one_to_many": st.one_to_many, "many_to_one": st.many_to_one,
                         "dice": d, "n_gt": len(s.polygons), "n_pred": len(pred)})
    per_image = pd.DataFrame(rows)
    summary = aggregate(stats)
    summary["dice_mean"] = float(per_image.dice.mean()) if len(per_image) else float("nan")
    summary["sec_per_image_model"] = t_model / max(len(observations), 1)
    summary["sec_per_image_post"] = t_post / max(len(observations), 1)
    return summary, per_image
