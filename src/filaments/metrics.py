"""Local evaluation: Panoptic Quality (Kirillov et al., 2019), Dice, and fragmentation stats.

PQ for one class:
    PQ = sum_{(g,p) in TP} IoU(g,p) / (|TP| + 0.5|FP| + 0.5|FN|)
where a (gt, pred) pair is a TP iff IoU > 0.5. With IoU > 0.5 every gt can match at most
one pred and vice versa, so matching is unique and needs no Hungarian assignment.

Kaggle's exact aggregation (dataset-level vs. mean of per-image PQ) is not stated, so both
are reported; check them against the self-evaluation notebook / first LB submission.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Minimum IoU for a pred to count as "touching" a gt when counting one-to-many / many-to-one.
FRAG_IOU = 0.05


@dataclass
class ImageStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    one_to_many: int = 0   # gt instances split across >1 prediction (fragmentation)
    many_to_one: int = 0   # predictions covering >1 gt instance (over-merging)
    matched_ious: list[float] = field(default_factory=list)

    @property
    def pq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denom if denom else float("nan")

    @property
    def sq(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denom if denom else float("nan")


def _as_masks(x) -> list[np.ndarray]:
    """Accept a label map (HxW int, 0 = bg) or a list/stack of boolean masks."""
    if isinstance(x, np.ndarray) and x.ndim == 2 and not x.dtype == bool:
        return [x == lab for lab in np.unique(x[x > 0])]
    return [np.asarray(m, dtype=bool) for m in x]


def iou_matrix(gts: list[np.ndarray], preds: list[np.ndarray]) -> np.ndarray:
    if not gts or not preds:
        return np.zeros((len(gts), len(preds)))
    g = np.stack(gts).reshape(len(gts), -1).astype(np.float32)
    p = np.stack(preds).reshape(len(preds), -1).astype(np.float32)
    inter = g @ p.T
    union = g.sum(1)[:, None] + p.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def stats_from_iou(iou: np.ndarray, n_gt: int, n_pred: int) -> ImageStats:
    """PQ bookkeeping from an (n_gt, n_pred) IoU matrix (lets callers compute IoU however is
    fastest, e.g. pycocotools on RLEs)."""
    s = ImageStats()
    if n_gt and n_pred:
        match = iou > 0.5
        s.tp = int(match.sum())
        s.matched_ious = iou[match].tolist()
        s.iou_sum = float(iou[match].sum())
        s.fn = n_gt - int(match.any(1).sum())
        s.fp = n_pred - int(match.any(0).sum())
        touch = iou > FRAG_IOU
        s.one_to_many = int((touch.sum(1) > 1).sum())
        s.many_to_one = int((touch.sum(0) > 1).sum())
    else:
        s.fn, s.fp = n_gt, n_pred
    return s


def image_stats(gt, pred) -> ImageStats:
    gts, preds = _as_masks(gt), _as_masks(pred)
    return stats_from_iou(iou_matrix(gts, preds), len(gts), len(preds))


def aggregate(stats: list[ImageStats]) -> dict[str, float]:
    tp = sum(s.tp for s in stats)
    fp = sum(s.fp for s in stats)
    fn = sum(s.fn for s in stats)
    iou_sum = sum(s.iou_sum for s in stats)
    denom = tp + 0.5 * fp + 0.5 * fn
    per_image = np.array([s.pq for s in stats], dtype=float)
    return {
        "pq_global": iou_sum / denom if denom else float("nan"),
        "pq_mean_image": float(np.nanmean(per_image)) if np.isfinite(per_image).any() else float("nan"),
        "sq": iou_sum / tp if tp else 0.0,
        "rq": tp / denom if denom else float("nan"),
        "tp": tp, "fp": fp, "fn": fn,
        "one_to_many": sum(s.one_to_many for s in stats),
        "many_to_one": sum(s.many_to_one for s in stats),
    }


def dice(gt_binary: np.ndarray, pred_binary: np.ndarray, eps: float = 1e-7) -> float:
    """Semantic (union-of-instances) Dice for one image."""
    g, p = gt_binary.astype(bool), pred_binary.astype(bool)
    denom = g.sum() + p.sum()
    if denom == 0:
        return 1.0
    return float(2 * (g & p).sum() / (denom + eps))
