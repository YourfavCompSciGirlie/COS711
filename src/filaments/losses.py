"""Segmentation losses. All take raw logits (B,1,H,W) and targets in [0,1] (soft labels allowed).

Specified as a weighted sum string, e.g. "bce:1,dice:1" or "focal:1,tversky:1,cldice:0.5".
Overlap losses are computed over the whole batch (not per crop), so crops without any filament
do not produce undefined or exploding per-sample Dice terms.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bce(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def focal(logits, target, gamma: float = 2.0, alpha: float = 0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t) ** gamma * ce).mean()


def dice(logits, target, eps: float = 1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


def tversky(logits, target, alpha: float = 0.3, beta: float = 0.7, eps: float = 1.0):
    """beta > alpha penalises false negatives more: favours recall of thin/faint structure."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    fn = ((1 - p) * target).sum()
    return 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


# ---- clDice (Shit et al., CVPR 2021): overlap of soft skeletons -> rewards topological continuity

def _soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def _soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters: int = 10):
    skel = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_dilate(_soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice(logits, target, iters: int = 10, eps: float = 1.0):
    p = torch.sigmoid(logits)
    sp, st = soft_skeleton(p, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).sum() + eps) / (sp.sum() + eps)
    tsens = ((st * p).sum() + eps) / (st.sum() + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


LOSSES = {"bce": bce, "focal": focal, "dice": dice, "tversky": tversky, "cldice": cldice}


class Combined:
    def __init__(self, spec: str):
        self.terms = []
        for part in spec.split(","):
            name, w = part.split(":") if ":" in part else (part, "1")
            self.terms.append((name.strip(), LOSSES[name.strip()], float(w)))

    def __call__(self, logits, target):
        parts = {name: fn(logits.float(), target) for name, fn, _ in self.terms}
        total = sum(w * parts[name] for name, _, w in self.terms)
        return total, {k: float(v.detach()) for k, v in parts.items()}
