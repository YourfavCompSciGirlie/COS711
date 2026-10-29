"""Probability map -> instance label map.

This is the *baseline* post-processing (threshold + connected components + small-object removal).
Instance separation / fragment merging / other enhancements build on `instances_from_prob`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PostConfig:
    threshold: float = 0.5
    min_area: int = 100        # native (2048²) pixels; EDA: 10th-percentile filament area ≈ 410 px


def instances_from_prob(prob: np.ndarray, cfg: PostConfig = PostConfig()) -> np.ndarray:
    """prob: float HxW at native resolution. Returns int32 label map (0 = background)."""
    binary = (prob >= cfg.threshold).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, np.int32)
    k = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_area:
            k += 1
            keep[i] = k
    return keep[lab]
