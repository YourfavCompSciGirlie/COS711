"""COCO RLE helpers and submission writing.

Kaggle expects one row per predicted filament: `filament_id` (<image stem>_<k>) and
`segmentation_rle` (COCO compressed RLE counts only; size is fixed at 2048x2048).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

IMAGE_SIZE = 2048


def encode(mask: np.ndarray) -> str:
    """Binary HxW mask -> COCO compressed RLE counts string."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def decode(counts: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """COCO compressed RLE counts string -> boolean HxW mask."""
    return mask_utils.decode({"size": [size, size], "counts": counts.encode("ascii")}).astype(bool)


def polygon_to_mask(polygon: list[float], height: int, width: int) -> np.ndarray:
    """Flat [x0, y0, x1, y1, ...] polygon -> boolean mask (same rasterisation as pycocotools)."""
    rles = mask_utils.frPyObjects([polygon], height, width)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def label_map_to_rows(image_stem: str, labels: np.ndarray) -> list[dict]:
    """Instance label map (0 = background) -> submission rows for one image."""
    rows = []
    for k, lab in enumerate(np.unique(labels[labels > 0]), start=1):
        rows.append({"filament_id": f"{image_stem}_{k}", "segmentation_rle": encode(labels == lab)})
    return rows


def write_submission(rows: list[dict], path: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    # Kaggle forbids quotes around the RLE strings; RLE counts never contain commas.
    df.to_csv(path, index=False, quoting=3)  # csv.QUOTE_NONE
    return df


def read_submission(path: str) -> dict[str, list[np.ndarray]]:
    """Submission CSV -> {image_stem: [instance masks]} (for sanity-checking before upload)."""
    df = pd.read_csv(path, quoting=3, dtype=str)
    out: dict[str, list[np.ndarray]] = {}
    for fid, counts in zip(df.filament_id, df.segmentation_rle):
        out.setdefault(fid.rsplit("_", 1)[0], []).append(decode(counts))
    return out
