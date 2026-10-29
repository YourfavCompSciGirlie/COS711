"""Dataset analysis (notebook section 1.2: data inventory, image quality, filament geometry, annotator agreement).

Heavy per-image / per-annotation-set work lives here so it can run in a process pool and be
cached to CSV; the notebook only loads tables and plots.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from skimage.morphology import skeletonize

from .data import AnnotationSet, fit_disk, image_path, load_image, parse_observation
from .metrics import stats_from_iou

N_RADIAL = 20      # radial bins over r/R in [0, 1] for limb-darkening profiles
GAP_CAP = 150      # px; nearest-neighbour gaps beyond this are reported as the cap


# ---------------------------------------------------------------- image quality

def _image_row(observation: str) -> dict:
    img = load_image(image_path(observation))
    cx, cy, r = fit_disk(img)
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    rho = np.hypot(xx - cx, yy - cy) / r
    disk = rho < 1.0
    inner = rho < 0.9
    f = img.astype(np.float32)

    bins = np.minimum((rho * N_RADIAL).astype(int), N_RADIAL)
    sums = np.bincount(bins[disk], weights=f[disk], minlength=N_RADIAL)[:N_RADIAL]
    counts = np.bincount(bins[disk], minlength=N_RADIAL)[:N_RADIAL]
    profile = sums / np.maximum(counts, 1)

    # Limb-flattened image: divide out the radial profile, then measure what is left at large
    # scales (clouds, haze, gradients) vs. small scales (sharpness / seeing).
    flat = np.ones_like(f)
    flat[disk] = f[disk] / np.maximum(profile[bins[disk]], 1)
    large = cv2.GaussianBlur(flat, (0, 0), 40)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)

    t, site = parse_observation(observation)
    row = {
        "observation": observation, "site": site, "time": t,
        "cx": cx, "cy": cy, "r": r,
        "median": float(np.median(f[inner])),
        "iqr": float(np.subtract(*np.percentile(f[inner], [75, 25]))),
        "sharpness": float(lap[inner].var()),
        "nonuniformity": float(large[inner].std()),
        "dark_frac": float((flat[inner] < 0.75).mean()),   # fraction of clearly dark disk pixels
    }
    row.update({f"prof_{i:02d}": float(v) for i, v in enumerate(profile)})
    return row


def image_quality_table(observations: list[str], workers: int = 8) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_image_row, observations, chunksize=8))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- filament geometry

def _bbox(m: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(m)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def _set_rows(args) -> tuple[list[dict], dict]:
    s, disk = args
    cx, cy, r = disk
    masks = s.instance_masks()
    occ = np.zeros((s.height, s.width), np.uint8)
    for m in masks:
        occ += m
    rows = []
    for k, (m, cat) in enumerate(zip(masks, s.categories)):
        if not m.any():
            rows.append({"set_id": s.set_id, "observation": s.observation, "k": k, "area": 0})
            continue
        y0, y1, x0, x1 = _bbox(m)
        Y0, Y1 = max(y0 - GAP_CAP, 0), min(y1 + GAP_CAP, s.height)
        X0, X1 = max(x0 - GAP_CAP, 0), min(x1 + GAP_CAP, s.width)
        mc = m[Y0:Y1, X0:X1]
        others = (occ[Y0:Y1, X0:X1] - mc) > 0
        if others.any():
            dist_to_others = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 5)
            gap = float(min(dist_to_others[mc].min(), GAP_CAP))
        else:
            gap = float(GAP_CAP)

        own = np.pad(m[y0:y1, x0:x1], 1)
        edt = cv2.distanceTransform(own.astype(np.uint8), cv2.DIST_L2, 5)
        skel = skeletonize(own)
        half = edt[skel] if skel.any() else edt[own]
        contour = max(cv2.findContours(own.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        area = int(m.sum())
        ys, xs = np.nonzero(m)
        rows.append({
            "set_id": s.set_id, "observation": s.observation, "site": s.site, "k": k,
            "category": cat, "area": area, "bbox_w": x1 - x0, "bbox_h": y1 - y0,
            "width_median": 2 * float(np.median(half)), "width_max": 2 * float(half.max()),
            "skeleton_len": int(skel.sum()),
            "solidity": area / hull_area if hull_area else 1.0,
            "r_norm": float(np.hypot(xs.mean() - cx, ys.mean() - cy) / r),
            "gap_to_nearest": gap,
            "touches_other": bool(((occ > 1) & m).any()),
        })
    set_row = {
        "set_id": s.set_id, "observation": s.observation, "site": s.site,
        "n_filaments": len(masks),
        "filament_px": int((occ > 0).sum()),
        "overlap_px": int((occ > 1).sum()),
        "disk_px": float(np.pi * r * r),
    }
    return rows, set_row


def filament_tables(sets: list[AnnotationSet], disks: dict[str, tuple[float, float, float]],
                    workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-filament geometry and per-annotation-set summaries."""
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_set_rows, [(s, disks[s.observation]) for s in sets], chunksize=4))
    fil = pd.DataFrame([r for rows, _ in out for r in rows])
    per_set = pd.DataFrame([sr for _, sr in out])
    per_set["filament_frac"] = per_set.filament_px / per_set.disk_px
    return fil, per_set


# ---------------------------------------------------------------- annotator agreement

def _rles(s: AnnotationSet):
    return [mask_utils.merge(mask_utils.frPyObjects([p], s.height, s.width)) for p in s.polygons]


def agreement_table(sets: list[AnnotationSet]) -> pd.DataFrame:
    """PQ / Dice between every pair of annotation sets of the same observation.
    PQ is symmetric in its arguments, so each unordered pair is scored once."""
    by_obs: dict[str, list[AnnotationSet]] = {}
    for s in sets:
        by_obs.setdefault(s.observation, []).append(s)
    rows = []
    for obs, group in by_obs.items():
        if len(group) < 2:
            continue
        rles = {s.set_id: _rles(s) for s in group}
        for a, b in combinations(group, 2):
            ra, rb = rles[a.set_id], rles[b.set_id]
            # pycocotools.iou(dt, gt) -> (len(dt), len(gt)); we want (gt=a, pred=b).
            iou = np.asarray(mask_utils.iou(rb, ra, [0] * len(ra))).T if ra and rb else np.zeros((len(ra), len(rb)))
            st = stats_from_iou(iou, len(ra), len(rb))
            ua, ub = mask_utils.merge(ra), mask_utils.merge(rb)
            A, B = mask_utils.area(ua), mask_utils.area(ub)
            u_iou = float(mask_utils.iou([ub], [ua], [0])[0][0])
            inter = u_iou * (A + B) / (1 + u_iou)
            rows.append({
                "observation": obs, "site": a.site, "set_a": a.set_id, "set_b": b.set_id,
                "n_a": len(ra), "n_b": len(rb),
                "pq": st.pq, "sq": st.sq, "rq": st.rq,
                "tp": st.tp, "fp": st.fp, "fn": st.fn,
                "one_to_many": st.one_to_many, "many_to_one": st.many_to_one,
                "dice": 2 * inter / (A + B) if A + B else 1.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- drawing

def overlay(img: np.ndarray, masks: list[np.ndarray], colors: list[tuple[int, int, int]],
            thickness: int = 3) -> np.ndarray:
    """RGB image with each instance outlined in its own colour."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i, m in enumerate(masks):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, colors[i % len(colors)], thickness)
    return out
