"""Train a segmentation model on one CV fold (or on all data with --fold -1).

Example (quick local smoke test):
    python -m filaments.train --name smoke --epochs 1 --samples-per-epoch 64 --val-limit 4
Example (Kaggle GPU baseline):
    python -m filaments.train --name unet_r34_s05 --fold 0 --epochs 40

Outputs in runs/<name>/fold<k>/: config.json, history.csv, best.pt, summary.json,
val_per_image.csv and (with --save-probs) out-of-fold probability maps for post-processing tuning.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import load_annotation_sets
from .dataset import CropConfig, CropDataset
from .evaluate import evaluate
from .losses import Combined
from .models import build_model, count_params, pick_device, size_mb
from .postprocess import PostConfig

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--fold", type=int, default=0, help="validation fold; -1 = train on everything")
    p.add_argument("--folds-file", default=str(ROOT / "data" / "folds.csv"))
    p.add_argument("--out-dir", default=str(ROOT / "runs"))
    # model
    p.add_argument("--arch", default="unet")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--weights", default="imagenet", help="'none' for random init")
    # data
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--pos-frac", type=float, default=0.7)
    p.add_argument("--label-mode", default="separate", choices=["separate", "soft"])
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--no-photometric", action="store_true")
    # optimisation
    p.add_argument("--loss", default="bce:1,dice:1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--amp", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    # validation
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--val-limit", type=int, default=None, help="only first N val observations (smoke tests)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=100)
    p.add_argument("--save-probs", action="store_true", help="save OOF probability maps of the best model")
    return p.parse_args(argv)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = pick_device()
    amp = device.type == "cuda" if args.amp == "auto" else args.amp == "on"
    out = Path(args.out_dir) / args.name / (f"fold{args.fold}" if args.fold >= 0 else "full")
    out.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(args.folds_file)
    val_obs = set(folds.loc[folds.fold == args.fold, "observation"])
    sets = load_annotation_sets()
    train_sets = [s for s in sets if s.observation not in val_obs]
    val_sets = [s for s in sets if s.observation in val_obs]

    crop_cfg = CropConfig(crop=args.crop, scale=args.scale, pos_frac=args.pos_frac,
                          rotate=not args.no_rotate, photometric=not args.no_photometric)
    ds = CropDataset(train_sets, crop_cfg, label_mode=args.label_mode,
                     samples_per_epoch=args.samples_per_epoch, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=device.type == "cuda", drop_last=True,
                    persistent_workers=args.workers > 0)

    model = build_model(args.arch, args.encoder, None if args.weights == "none" else args.weights).to(device)
    loss_fn = Combined(args.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(dl)
    total, warm = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(warm, 1) if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=amp and device.type == "cuda")
    post = PostConfig(args.threshold, args.min_area)

    config = vars(args) | {"device": str(device), "amp": amp, "crop_config": asdict(crop_cfg),
                           "n_train_sets": len(train_sets), "n_val_sets": len(val_sets),
                           "params": count_params(model), "size_mb": size_mb(model)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[{args.name}] device={device} amp={amp} params={config['params']/1e6:.1f}M "
          f"train_sets={len(train_sets)} val_sets={len(val_sets)}", flush=True)

    history, best, t_start = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, parts_sum = time.time(), 0.0, {}
        for batch in dl:
            x, y = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss, parts = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach())
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0) + v
        row = {"epoch": epoch, "train_loss": running / steps_per_epoch, "lr": sched.get_last_lr()[0],
               "epoch_sec": time.time() - t0} | {f"loss_{k}": v / steps_per_epoch for k, v in parts_sum.items()}

        if val_sets and (epoch % args.val_every == 0 or epoch == args.epochs):
            summ, _ = evaluate(model, val_sets, args.scale, device, post, amp=amp, limit=args.val_limit)
            row |= {f"val_{k}": v for k, v in summ.items() if isinstance(v, (int, float))}
            if summ["pq_global"] > best:
                best = summ["pq_global"]
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, out / "best.pt")
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                       if k in ("epoch", "train_loss", "epoch_sec", "val_pq_global", "val_pq_mean_image",
                                "val_dice_mean", "val_sq", "val_rq"))
        print(msg, flush=True)

    train_sec = time.time() - t_start
    if not val_sets:  # full-data model: keep the last epoch
        torch.save({"model": model.state_dict(), "config": config, "epoch": args.epochs}, out / "best.pt")

    summary = {"train_sec": train_sec, "params": config["params"], "size_mb": config["size_mb"]}
    if val_sets:
        ckpt = torch.load(out / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        summ, per_image = evaluate(model, val_sets, args.scale, device, post, amp=amp, limit=args.val_limit,
                                   prob_dir=out / "oof_probs" if args.save_probs else None)
        per_image.to_csv(out / "val_per_image.csv", index=False)
        summary |= {"best_epoch": ckpt["epoch"]} | {f"val_{k}": v for k, v in summ.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
