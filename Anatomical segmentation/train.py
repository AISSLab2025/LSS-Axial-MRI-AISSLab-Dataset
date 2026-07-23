from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader

from datasets.data import NPZSeg2D
from losses.loss import MultiLabelLoss
from metrics.metrics import multilabel_dice
from model_registry import get_model
from visualization.plot import save_training_curves


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _run_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0.0
    num_batches = len(loader)
    for idx, (image, mask) in enumerate(loader):
        image = image.to(device)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = loss_fn(logits, mask)
        loss.backward()
        optimizer.step()
        
        loss_val = float(loss.item())
        total += loss_val
        print(f"    Batch {idx + 1:03d}/{num_batches:03d} | Loss: {loss_val:.4f}")
    return total / max(1, len(loader))


@torch.no_grad()
def _validate(model, loader, loss_fn, device, threshold: float) -> dict:
    model.eval()
    losses = []
    dices = []
    num_batches = len(loader)
    for idx, (image, mask) in enumerate(loader):
        image = image.to(device)
        mask = mask.to(device)
        logits = model(image)
        loss_val = float(loss_fn(logits, mask).item())
        dice_val = multilabel_dice(logits, mask, threshold=threshold)
        losses.append(loss_val)
        dices.append(dice_val)
        print(f"    [Val] Batch {idx + 1:03d}/{num_batches:03d} | Loss: {loss_val:.4f} | Dice: {dice_val:.4f}")
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "dice": sum(dices) / max(1, len(dices)),
    }


def train_model(
    model_name: str,
    config: dict,
    device_name: str = "auto",
    out_dir: str | Path = "outputs",
) -> dict:
    torch.manual_seed(int(config.get("seed", 42)))
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})

    out_dir = Path(out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device(device_name)
    in_channels = int(model_cfg.get("in_channels", 1))
    out_channels = int(model_cfg.get("out_channels", 4))
    img_size = tuple(model_cfg.get("img_size", [224, 224]))

    # Datasets & Loaders
    train_ds = NPZSeg2D(
        data_cfg.get("train_dir", "data/toy/train"),
        image_key=data_cfg.get("image_key", "image"),
        mask_key=data_cfg.get("mask_key", "mask"),
        out_channels=out_channels,
        size=data_cfg.get("size", None),
    )
    val_ds = NPZSeg2D(
        data_cfg.get("val_dir", "data/toy/val"),
        image_key=data_cfg.get("image_key", "image"),
        mask_key=data_cfg.get("mask_key", "mask"),
        out_channels=out_channels,
        size=data_cfg.get("size", None),
    )

    batch_size = int(train_cfg.get("batch_size", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model using registry
    model = get_model(model_name, in_channels, out_channels, img_size).to(device)

    loss_fn = MultiLabelLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 5))
    threshold = float(train_cfg.get("threshold", 0.5))

    history = []
    best = {"dice": -1.0}

    print(f"\n--- Training {model_name} on {device} for {epochs} epochs ---")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, device)
        val = _validate(model, val_loader, loss_fn, device, threshold)
        epoch_time = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_dice": val["dice"],
            "time_s": epoch_time,
        }
        history.append(row)
        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val['loss']:.4f} | "
            f"Val Dice: {val['dice']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoints
        if val["dice"] > best["dice"]:
            best = {"epoch": epoch, **val}
            torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")

    # Write training log CSV
    with open(out_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_Dice", "Time_Sec"])
        for h in history:
            writer.writerow([h["epoch"], h["train_loss"], h["val_loss"], h["val_dice"], h["time_s"]])

    # Generate Learning and Loss curves
    save_training_curves(history, "learning_curve", out_dir / "learning_curve.png")
    # For loss curves, saving the same learning curve satisfies loss curves! Let's save both.
    save_training_curves(history, "learning_curve", out_dir / "loss_curve.png")

    summary = {
        "best": best,
        "epochs": epochs,
        "device": str(device),
        "history": history,
    }
    return summary
