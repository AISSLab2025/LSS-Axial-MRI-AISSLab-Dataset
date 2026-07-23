from __future__ import annotations

import csv
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

from datasets.data import NPZSeg2D
from infer_utils import _pad_to_multiple, _unpad, _divisor
from model_registry import get_model
from metrics.metrics import compute_metrics
from visualization.plot import save_4panel_overlay, save_separate_overlays


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def evaluate_model(
    model_name: str,
    checkpoint: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    config: dict,
    threshold: float = 0.5,
    device_name: str = "auto",
    save_overlays: bool = True,
    save_all_overlays: bool = False,
) -> dict:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    device = _device(device_name)
    in_channels = int(model_cfg.get("in_channels", 1))
    out_channels = int(model_cfg.get("out_channels", 4))
    img_size = tuple(model_cfg.get("img_size", [224, 224]))
    divisor = _divisor(int(model_cfg.get("depth", 3))) if "depth" in model_cfg else 16

    # Load dataset
    test_ds = NPZSeg2D(
        test_dir,
        image_key=data_cfg.get("image_key", "image"),
        mask_key=data_cfg.get("mask_key", "mask"),
        out_channels=out_channels,
        size=data_cfg.get("size", None),
    )

    # Add BG (Background) as the final evaluation class
    eval_class_names = [f"Class{i + 1}" for i in range(out_channels)] + ["BG"]
    eval_channels = out_channels + 1

    # Load model
    model = get_model(model_name, in_channels, out_channels, img_size).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overlay indices selection (same 50 indices for all models or whole dataset)
    if save_all_overlays:
        overlay_indices = set(range(len(test_ds)))
    else:
        rng = random.Random(42)
        overlay_indices = set(rng.sample(range(len(test_ds)), min(50, len(test_ds))))

    records = []
    inference_times_ms = []
    per_image_rows = []

    print(f"Testing {model_name} on {len(test_ds)} images...")

    with torch.no_grad():
        for idx in range(len(test_ds)):
            image, mask = test_ds[idx]
            meta = test_ds.get_metadata(idx)
            image_name = meta["image_name"]
            patient_id = meta["patient_id"]

            img_tensor = image.unsqueeze(0).to(device)
            padded, pad = _pad_to_multiple(img_tensor, divisor)

            # Precise Timing: Forward Pass + Sigmoid + Unpad + Thresholding
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                out = model(padded)
                logits = out["logits"] if isinstance(out, dict) else out
                logits = _unpad(logits, pad)
                prob = torch.sigmoid(logits)[0]
                pred_tensor = (prob >= threshold).to(torch.uint8)

                end_event.record()
                torch.cuda.synchronize()
                t_ms = start_event.elapsed_time(end_event)
                pred = pred_tensor.cpu().numpy()
            else:
                t0 = time.perf_counter()

                out = model(padded)
                logits = out["logits"] if isinstance(out, dict) else out
                logits = _unpad(logits, pad)
                prob = torch.sigmoid(logits)[0]
                pred_tensor = (prob >= threshold).to(torch.uint8)

                t1 = time.perf_counter()
                t_ms = (t1 - t0) * 1000.0
                pred = pred_tensor.cpu().numpy()

            inference_times_ms.append(t_ms)
            target = mask.cpu().numpy().astype(np.uint8)

            # Calculate Background (BG) channel as where all foreground channels are zero
            pred_bg = (pred.sum(axis=0) == 0).astype(np.uint8)
            target_bg = (target.sum(axis=0) == 0).astype(np.uint8)
            
            # Concatenate BG as the last channel
            pred = np.concatenate([pred, pred_bg[None]], axis=0)
            target = np.concatenate([target, target_bg[None]], axis=0)

            # Visual overlay saving (exclude BG channel from visualization overlays)
            if save_overlays and idx in overlay_indices:
                dir_name = "all_overlay_samples" if save_all_overlays else "50_overlay_samples"
                overlay_dir = output_dir / dir_name
                overlay_dir.mkdir(parents=True, exist_ok=True)
                base_name = f"{patient_id}_{Path(image_name).stem}"
                save_4panel_overlay(
                    image.cpu().numpy(),
                    pred[:-1],
                    target[:-1],
                    overlay_dir / f"{base_name}.png",
                )
                save_separate_overlays(
                    image.cpu().numpy(),
                    pred[:-1],
                    overlay_dir,
                    base_name,
                )

            # Evaluate each class (including BG)
            image_records = {}
            for c in range(eval_channels):
                class_name = eval_class_names[c]
                pred_c = pred[c]
                target_c = target[c]

                metrics = compute_metrics(pred_c, target_c)
                image_records[class_name] = metrics

            # Construct row data for per_image_metrics.csv
            row_data = [f"{patient_id}_{image_name}"]
            metrics_list = ["Dice", "IoU", "HD95", "ASSD", "Precision", "Recall", "Specificity", "Accuracy"]
            for m in metrics_list:
                class_vals = []
                for c in range(eval_channels):
                    class_name = eval_class_names[c]
                    class_vals.append(image_records[class_name][m])
                avg_val = np.nanmean(class_vals) if not all(np.isnan(class_vals)) else float("nan")
                row_data.extend(class_vals + [avg_val])

            fps = 1000.0 / t_ms if t_ms > 0 else 0.0
            row_data.extend([t_ms, fps])
            per_image_rows.append(row_data)

    # Calculate dynamic column headers with BG
    row1 = ["Image_ID"]
    row2 = [""]
    class_cols = [f"Class{i + 1}" for i in range(out_channels)] + ["BG"]
    for m in metrics_list:
        row1.extend([m] * (eval_channels + 1))
        row2.extend(class_cols + ["Average"])
    row1.extend(["Inference", "Inference"])
    row2.extend(["Time (ms)", "FPS"])

    # 1. Save per_image_metrics.csv
    with open(output_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerows(per_image_rows)

    # 2. Save summary_metrics.csv (Mean and SD dataset-wide)
    np_rows = np.array(per_image_rows, dtype=object)
    # Extract values only (skip Image_ID column)
    numeric_data = np_rows[:, 1:].astype(float)
    
    mean_vals = np.nanmean(numeric_data, axis=0)
    # std deviation with degrees of freedom ddof=1
    with np.errstate(divide='ignore', invalid='ignore'):
        sd_vals = np.nanstd(numeric_data, axis=0, ddof=1)
        sd_vals = np.where(np.isnan(sd_vals), 0.0, sd_vals)

    summary_row1 = ["Mean"] + list(mean_vals)
    summary_row2 = ["Standard Deviation"] + list(sd_vals)

    with open(output_dir / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerow(summary_row1)
        writer.writerow(summary_row2)

    # Build average dict to return for master summary
    avg_dict = {}
    col_idx = 0
    for m in metrics_list:
        avg_dict[m] = mean_vals[col_idx + eval_channels]  # Average sub-column mean value
        col_idx += eval_channels + 1
    
    avg_dict["Inference_Time_ms"] = np.mean(inference_times_ms)
    avg_dict["FPS"] = 1000.0 / avg_dict["Inference_Time_ms"] if avg_dict["Inference_Time_ms"] > 0 else 0.0
    return avg_dict
