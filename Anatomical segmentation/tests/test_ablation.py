import os
import pathlib
import tempfile
import yaml
import numpy as np
import pandas as pd
import torch
from PIL import Image

from datasets.data import make_toy_dataset
from train import train_model
from test import evaluate_model
from run_all_models import compute_model_complexity


def run_single_model_test(model_name: str, tmp_dir: pathlib.Path) -> None:
    # Setup directories
    data_dir = tmp_dir / "data"
    output_dir = tmp_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create toy dataset of size 224x224, 1 channel, 4 classes
    make_toy_dataset(
        out_dir=data_dir,
        train_cases=2,
        val_cases=2,
        size=224,
        channels=1,
        classes=4,
        seed=42,
    )

    # 2. Build smoke test config
    config = {
        "seed": 42,
        "device": "cpu",
        "output_dir": str(output_dir),
        "model": {
            "in_channels": 1,
            "out_channels": 4,
            "img_size": [224, 224],
        },
        "data": {
            "train_dir": str(data_dir / "train"),
            "val_dir": str(data_dir / "val"),
            "test_dir": str(data_dir / "val"),
            "image_key": "image",
            "mask_key": "mask",
            "size": 224,
            "num_workers": 0,
        },
        "train": {
            "epochs": 1,  # 1 epoch for smoke test
            "batch_size": 2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "threshold": 0.5,
        },
    }

    # 3. Save config yaml
    config_path = tmp_dir / "smoke_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # 4. Train the model
    print(f"\n[Test] Training model {model_name}...")
    train_summary = train_model(model_name, config, device_name="cpu", out_dir=output_dir)
    
    # Assert checkpoints were saved
    model_out_dir = output_dir / model_name
    assert (model_out_dir / "best.pt").exists(), f"best.pt not saved for {model_name}"
    assert (model_out_dir / "last.pt").exists(), f"last.pt not saved for {model_name}"
    assert (model_out_dir / "training_log.csv").exists(), f"training_log.csv not saved for {model_name}"

    # 5. Evaluate the model
    print(f"[Test] Evaluating model {model_name}...")
    test_metrics = evaluate_model(
        model_name=model_name,
        checkpoint=model_out_dir / "best.pt",
        test_dir=data_dir / "val",
        output_dir=output_dir,
        config=config,
        threshold=0.5,
        device_name="cpu",
        save_overlays=True,
    )

    # Assert evaluation files are written correctly
    assert (model_out_dir / "per_image_metrics.csv").exists(), f"per_image_metrics.csv missing for {model_name}"
    assert (model_out_dir / "summary_metrics.csv").exists(), f"summary_metrics.csv missing for {model_name}"
    
    # Verify double-header CSV format
    df_img = pd.read_csv(model_out_dir / "per_image_metrics.csv", header=[0, 1])
    assert len(df_img) == 2, f"Expected 2 test image rows, got {len(df_img)} for {model_name}"
    assert ("Dice", "Class1") in df_img.columns
    assert ("Dice", "Class4") in df_img.columns
    assert ("Dice", "BG") in df_img.columns
    assert ("Dice", "Average") in df_img.columns
    assert ("Inference", "Time (ms)") in df_img.columns
    assert ("Inference", "FPS") in df_img.columns

    # Verify summary metrics format (Mean and Standard Deviation rows)
    df_sum = pd.read_csv(model_out_dir / "summary_metrics.csv", header=[0, 1])
    assert len(df_sum) == 2, f"Expected Mean and SD rows, got {len(df_sum)} for {model_name}"
    
    # Verify overlay images
    overlay_dir = model_out_dir / "50_overlay_samples"
    assert overlay_dir.exists(), f"overlays folder missing for {model_name}"
    overlay_files = list(overlay_dir.glob("*.png"))
    assert len(overlay_files) > 0, f"No overlay PNG files written for {model_name}"

    # 6. Verify complexity calculation
    print(f"[Test] Calculating complexity for {model_name}...")
    complexity = compute_model_complexity(model_name, in_channels=1, img_size=(224, 224))
    assert complexity["Total_Parameters"] > 0
    assert complexity["GFLOPs"] >= 0.0
    assert complexity["Model_Size_MB"] > 0.0
    print(f"[Test] Model {model_name} passed all smoke test assertions.")


def test_all_architectures() -> None:
    models_to_test = [
        "DeepLabV3",
        "AttentionUNet",
        "UNETR",
        "SwinUNETR",
        "SwinTransformer",
        "SegResNet",
        "TransUNet",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        print(f"Running sequential smoke test for all {len(models_to_test)} architectures inside {tmp_path}...")
        
        for model_name in models_to_test:
            try:
                run_single_model_test(model_name, tmp_path)
            except Exception as e:
                print(f"ERROR: Model '{model_name}' failed during test execution!")
                raise e

        print("\nAll 7 architectures verified and passed successfully in sequence!")


if __name__ == "__main__":
    test_all_architectures()
