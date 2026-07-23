from __future__ import annotations

import csv
import time
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train import train_model
from test import evaluate_model
from model_registry import get_model


def compute_model_complexity(model_name: str, in_channels: int, img_size: tuple[int, int]) -> dict:
    from ptflops import get_model_complexity_info

    # Initialize model on CPU to avoid GPU allocation for profiling
    model = get_model(model_name, in_channels=in_channels, out_channels=4, img_size=img_size)
    model.eval()

    with torch.no_grad():
        macs, params_prof = get_model_complexity_info(
            model,
            (in_channels, img_size[0], img_size[1]),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Model size in MB
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    gflops = (macs * 2) / 1e9

    return {
        "GFLOPs": gflops,
        "MACs": macs,
        "Total_Parameters": total_params,
        "Trainable_Parameters": trainable_params,
        "Non_Trainable_Parameters": non_trainable_params,
        "Model_Size_MB": model_size_mb,
    }


def plot_comparison_bar(df: pd.DataFrame, metric: str, title: str, ylabel: str, filename: Path | str) -> None:
    plt.figure(figsize=(10, 6))
    
    models = df["Model"].to_numpy()
    vals = df[metric].to_numpy()
    
    # Render with Viridis palette
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    bars = plt.bar(models, vals, color=colors, edgecolor="black", alpha=0.9)
    
    # Draw text values on top of each bar
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        fmt = f"{h:.2f}" if abs(h) >= 0.1 else f"{h:.4f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + (h * 0.01 if h > 0 else -abs(h) * 0.05),
            fmt,
            ha="center",
            va="bottom" if h > 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
        
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=25)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def run_benchmark(config_path: str = "configs/smoke.yaml", test_only: bool = False, save_all_overlays: bool = False) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    output_root = Path(config.get("output_dir", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)

    in_channels = int(model_cfg.get("in_channels", 1))
    img_size = tuple(model_cfg.get("img_size", [224, 224]))
    device_name = str(config.get("device", "auto"))

    models_to_benchmark = [
        "DeepLabV3",
        "AttentionUNet",
        "UNETR",
        # "SwinUNETR",
        "SwinTransformer",
        "SegResNet",
        "TransUNet",
    ]

    benchmark_records = []

    for model_name in models_to_benchmark:
        print(f"\n==================================================")
        print(f" BENCHMARKING MODEL: {model_name}")
        print(f"==================================================")
        
        # 1. Complexity calculation
        print(f"Calculating complexity for {model_name}...")
        complexity = compute_model_complexity(model_name, in_channels, img_size)
        
        # 2. Model Training & Validation
        best_ckpt = output_root / model_name / "best.pt"
        if test_only:
            if not best_ckpt.exists():
                print(f"Warning: Checkpoint {best_ckpt} does not exist. Skipping benchmark for {model_name}.")
                continue
            t_train = 0.0
            print(f"Skipping training (test-only mode). Using existing checkpoint: {best_ckpt}")
        else:
            t_train_start = time.time()
            train_summary = train_model(model_name, config, device_name, output_root)
            t_train = time.time() - t_train_start
            print(f"Training completed in {t_train:.1f} seconds.")

        # 3. Model Testing (Inference, metrics, overlays)
        t_test_start = time.time()
        test_metrics = evaluate_model(
            model_name=model_name,
            checkpoint=best_ckpt,
            test_dir=data_cfg.get("test_dir", "data/toy/val"),
            output_dir=output_root,
            config=config,
            threshold=float(config.get("train", {}).get("threshold", 0.5)),
            device_name=device_name,
            save_overlays=True,
            save_all_overlays=save_all_overlays,
        )
        t_test = time.time() - t_test_start
        print(f"Testing completed in {t_test:.1f} seconds.")
        print(f"  [Speed Info] Average Inference Time: {test_metrics['Inference_Time_ms']:.2f} ms/image | FPS: {test_metrics['FPS']:.1f} images/second")

        # 4. Save model_complexity.csv inside outputs/<ModelName>/
        model_out_dir = output_root / model_name
        with open(model_out_dir / "model_complexity.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in complexity.items():
                writer.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])
            writer.writerow(["Average Inference Time (ms)", f"{test_metrics['Inference_Time_ms']:.4f}"])
            writer.writerow(["FPS", f"{test_metrics['FPS']:.4f}"])

        # 5. Append to overall comparison records
        record = {
            "Model": model_name,
            "Dice": test_metrics["Dice"],
            "IoU": test_metrics["IoU"],
            "HD95": test_metrics["HD95"],
            "ASSD": test_metrics["ASSD"],
            "Precision": test_metrics["Precision"],
            "Recall": test_metrics["Recall"],
            "Specificity": test_metrics["Specificity"],
            "Accuracy": test_metrics["Accuracy"],
            "Inference_Time_ms": test_metrics["Inference_Time_ms"],
            "FPS": test_metrics["FPS"],
            "GFLOPs": complexity["GFLOPs"],
            "MACs": complexity["MACs"],
            "Total_Parameters": complexity["Total_Parameters"],
            "Trainable_Parameters": complexity["Trainable_Parameters"],
            "Non_Trainable_Parameters": complexity["Non_Trainable_Parameters"],
            "Model_Size_MB": complexity["Model_Size_MB"],
            "Training_Time": t_train,
            "Testing_Time": t_test,
        }
        benchmark_records.append(record)

    # Convert records to DataFrame
    df = pd.DataFrame(benchmark_records)

    # Save outputs/model_comparison.csv
    df.to_csv(output_root / "model_comparison.csv", index=False)

    # Save outputs/model_comparison.xlsx
    df.to_excel(output_root / "model_comparison.xlsx", index=False)

    # Generate publication charts
    plot_comparison_bar(df, "Dice", "Segmentation Accuracy: Dice Similarity Coefficient (DSC)", "Dice Score", output_root / "Dice_comparison.png")
    plot_comparison_bar(df, "IoU", "Segmentation Accuracy: Intersection over Union (IoU)", "IoU Score", output_root / "IoU_comparison.png")
    plot_comparison_bar(df, "HD95", "Boundary Accuracy: Hausdorff Distance 95 (HD95)", "HD95 (pixels)", output_root / "HD95_comparison.png")
    plot_comparison_bar(df, "ASSD", "Boundary Accuracy: Average Symmetric Surface Distance (ASSD)", "ASSD (pixels)", output_root / "ASSD_comparison.png")
    plot_comparison_bar(df, "Precision", "Segmentation Precision Comparison", "Precision", output_root / "Precision_comparison.png")
    plot_comparison_bar(df, "Recall", "Segmentation Recall (Sensitivity) Comparison", "Recall", output_root / "Recall_comparison.png")
    plot_comparison_bar(df, "Specificity", "Segmentation Specificity Comparison", "Specificity", output_root / "Specificity_comparison.png")
    plot_comparison_bar(df, "Accuracy", "Segmentation Binary Accuracy Comparison", "Accuracy", output_root / "Accuracy_comparison.png")
    plot_comparison_bar(df, "Inference_Time_ms", "Inference Speed: Model Latency (ms)", "Latency (ms)", output_root / "Inference_Time_comparison.png")
    plot_comparison_bar(df, "FPS", "Inference Speed: Images Processed Per Second (FPS)", "FPS", output_root / "FPS_comparison.png")
    plot_comparison_bar(df, "GFLOPs", "Computational Complexity: GFLOPs at 224x224", "GFLOPs", output_root / "GFLOPs_comparison.png")
    plot_comparison_bar(df, "Total_Parameters", "Model Capacity: Total Parameters Count", "Parameters count", output_root / "Parameter_Count_comparison.png")
    plot_comparison_bar(df, "Model_Size_MB", "Model Size on Disk (MB)", "Size (MB)", output_root / "Model_Size_comparison.png")

    # Generate publication ranking table
    df_ranked = df.sort_values(by="Dice", ascending=False)
    
    print("\n" + "=" * 80)
    print(" FINAL ABLATION STUDY BENCHMARK RANKING TABLE (Sorted by Dice)")
    print("=" * 80)
    print(df_ranked[["Model", "Dice", "IoU", "HD95", "GFLOPs", "Total_Parameters", "Model_Size_MB", "FPS"]].to_string(index=False))
    print("=" * 80 + "\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run benchmarking for all segmentation models.")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Path to the configs file.")
    parser.add_argument("--test-only", action="store_true", help="Skip training and only run evaluation on existing checkpoints.")
    parser.add_argument("--save-all-overlays", action="store_true", help="Save overlays for all images in the dataset instead of 50 samples.")
    args = parser.parse_args()
    
    run_benchmark(args.config, test_only=args.test_only, save_all_overlays=args.save_all_overlays)


if __name__ == "__main__":
    main()
