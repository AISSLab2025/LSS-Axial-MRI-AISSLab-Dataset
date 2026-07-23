import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from pathlib import Path


class_colors = [
    [0.0, 1.0, 0.0],  # 1. Green
    [0.0, 1.0, 1.0],  # 2. Cyan
    [1.0, 0.0, 1.0],  # 3. Magenta
    [1.0, 1.0, 0.0],  # 4. Yellow
    [1.0, 0.5, 0.0],  # 5. Orange
]


def make_rgb_mask(mask: np.ndarray, colors: list) -> np.ndarray:
    h, w = mask.shape[1], mask.shape[2]
    rgb = np.zeros((h, w, 3))
    for c in range(mask.shape[0]):
        color = np.array(colors[c % len(colors)])
        rgb[mask[c] > 0] = color
    return rgb


def save_4panel_overlay(
    image: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    output_path: Path | str,
) -> None:
    """
    Saves a publication-quality 4-panel figure:
    1. Original MRI image
    2. Ground Truth mask
    3. Predicted mask
    4. Overlay of prediction and ground truth on background image
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Grayscale image
    if image.ndim == 3:
        img_gray = image[0]
    else:
        img_gray = image
    
    if img_gray.max() > 1.0:
        img_gray = img_gray / 255.0
    img_gray = np.clip(img_gray, 0.0, 1.0)

    # 1. Original Image
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original MRI", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth
    gt_rgb = make_rgb_mask(target, class_colors)
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth Mask", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # 3. Predicted Mask
    pred_rgb = make_rgb_mask(pred, class_colors)
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Predicted Mask", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # 4. Overlay GT (translucent fill) + Pred (contours)
    overlay_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).copy()
    out_channels = pred.shape[0]

    for c in range(out_channels):
        color = np.array(class_colors[c % len(class_colors)])
        gt_mask = target[c] > 0
        overlay_rgb[gt_mask] = overlay_rgb[gt_mask] * 0.7 + color * 0.3

        pred_mask = pred[c]
        eroded = binary_erosion(pred_mask)
        border = pred_mask ^ eroded
        overlay_rgb[border > 0] = color

    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Overlay (Fill:GT, Border:Pred)", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_curves(
    history: list[dict],
    curve_type: str,
    output_path: Path | str,
) -> None:
    epochs = [h["epoch"] for h in history]
    
    plt.figure(figsize=(8, 5))
    if curve_type == "learning_curve":
        train_vals = [h.get("train_loss", 0.0) for h in history]
        val_vals = [h.get("val_loss", 0.0) for h in history]
        plt.plot(epochs, train_vals, "b-o", label="Train Loss")
        plt.plot(epochs, val_vals, "r-x", label="Val Loss")
        plt.ylabel("Loss")
        plt.title("Learning Curves (Loss)")
    elif curve_type == "dice_curve":
        train_vals = [h.get("train_loss", 0.0) for h in history] # fallback if needed
        val_vals = [h.get("val_dice", 0.0) for h in history]
        plt.plot(epochs, val_vals, "g-^", label="Val Dice")
        plt.ylabel("Dice Score")
        plt.title("Validation Dice Coefficient Curve")
    
    plt.xlabel("Epoch")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_separate_overlays(
    image: np.ndarray,
    pred: np.ndarray,
    output_dir: Path | str,
    base_name: str,
) -> None:
    """
    Saves three separate files:
    1. <base_name>_image.png
    2. <base_name>_predicted_mask.png
    3. <base_name>_overlay_with_predicted_mask.png
    """
    # 1. Prepare grayscale background image
    if image.ndim == 3:
        img_gray = image[0]
    else:
        img_gray = image
    
    if img_gray.max() > 1.0:
        img_gray = img_gray / 255.0
    img_gray = np.clip(img_gray, 0.0, 1.0)
    
    # Save original image as grayscale PNG
    plt.figure(figsize=(4, 4))
    plt.imshow(img_gray, cmap="gray")
    plt.axis("off")
    plt.savefig(Path(output_dir) / f"{base_name}_image.png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    # 2. Save predicted mask
    pred_rgb = make_rgb_mask(pred, class_colors)
    plt.figure(figsize=(4, 4))
    plt.imshow(pred_rgb)
    plt.axis("off")
    plt.savefig(Path(output_dir) / f"{base_name}_predicted_mask.png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    # 3. Save overlay of predicted mask on original image
    overlay_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).copy()
    out_channels = pred.shape[0]
    for c in range(out_channels):
        color = np.array(class_colors[c % len(class_colors)])
        pred_mask = pred[c] > 0
        overlay_rgb[pred_mask] = overlay_rgb[pred_mask] * 0.7 + color * 0.3
        
        eroded = binary_erosion(pred_mask)
        border = pred_mask ^ eroded
        overlay_rgb[border > 0] = color
        
    plt.figure(figsize=(4, 4))
    plt.imshow(overlay_rgb)
    plt.axis("off")
    plt.savefig(Path(output_dir) / f"{base_name}_overlay_with_predicted_mask.png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

