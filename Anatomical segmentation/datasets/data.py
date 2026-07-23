from __future__ import annotations

import argparse
from pathlib import Path
import platform
import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_path(p: str | Path) -> Path:
    p_str = str(p).replace("\\", "/")
    if p_str.startswith("/mnt/c/"):
        if platform.system() == "Windows":
            p_str = "c:/" + p_str[7:]
    return Path(p_str)


class NPZSeg2D(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_key: str = "image",
        mask_key: str = "mask",
        out_channels: int = 3,
        size: int | tuple[int, int] | None = None,
    ) -> None:
        self.root = resolve_path(root)
        self.image_key = image_key
        self.mask_key = mask_key
        self.out_channels = int(out_channels)

        # Check if there are .npz files directly in root
        self.npz_files = sorted(self.root.glob("*.npz"))
        if self.npz_files:
            self.mode = "npz"
        else:
            # Fallback to PNG mode: recursively search for masks ending in _M.png
            self.mask_files = sorted(self.root.rglob("*_M.png"))
            self.image_files = []
            valid_masks = []
            for mask_path in self.mask_files:
                image_name = mask_path.name.replace("_M.png", ".png")
                image_path = mask_path.with_name(image_name)
                if image_path.exists():
                    self.image_files.append(image_path)
                    valid_masks.append(mask_path)
            self.mask_files = valid_masks

            if self.image_files:
                self.mode = "png"
                if size is not None:
                    if isinstance(size, int):
                        self.target_size = (size, size)
                    else:
                        self.target_size = tuple(size)
                else:
                    # Determine target size (width, height) from the first image
                    # and round to the nearest multiple of 16 to fit the model downsampling depth.
                    from PIL import Image
                    with Image.open(self.image_files[0]) as img:
                        w, h = img.size
                    self.target_size = (
                        max(16, ((w + 8) // 16) * 16),
                        max(16, ((h + 8) // 16) * 16),
                    )

                # Scan dataset to dynamically discover non-zero class values in masks
                # (e.g. 50, 100, 150, 200). Scan a subset of masks for performance.
                unique_classes = set()
                from PIL import Image
                for mask_path in self.mask_files[:100]:
                    with Image.open(mask_path) as msk:
                        unique_classes.update(np.unique(np.array(msk)))
                unique_classes.discard(0)
                self.class_values = sorted(list(unique_classes))

                # Fill up/extend class_values to out_channels if we found fewer
                while len(self.class_values) < self.out_channels:
                    if len(self.class_values) >= 2:
                        step = self.class_values[1] - self.class_values[0]
                        next_val = self.class_values[-1] + step
                    elif len(self.class_values) == 1:
                        next_val = self.class_values[0] * 2
                    else:
                        next_val = 1
                    self.class_values.append(next_val)
                # Trim to out_channels
                self.class_values = self.class_values[:self.out_channels]
            else:
                raise FileNotFoundError(
                    f"no .npz files or matching PNG image/mask pairs found in {self.root}"
                )

    def __len__(self) -> int:
        if self.mode == "npz":
            return len(self.npz_files)
        return len(self.image_files)

    def get_metadata(self, index: int) -> dict:
        if self.mode == "npz":
            path = self.npz_files[index]
            patient_id = path.parent.name
            image_name = path.name
        else:
            path = self.image_files[index]
            patient_id = path.parent.name
            image_name = path.name
        return {"patient_id": patient_id, "image_name": image_name}

    def _mask_to_channels(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            return mask.astype(np.float32)
        if mask.ndim != 2:
            raise ValueError(f"mask must be (K,H,W) or (H,W), got {mask.shape}")
        
        # If out_channels is 1, treat any positive value as the foreground class
        if self.out_channels == 1:
            return (mask > 0).astype(np.float32)[None]

        channels = [(mask == idx + 1).astype(np.float32) for idx in range(self.out_channels)]
        return np.stack(channels, axis=0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "npz":
            with np.load(self.npz_files[index]) as sample:
                image = sample[self.image_key].astype(np.float32)
                mask = self._mask_to_channels(sample[self.mask_key])
            if image.ndim == 2:
                image = image[None]
            if image.ndim != 3:
                raise ValueError(f"image must be (C,H,W) or (H,W), got {image.shape}")
            return torch.from_numpy(image), torch.from_numpy(mask)
        else:
            from PIL import Image
            img_path = self.image_files[index]
            msk_path = self.mask_files[index]

            with Image.open(img_path) as img:
                if img.size != self.target_size:
                    img = img.resize(self.target_size, Image.BILINEAR)
                image = np.array(img, dtype=np.float32)
            with Image.open(msk_path) as msk:
                if msk.size != self.target_size:
                    msk = msk.resize(self.target_size, Image.NEAREST)
                mask = np.array(msk, dtype=np.float32)

            # Map custom values (e.g. 50, 100, 150, 200) to consecutive class integers (1, 2, 3, 4)
            if hasattr(self, "class_values") and self.class_values:
                mapped_mask = np.zeros_like(mask)
                for idx, val in enumerate(self.class_values):
                    mapped_mask[mask == val] = idx + 1
                mask = mapped_mask

            if image.max() > 1.0:
                image = image / 255.0

            mask = self._mask_to_channels(mask)

            if image.ndim == 2:
                image = image[None]
            elif image.ndim == 3:
                image = image.transpose(2, 0, 1)

            return torch.from_numpy(image), torch.from_numpy(mask)


def _circle_mask(height: int, width: int, cy: int, cx: int, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def make_toy_dataset(
    out_dir: str | Path,
    train_cases: int = 8,
    val_cases: int = 4,
    size: int = 64,
    channels: int = 1,
    classes: int = 3,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    for split, count in {"train": train_cases, "val": val_cases}.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            image = rng.normal(0, 0.05, size=(channels, size, size)).astype(np.float32)
            mask = np.zeros((classes, size, size), dtype=np.float32)
            for cls in range(classes):
                cy = int(rng.integers(size // 4, 3 * size // 4))
                cx = int(rng.integers(size // 4, 3 * size // 4))
                radius = int(rng.integers(size // 10, size // 5))
                blob = _circle_mask(size, size, cy, cx, radius)
                mask[cls, blob] = 1.0
                image[0, blob] += 0.35 + 0.15 * cls
            np.savez_compressed(split_dir / f"case_{idx:03d}.npz", image=image, mask=mask)
