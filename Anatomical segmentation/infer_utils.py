from __future__ import annotations

import torch
import torch.nn.functional as F


def _divisor(depth: int) -> int:
    return 16 if int(depth) == 4 else 8


def _pad_to_multiple(image: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    h, w = image.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0, 0, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = F.pad(image, (left, right, top, bottom), mode="reflect")
    return padded, (left, right, top, bottom)


def _unpad(tensor: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pad
    h, w = tensor.shape[-2:]
    return tensor[..., top : h - bottom, left : w - right]
