"""Perturbation support masks for Tier A (and the Hou et al. edge region).

A mask is float32 [1, T, H, W] in {0, 1} on the input's device: 1 = pixel may be perturbed.
Tier A varies ONLY the mask; the optimiser (src/attacks/pgd.py) is shared.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np
import torch


def sample_seed(seed: int, sample_id: str) -> int:
    """Stable per-sample seed (independent of probe order and Python hash randomisation)."""
    return (seed * 1_000_003 + zlib.crc32(sample_id.encode())) % (2 ** 31)


def dense_mask(x: torch.Tensor) -> torch.Tensor:
    """All pixels, all frames. x: [1, T, H, W] -> [1, T, H, W]."""
    return torch.ones_like(x)


def edge_mask(x: torch.Tensor, tau: float = 1.0, threshold: float = 0.5) -> torch.Tensor:
    """Hou et al. 2025 edge region: pixels whose distance to the silhouette boundary is < tau.

    The boundary lies between foreground and background pixels, so a pixel's distance to it is
    its cv2.distanceTransform distance to the nearest pixel of the other class minus 0.5. With
    tau = 1.0 this is the 1-pixel ring on each side of the contour (8-connected; a diagonal
    neighbour is at sqrt(2) - 0.5 < 1), matching the thin outline in Hou et al. Fig. 3 (choice recorded in docs/deviations.md).
    Computed on the binarised clean frame (> threshold). x: [1, T, H, W] -> [1, T, H, W].
    """
    fg = (x[0] > threshold).cpu().numpy().astype(np.uint8)           # [T, H, W]
    out = np.zeros(fg.shape, dtype=np.float32)
    for t, f in enumerate(fg):
        if f.all() or not f.any():          # no boundary in this frame
            continue
        d_in = cv2.distanceTransform(f, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)       # fg -> nearest bg
        d_out = cv2.distanceTransform(1 - f, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # bg -> nearest fg
        out[t] = (np.where(f == 1, d_in, d_out) - 0.5) < tau
    return torch.from_numpy(out).unsqueeze(0).to(x.device)


def frame_mask(x: torch.Tensor, frames: list[int] | np.ndarray) -> torch.Tensor:
    """All pixels in the given frames only. x: [1, T, H, W] -> [1, T, H, W]."""
    m = torch.zeros_like(x)
    m[:, list(frames)] = 1.0
    return m


def random_frames(n_frames: int, k: int, seed: int) -> list[int]:
    """k distinct frames uniformly at random (mirrors He et al.'s random K frames), sorted."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_frames, size=min(k, n_frames), replace=False).tolist())


def topk_grad_frames(grad: torch.Tensor, k: int) -> list[int]:
    """k frames with the largest L1 gradient magnitude. grad: [1, T, H, W]. Sorted indices."""
    score = grad[0].abs().flatten(1).sum(1)
    return sorted(score.topk(min(k, score.numel())).indices.tolist())
