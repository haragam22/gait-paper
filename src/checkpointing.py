"""Resumable-job helpers for free Kaggle/Colab sessions.

Usage in a long job:
    state = load_latest(ckpt_dir) or fresh_state
    timer = CheckpointTimer(minutes=30)
    for step in range(state["step"], total):
        ...
        if timer.due():
            save_checkpoint({...,"step": step + 1}, ckpt_dir, step + 1)
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import torch

_PAT = re.compile(r"ckpt_(\d+)\.pt$")


def save_checkpoint(state: dict[str, Any], ckpt_dir: str | Path, step: int, keep: int = 2) -> Path:
    """Atomically write `ckpt_{step}.pt` (write tmp, then rename) and keep the newest `keep`."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"ckpt_{step:08d}.pt"
    tmp = final.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, final)
    for old in list_checkpoints(ckpt_dir)[:-keep]:
        old.unlink(missing_ok=True)
    return final


def list_checkpoints(ckpt_dir: str | Path) -> list[Path]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return []
    found = [p for p in ckpt_dir.iterdir() if _PAT.search(p.name)]
    return sorted(found, key=lambda p: int(_PAT.search(p.name).group(1)))


def load_latest(ckpt_dir: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
    """Return the newest checkpoint's state dict, or None if there is none."""
    ckpts = list_checkpoints(ckpt_dir)
    return torch.load(ckpts[-1], map_location=map_location, weights_only=True) if ckpts else None


class CheckpointTimer:
    """`due()` is True once every `minutes` (CLAUDE.md: checkpoint at least every 30 min)."""

    def __init__(self, minutes: float = 30.0):
        self.period = minutes * 60
        self.last = time.monotonic()

    def due(self) -> bool:
        if time.monotonic() - self.last >= self.period:
            self.last = time.monotonic()
            return True
        return False
