"""Shared CPU fixtures: a fake OpenGait DataSet and model with the same interfaces."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data import CasiaB
from src.evaluation import Gallery, GaitModel

VIEWS = ["000", "018", "036"]
TYPES = ["nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06", "bg-01", "bg-02", "cl-01", "cl-02"]
GALLERY = TYPES[:4]
PROBES = {"NM": ["nm-05", "nm-06"], "BG": ["bg-01", "bg-02"], "CL": ["cl-01", "cl-02"]}


class FakeDataSet:
    """Mimics OpenGait DataSet: seqs_info rows [label, type, view, paths]; item -> ([frames], info)."""

    def __init__(self, n_subjects: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.seqs_info, self.frames = [], []
        for s in range(n_subjects):
            base = rng.integers(0, 256, (64, 64))
            for t in TYPES:
                for v in VIEWS:
                    n = int(rng.integers(8, 14))
                    noise = rng.integers(-60, 60, (n, 64, 64))
                    self.frames.append(np.clip(base + noise, 0, 255).astype(np.uint8))
                    self.seqs_info.append([f"{s:03d}", t, v, []])

    def __getitem__(self, i):
        return [self.frames[i]], self.seqs_info[i]


class FakeNet(torch.nn.Module):
    """[N,1? T,H,W] -> embeddings [N, C=8, P=4], same output dict as OpenGait models."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(1, 8, 3, padding=1)

    def forward(self, inputs):
        x = inputs[0][0]                                  # [N, T, H, W]
        n, t, h, w = x.shape
        f = self.conv(x.reshape(n * t, 1, h, w)).reshape(n, t, 8, h, w).max(1).values
        emb = f.reshape(n, 8, 4, -1).mean(-1)            # 4 horizontal strips
        return {"inference_feat": {"embeddings": emb}}


@pytest.fixture()
def setup():
    data = CasiaB(FakeDataSet(), lambda f: f[..., 10:-10] / 255.0, GALLERY, PROBES)
    model = GaitModel(FakeNet(), "fake", "none", fp16_eval=False)
    gallery = Gallery.build(model, data)
    return model, data, gallery


