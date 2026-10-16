"""CASIA-B access through OpenGait's own DataSet and transforms (never reimplemented).

A sequence tensor is float32 in [0, 1], shape [1, T, H, W] = [1, T, 64, 44] after
OpenGait's BaseSilCuttingTransform (pkl frames are 64x64 uint8; 10 px cut each side).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CONDITIONS = ("NM", "BG", "CL")


@dataclass(frozen=True)
class SeqRecord:
    """One test sequence. `index` is its position in the OpenGait test DataSet."""
    index: int
    subject: str
    seq_type: str   # e.g. "nm-05"
    view: str       # e.g. "090"

    @property
    def sample_id(self) -> str:
        return f"{self.subject}-{self.seq_type}-{self.view}"

    @property
    def condition(self) -> str:
        return self.seq_type[:2].upper()


class CasiaB:
    """Test split of CASIA-B as OpenGait sees it, with gallery/probe partition.

    Args:
        dataset: an OpenGait `data.dataset.DataSet` built with training=False.
        transform: OpenGait evaluator transform (callable on a [T, 64, 64] uint8 array).
        gallery_types / probe_types: from configs/baseline.yaml.
        frames_all_limit: OpenGait evaluator sampler limit (-1 for none); first N frames kept.
    """

    def __init__(self, dataset, transform, gallery_types: list[str],
                 probe_types: dict[str, list[str]], frames_all_limit: int = -1):
        self.dataset = dataset
        self.transform = transform
        self.frames_all_limit = frames_all_limit
        self.records = [SeqRecord(i, s[0], s[1], s[2]) for i, s in enumerate(dataset.seqs_info)]
        self.gallery = [r for r in self.records if r.seq_type in gallery_types]
        probe_set = {t for ts in probe_types.values() for t in ts}
        self.probes = [r for r in self.records if r.seq_type in probe_set]

    def load(self, rec: SeqRecord) -> torch.Tensor:
        """Sequence as float32 [1, T, 64, 44] in [0, 1], frames in natural order
        (OpenGait `all_ordered` sampling)."""
        data_list, _ = self.dataset[rec.index]
        frames = data_list[0]
        if 0 < self.frames_all_limit < len(frames):
            frames = frames[: self.frames_all_limit]
        x = np.asarray([self.transform(f) for f in frames], dtype=np.float32)
        return torch.from_numpy(x).unsqueeze(0)

    def n_frames(self, rec: SeqRecord) -> int:
        return len(self.dataset[rec.index][0][0])

    def summary(self) -> pd.DataFrame:
        """One row per sequence: subject, type, view, condition, role."""
        g = {r.index for r in self.gallery}
        p = {r.index for r in self.probes}
        return pd.DataFrame([{
            "index": r.index, "sample_id": r.sample_id, "subject": r.subject,
            "seq_type": r.seq_type, "view": r.view, "condition": r.condition,
            "role": "gallery" if r.index in g else "probe" if r.index in p else "unused",
        } for r in self.records])


def stratified_subset(probes: list[SeqRecord], n_per_condition: int, seed: int) -> list[SeqRecord]:
    """Seeded subset with exactly `n_per_condition` probes from each of NM/BG/CL,
    spread across views by sampling within (condition, view) cells round-robin."""
    rng = np.random.default_rng(seed)
    chosen: list[SeqRecord] = []
    for cond in CONDITIONS:
        cells: dict[str, list[SeqRecord]] = {}
        for r in probes:
            if r.condition == cond:
                cells.setdefault(r.view, []).append(r)
        if sum(map(len, cells.values())) < n_per_condition:
            raise ValueError(f"only {sum(map(len, cells.values()))} {cond} probes")
        for v in cells:
            cells[v] = [cells[v][i] for i in rng.permutation(len(cells[v]))]
        views = sorted(cells)
        picked: list[SeqRecord] = []
        while len(picked) < n_per_condition:
            for v in views:
                if cells[v] and len(picked) < n_per_condition:
                    picked.append(cells[v].pop())
        chosen += sorted(picked, key=lambda r: r.index)
    return chosen


def save_subset(subset: list[SeqRecord], path: str | Path) -> None:
    """Persist a subset; this file (not the seed) is the source of truth shared by all attacks."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; subsets are fixed once written")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"index": r.index, "sample_id": r.sample_id, "subject": r.subject,
                   "seq_type": r.seq_type, "view": r.view, "condition": r.condition}
                  for r in subset]).to_csv(path, index=False)


def load_subset(path: str | Path, data: CasiaB) -> list[SeqRecord]:
    """Reload a saved subset and check it still matches the dataset index."""
    df = pd.read_csv(path, dtype=str)
    by_id = {r.sample_id: r for r in data.records}
    missing = [s for s in df["sample_id"] if s not in by_id]
    if missing:
        raise KeyError(f"subset samples not in dataset: {missing[:5]}")
    return [by_id[s] for s in df["sample_id"]]
