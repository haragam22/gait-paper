"""CPU tests for src/evaluation.py. Run: python -m pytest tests -q

- zero perturbation reproduces clean results exactly (CLAUDE.md rule 6)
- an attack that touches the weights is caught (rule 5)
- our per-probe protocol + aggregate_rank1 == OpenGait's own evaluate_indoor_dataset (rule 1)
- resuming a run skips finished probes and never overwrites the CSV (rule 4)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import CasiaB, stratified_subset
from src.evaluation import (AttackTarget, Gallery, GaitModel, aggregate_rank1, evaluate_attack,
                            margin, part_dist, probe_metrics)

OPENGAIT = Path(__file__).resolve().parents[2] / "OpenGait"
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


def test_part_dist_matches_bruteforce():
    a, b = torch.randn(3, 5, 4), torch.randn(6, 5, 4)
    ref = torch.stack([torch.stack([(a[i] - b[j]).norm(dim=0).mean() for j in range(6)]) for i in range(3)])
    assert torch.allclose(part_dist(a, b), ref, atol=1e-5)


def test_zero_perturbation_reproduces_clean(setup):
    model, data, gallery = setup
    clean = evaluate_attack(model, data, gallery, data.probes)
    zero = evaluate_attack(model, data, gallery, data.probes, lambda m, x, t: x.clone(), attack_name="zero")
    for c in ["clean_rank1", "clean_rank1_viewavg", "clean_margin"]:
        assert (clean[c].values == zero[c].values).all()
    assert (zero["adv_rank1"] == zero["clean_rank1"]).all()
    assert (zero["adv_margin"] == zero["clean_margin"]).all()
    assert (zero["linf_actual"] == 0).all() and (zero["n_pixels_perturbed"] == 0).all()
    assert not zero["attack_success"].any()


def test_weight_change_is_caught(setup):
    model, data, gallery = setup

    def bad_attack(m, x, t):
        with torch.no_grad():
            m.net.conv.weight.add_(1e-3)
        return x

    with pytest.raises(AssertionError, match="weights changed"):
        evaluate_attack(model, data, gallery, data.probes[:2], bad_attack)


def test_out_of_range_attack_rejected(setup):
    model, data, gallery = setup
    with pytest.raises(AssertionError, match="outside"):
        evaluate_attack(model, data, gallery, data.probes[:1], lambda m, x, t: x + 2.0)


def test_margin_sign_matches_rank1(setup):
    model, data, gallery = setup
    for rec in data.probes[:10]:
        e = model.embed(data.load(rec))
        m = probe_metrics(e, gallery, rec)
        assert m["rank1"] == (m["margin"] < 0)
        assert abs(float(margin(e, AttackTarget.for_probe(gallery, rec))[0]) - m["margin"]) < 1e-5


def test_resume_appends_and_skips(setup, tmp_path):
    model, data, gallery = setup
    csv = tmp_path / "raw" / "clean.csv"
    evaluate_attack(model, data, gallery, data.probes[:5], out_csv=csv, run_id="r1")
    df = evaluate_attack(model, data, gallery, data.probes[:9], out_csv=csv, run_id="r1")
    assert len(df) == 9 and df["sample_id"].is_unique
    assert len(pd.read_csv(csv)) == 9
    assert (tmp_path / "raw" / "runs" / "r1.json").exists()


def test_stratified_subset_is_seeded_and_balanced(setup):
    _, data, _ = setup
    a = stratified_subset(data.probes, 6, seed=0)
    b = stratified_subset(data.probes, 6, seed=0)
    assert [r.sample_id for r in a] == [r.sample_id for r in b]
    assert pd.Series([r.condition for r in a]).value_counts().to_dict() == {"NM": 6, "BG": 6, "CL": 6}


@pytest.mark.skipif(not OPENGAIT.exists(), reason="OpenGait clone not found next to repo")
def test_aggregate_equals_opengait_evaluator(setup, monkeypatch):
    """Our protocol must give OpenGait's exact CASIA-B numbers on the same embeddings."""
    from src.utils import setup_opengait
    setup_opengait(OPENGAIT)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: 0)  # no process group needed
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)  # run OpenGait's cuda_dist on CPU
    import utils as og_utils
    og_utils.get_msg_mgr().init_logger("/tmp", False)
    from evaluation.evaluator import evaluate_indoor_dataset

    model, data, gallery = setup
    emb = np.concatenate([model.embed(data.load(r)).numpy() for r in data.records])
    info = {"embeddings": emb, "labels": [r.subject for r in data.records],
            "types": [r.seq_type for r in data.records], "views": [r.view for r in data.records]}
    official = evaluate_indoor_dataset(info, "CASIA-B", metric="euc")

    ours = aggregate_rank1(evaluate_attack(model, data, gallery, data.probes))
    for cond in ["NM", "BG", "CL"]:
        assert ours.loc[cond, "mean"] == pytest.approx(official[f"scalar/test_accuracy/{cond}@R1"], abs=0.01)
