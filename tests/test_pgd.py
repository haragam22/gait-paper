"""Notebook 02 step 1 unit tests for Tier A PGD (CPU, fake model from conftest):
eps=0 reproduces clean; ||delta||_inf <= eps; x+delta in [0, 1]; weights unchanged (asserted
inside evaluate_attack); seed-stable; perturbation stays inside the support mask."""
from __future__ import annotations

import pytest
import torch

from src.attacks import masks as M
from src.attacks.pgd import PGD, dense, edge, temporal
from src.evaluation import AttackTarget, evaluate_attack

EPS = 16 / 255


def test_eps_zero_reproduces_clean(setup):
    model, data, gallery = setup
    clean = evaluate_attack(model, data, gallery, data.probes[:12])
    zero = evaluate_attack(model, data, gallery, data.probes[:12], PGD(0.0, steps=5), attack_name="pgd0")
    assert (zero["adv_margin"].values == clean["clean_margin"].values).all()
    assert (zero["linf_actual"] == 0).all() and not zero["attack_success"].any()


@pytest.mark.parametrize("support", [dense(), edge(1.0), temporal(2, "random_k"), temporal(2, "topk_grad")])
def test_budget_range_and_support(setup, support):
    model, data, gallery = setup
    attack = PGD(EPS, support, steps=5)
    for rec in data.probes[:4]:
        x = data.load(rec)
        target = AttackTarget.for_probe(gallery, rec)
        x_adv = attack(model, x, target)
        delta = x_adv - x
        assert float(delta.abs().max()) <= EPS + 1e-6
        assert 0.0 <= float(x_adv.min()) and float(x_adv.max()) <= 1.0
        mask, _ = support(model, x, target, 0)
        assert float((delta * (1 - mask)).abs().max()) == 0.0     # nothing outside support
        if attack.info["frames_selected"]:
            frames = [int(f) for f in attack.info["frames_selected"].split(";")]
            assert len(frames) == 2 and set(delta[0].flatten(1).abs().sum(1).nonzero().flatten().tolist()) <= set(frames)


def test_seed_stable_and_rows_logged(setup):
    model, data, gallery = setup
    rec = data.probes[0]
    x, target = data.load(rec), AttackTarget.for_probe(gallery, rec)
    a = PGD(EPS, temporal(3, "random_k"), steps=3, seed=1)
    assert torch.equal(a(model, x, target), a(model, x, target))
    df = evaluate_attack(model, data, gallery, data.probes[:3], a, attack_name="t")
    assert {"support", "frames_selected"} <= set(df.columns)


def test_pgd_increases_margin(setup):
    model, data, gallery = setup
    df = evaluate_attack(model, data, gallery, data.probes[:12], PGD(EPS, steps=10), attack_name="pgd")
    assert (df["adv_margin"] > df["clean_margin"]).mean() > 0.9


def test_edge_mask_is_one_pixel_ring_each_side():
    x = torch.zeros(1, 1, 20, 20)
    x[0, 0, 5:15, 5:15] = 1.0                       # 10x10 square
    m = M.edge_mask(x, tau=1.0)[0, 0]
    assert int(m[5:15, 5:15].sum()) == 36            # inner ring
    assert int(m.sum()) - 36 == 44                   # outer 8-connected ring
    assert m[7:13, 7:13].sum() == 0 and m[0].sum() == 0
    assert M.edge_mask(torch.zeros(1, 2, 8, 8)).sum() == 0     # empty frames: no edge
