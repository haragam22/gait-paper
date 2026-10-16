"""Tier A: one L-inf PGD optimiser, three support masks (CLAUDE.md "Attack design").

    attack = PGD(eps=8/255, support=dense())                    # dense PGD
    attack = PGD(eps=8/255, support=edge(tau=1.0))              # edge-masked PGD
    attack = PGD(eps=8/255, support=temporal(k=3, rule="random_k"))
    evaluate_attack(model, data, gallery, probes, attack, ...)

Objective: maximise margin = d(nearest true) - d(nearest other) on the pooled cross-view gallery
(src.evaluation.margin), fp32 gradients, model frozen. After every step delta is projected onto
the L-inf ball, the support mask, and x + delta in [0, 1].
"""
from __future__ import annotations

from typing import Callable

import torch

from ..evaluation import AttackTarget, GaitModel, margin
from . import masks as M

# support(model, x, target, seed) -> (mask [1, T, H, W], info dict logged per sample)
Support = Callable[[GaitModel, torch.Tensor, AttackTarget, int], tuple[torch.Tensor, dict]]


def margin_grad(model: GaitModel, x: torch.Tensor, target: AttackTarget) -> tuple[float, torch.Tensor]:
    """Margin and its gradient w.r.t. the input. x: [1, T, H, W] -> (float, [1, T, H, W])."""
    x = x.detach().requires_grad_(True)
    loss = margin(model.embed(x, grad=True), target, eps=1e-12).sum()
    (g,) = torch.autograd.grad(loss, x)
    return float(loss.detach()), g


def dense() -> Support:
    return lambda model, x, target, seed: (M.dense_mask(x), {"support": "dense", "frames_selected": ""})


def edge(tau: float = 1.0) -> Support:
    return lambda model, x, target, seed: (M.edge_mask(x, tau), {"support": f"edge_tau{tau:g}", "frames_selected": ""})


def temporal(k: int, rule: str) -> Support:
    """rule: "random_k" (seeded per sample, mirrors He et al.) or "topk_grad" (|grad| at the clean input)."""
    def support(model, x, target, seed):
        if rule == "random_k":
            frames = M.random_frames(x.shape[1], k, M.sample_seed(seed, target.sample_id))
        elif rule == "topk_grad":
            frames = M.topk_grad_frames(margin_grad(model, x, target)[1], k)
        else:
            raise ValueError(rule)
        return M.frame_mask(x, frames), {"support": f"temporal_{rule}_k{k}",
                                         "frames_selected": ";".join(map(str, frames))}
    return support


class PGD:
    """L-inf PGD restricted to `support`. Callable as attack(model, x, target) -> x_adv.

    eps: L-inf budget in [0, 1] units (e.g. 8/255). step = eps * step_ratio. One random start
    (uniform in the masked eps-ball, seeded per sample). `info` holds the last sample's log
    fields; evaluate_attack copies them into its row."""

    def __init__(self, eps: float, support: Support | None = None, steps: int = 50,
                 step_ratio: float = 0.1, random_start: bool = True, seed: int = 0):
        self.eps, self.steps, self.alpha = eps, steps, eps * step_ratio
        self.support, self.random_start, self.seed = support or dense(), random_start, seed
        self.info: dict = {}

    def __call__(self, model: GaitModel, x: torch.Tensor, target: AttackTarget) -> torch.Tensor:
        mask, self.info = self.support(model, x, target, self.seed)
        delta = torch.zeros_like(x)
        if self.random_start and self.eps > 0:
            g = torch.Generator(device=x.device).manual_seed(M.sample_seed(self.seed, target.sample_id))
            delta = (torch.rand(x.shape, generator=g, device=x.device) * 2 - 1) * self.eps * mask
            delta = (x + delta).clamp(0, 1) - x
        for _ in range(self.steps if self.eps > 0 else 0):
            _, grad = margin_grad(model, x + delta, target)
            delta = (delta + self.alpha * grad.sign() * mask).clamp(-self.eps, self.eps)
            delta = (x + delta).clamp(0, 1) - x
        return (x + delta).detach()


def run_tier_a(model: GaitModel, data, gallery, probes: list, family: str, support: Support, e255: float,
               atk: dict, seed: int, out_csv, subset_path, run_meta: dict | None = None):
    """One Tier A sweep point: PGD with `support` at eps = e255/255 on `probes`, through
    evaluate_attack. `family` names the run (e.g. "dense", "edge", "temporal_random_k_k3");
    run_id = pgd_{family}-eps{e255}-{checkpoint_id}, so re-running resumes. `atk` =
    configs/attacks.yaml. Returns this run's rows (pandas DataFrame)."""
    from ..evaluation import evaluate_attack
    cfg = {"family": family, "eps": e255 / 255, "eps_255": e255, "steps": atk["steps"],
           "step_size_ratio": atk["step_size_ratio"], "random_start": atk["random_start"],
           "subset": str(subset_path)}
    attack = PGD(e255 / 255, support, atk["steps"], atk["step_size_ratio"], atk["random_start"], seed)
    return evaluate_attack(model, data, gallery, probes, attack, attack_name=f"pgd_{family}", tier="A",
                           config=cfg, seed=seed, out_csv=out_csv,
                           run_id=f"pgd_{family}-eps{e255:g}-{model.checkpoint_id}", run_meta=run_meta)
