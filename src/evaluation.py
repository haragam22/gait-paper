"""Model loading, the shared threat-model objective, and THE evaluation entry point.

Every attack (Tier A and Tier B) is evaluated through `evaluate_attack`; clean evaluation
is `attack=None`. An attack is any callable

    attack(model: GaitModel, x: Tensor[1, T, H, W], target: AttackTarget) -> Tensor[1, T, H, W]

returning the adversarial sequence (same shape, values in [0, 1]).

Protocol (matches OpenGait `single_view_gallery_evaluation`, CASIA-B):
  gallery = nm-01..04, probes = nm-05/06, bg-01/02, cl-01/02. For each probe and each gallery
  view != probe view, Rank-1 against the gallery of that single view; a probe's
  `rank1_viewavg` is the mean over those 10 gallery views. Condition accuracy = mean over
  probe views of mean over probes of `rank1_viewavg` — identical to OpenGait's de_diag mean.
  The attack objective and `attack_success` use the pooled cross-view gallery (all gallery
  sequences whose view != probe view): `rank1` = nearest pooled gallery sample is the true
  identity. See CLAUDE.md "Threat model".
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from .data import CasiaB, SeqRecord
from .utils import (append_rows, file_sha256, init_single_process_dist, load_yaml, save_json,
                    setup_opengait, weights_digest)

MODEL_CLASSES = {"Baseline": ("modeling.models.baseline", "Baseline"),
                 "GaitSet": ("modeling.models.gaitset", "GaitSet")}


# --------------------------------------------------------------------------- model

class GaitModel:
    """Frozen OpenGait model. `embed` returns embeddings [N, C, P] float32.

    grad=False: evaluation forward, no_grad, fp16 autocast iff the OpenGait evaluator config
                enables it (so clean numbers match OpenGait).
    grad=True : fp32 forward with autograd, for attack gradients (CLAUDE.md compute rules).
    """

    def __init__(self, net: torch.nn.Module, name: str, checkpoint: str, fp16_eval: bool):
        self.net = net.eval()
        self.net.requires_grad_(False)
        self.name = name
        self.checkpoint = checkpoint
        self.checkpoint_id = file_sha256(checkpoint)[:12] if Path(checkpoint).exists() else "none"
        self.fp16_eval = fp16_eval
        self.device = next(net.parameters()).device

    def embed(self, x: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """x: [N, T, H, W] float in [0, 1] (N sequences of equal length) -> [N, C, P]."""
        assert not self.net.training, "model must stay in eval() mode"
        inputs = ([x.to(self.device)], None, None, None, None)
        if grad:
            return self.net(inputs)["inference_feat"]["embeddings"].float()
        with torch.no_grad(), torch.autocast("cuda", enabled=self.fp16_eval and self.device.type == "cuda"):
            return self.net(inputs)["inference_feat"]["embeddings"].float()

    def digest(self) -> str:
        return weights_digest(self.net)


def build_opengait_cfg(opengait_dir: Path, model_config: str, dataset_root: str,
                       checkpoint: str) -> dict:
    """OpenGait config (model yaml merged over configs/default.yaml, as config_loader does)
    with our paths, a single-sequence evaluator batch, and the checkpoint path as restore_hint."""
    from utils import MergeCfgsDict  # OpenGait's
    cfg = load_yaml(opengait_dir / "configs" / "default.yaml")
    MergeCfgsDict(load_yaml(opengait_dir / model_config), cfg)
    cfg["data_cfg"]["dataset_root"] = dataset_root
    cfg["data_cfg"]["dataset_partition"] = str(opengait_dir / "datasets/CASIA-B/CASIA-B.json")
    cfg["data_cfg"]["num_workers"] = 0
    cfg["evaluator_cfg"]["restore_hint"] = checkpoint
    cfg["evaluator_cfg"]["sampler"]["batch_size"] = 1  # one sequence per forward, as in OpenGait test
    return cfg


def load_model(baseline_cfg: dict, model_key: str) -> tuple[GaitModel, CasiaB, dict]:
    """Build an OpenGait model from configs/baseline.yaml `models[model_key]`, load its
    released checkpoint (strict), freeze it, and wrap the OpenGait test DataSet.
    Returns (model, data, opengait_cfg)."""
    import importlib
    opengait_dir = Path(baseline_cfg["opengait_dir"]).resolve()
    setup_opengait(opengait_dir)
    init_single_process_dist()
    from utils import get_msg_mgr  # OpenGait's
    mm = get_msg_mgr()
    if not hasattr(mm, "logger"):
        mm.init_logger(str(opengait_dir / "output"), False)

    spec = baseline_cfg["models"][model_key]
    cfg = build_opengait_cfg(opengait_dir, spec["model_config"],
                             baseline_cfg["dataset_root"], spec["checkpoint"])
    module, cls = MODEL_CLASSES[cfg["model_cfg"]["model"]]
    net = getattr(importlib.import_module(module), cls)(cfg, training=False)
    model = GaitModel(net, model_key, spec["checkpoint"], bool(cfg["evaluator_cfg"]["enable_float16"]))
    data = CasiaB(net.test_loader.dataset, net.evaluator_trfs[0],
                  baseline_cfg["gallery_types"], baseline_cfg["probe_types"],
                  cfg["evaluator_cfg"]["sampler"].get("frames_all_limit", -1))
    return model, data, cfg


# --------------------------------------------------------------------------- distances

def part_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Part-averaged Euclidean distance, same formula as OpenGait `cuda_dist(metric='euc')`.
    a: [N, C, P], b: [G, C, P] -> [N, G]. Use eps > 0 only inside attack losses
    (sqrt has an infinite gradient at 0)."""
    a, b = a.permute(2, 0, 1), b.permute(2, 0, 1)            # [P, N, C], [P, G, C]
    sq = (a ** 2).sum(-1).unsqueeze(2) + (b ** 2).sum(-1).unsqueeze(1) - 2 * a @ b.transpose(1, 2)
    return torch.sqrt(torch.relu(sq) + eps).mean(0)


@dataclass
class Gallery:
    """Clean gallery embeddings. emb: [G, C, P] float32 on the model device."""
    emb: torch.Tensor
    labels: np.ndarray   # [G] subject ids (str)
    views: np.ndarray    # [G] view strings

    @classmethod
    def build(cls, model: GaitModel, data: CasiaB, cache: Optional[str | Path] = None) -> "Gallery":
        """Embed every gallery sequence once; cached per model checkpoint id."""
        cache = Path(cache) if cache else None
        if cache and cache.exists():
            z = np.load(cache, allow_pickle=False)
            if str(z["checkpoint_id"]) == model.checkpoint_id:
                return cls(torch.from_numpy(z["emb"]).to(model.device), z["labels"], z["views"])
        emb = torch.cat([model.embed(data.load(r)) for r in data.gallery])
        g = cls(emb, np.array([r.subject for r in data.gallery]), np.array([r.view for r in data.gallery]))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, emb=emb.cpu().numpy(), labels=g.labels, views=g.views,
                     checkpoint_id=model.checkpoint_id)
        return g


@dataclass
class AttackTarget:
    """What an attack needs about one probe. gallery_emb: [G', C, P] pooled cross-view
    gallery (identical view excluded); is_true: [G'] bool, True where label == probe label."""
    label: str
    view: str
    gallery_emb: torch.Tensor
    is_true: torch.Tensor

    @classmethod
    def for_probe(cls, gallery: Gallery, rec: SeqRecord) -> "AttackTarget":
        keep = gallery.views != rec.view
        is_true = torch.from_numpy(gallery.labels[keep] == rec.subject).to(gallery.emb.device)
        return cls(rec.subject, rec.view, gallery.emb[torch.from_numpy(keep).to(gallery.emb.device)], is_true)


def margin(emb: torch.Tensor, target: AttackTarget, eps: float = 0.0) -> torch.Tensor:
    """Untargeted objective to MAXIMISE: d(nearest true-identity gallery) - d(nearest other).
    emb: [N, C, P] -> [N]. > 0 means the probe is misidentified."""
    d = part_dist(emb, target.gallery_emb, eps)
    inf = torch.finfo(d.dtype).max
    d_true = d.masked_fill(~target.is_true, inf).min(1).values
    d_other = d.masked_fill(target.is_true, inf).min(1).values
    return d_true - d_other


def probe_metrics(emb: torch.Tensor, gallery: Gallery, rec: SeqRecord) -> dict:
    """emb: [1, C, P]. rank1 (pooled cross-view), rank1_viewavg (OpenGait), margin."""
    d = part_dist(emb, gallery.emb)[0].cpu().numpy()
    correct = gallery.labels == rec.subject
    per_view = []
    for gv in np.unique(gallery.views):
        if gv == rec.view:
            continue
        idx = np.flatnonzero(gallery.views == gv)
        per_view.append(bool(correct[idx[np.argmin(d[idx])]]))
    pooled = np.flatnonzero(gallery.views != rec.view)
    dp, cp = d[pooled], correct[pooled]
    return {"rank1": bool(cp[np.argmin(dp)]),
            "rank1_viewavg": float(np.mean(per_view)),
            "margin": float(dp[cp].min() - dp[~cp].min())}


# --------------------------------------------------------------------------- evaluate_attack

Attack = Callable[[GaitModel, torch.Tensor, AttackTarget], torch.Tensor]


def perturbation_stats(x: torch.Tensor, x_adv: torch.Tensor) -> dict:
    """x, x_adv: [1, T, H, W]. Actual (not nominal) perturbation size."""
    delta = (x_adv - x)[0]
    changed = delta != 0
    return {"linf_actual": float(delta.abs().max()),
            "l2_actual": float(delta.flatten().norm()),
            "n_pixels_perturbed": int(changed.sum()),
            "frac_perturbed": float(changed.float().mean()),
            "n_frames_perturbed": int(changed.flatten(1).any(1).sum())}


def evaluate_attack(model: GaitModel, data: CasiaB, gallery: Gallery, probes: list[SeqRecord],
                    attack: Optional[Attack] = None, *, attack_name: str = "clean", tier: str = "-",
                    config: Optional[dict] = None, seed: int = 0, out_csv: Optional[str | Path] = None,
                    run_id: Optional[str] = None, run_meta: Optional[dict] = None) -> pd.DataFrame:
    """Evaluate `attack` (None = clean) on `probes`; one row per probe.

    Rows are appended to `out_csv` one at a time, so a disconnect loses at most one probe.
    Passing the same `run_id` again resumes: probes already in `out_csv` for that run are
    skipped. Run metadata (config, seed, env, checkpoint id) goes to results/raw/runs/{run_id}.json
    next to `out_csv`. Asserts the model weights are unchanged afterwards.
    """
    config = dict(config or {})
    run_id = run_id or f"{attack_name}-{uuid.uuid4().hex[:8]}"
    done: set[str] = set()
    if out_csv:
        out_csv = Path(out_csv)
        meta_path = out_csv.parent / "runs" / f"{run_id}.json"
        if not meta_path.exists():
            save_json({"run_id": run_id, "attack": attack_name, "tier": tier, "seed": seed,
                       "config": config, "model": model.name, "checkpoint": model.checkpoint,
                       "checkpoint_id": model.checkpoint_id, "n_probes": len(probes),
                       **(run_meta or {})}, meta_path)
        if out_csv.exists():
            prev = pd.read_csv(out_csv, usecols=["run_id", "sample_id"], dtype=str)
            done = set(prev.loc[prev["run_id"] == run_id, "sample_id"])

    digest_before = model.digest()
    rows = []
    for rec in probes:
        if rec.sample_id in done:
            continue
        x = data.load(rec).to(model.device)
        e_clean = model.embed(x)
        clean = probe_metrics(e_clean, gallery, rec)
        if attack is None:
            x_adv, e_adv, adv = x, e_clean, clean
        else:
            x_adv = attack(model, x, AttackTarget.for_probe(gallery, rec)).detach()
            assert x_adv.shape == x.shape, f"attack changed shape {tuple(x.shape)} -> {tuple(x_adv.shape)}"
            assert 0.0 <= float(x_adv.min()) and float(x_adv.max()) <= 1.0, "x_adv outside [0, 1]"
            e_adv = model.embed(x_adv)
            adv = probe_metrics(e_adv, gallery, rec)
        row = {"run_id": run_id, "attack": attack_name, "tier": tier, "model": model.name,
               "checkpoint_id": model.checkpoint_id, "seed": seed, "sample_id": rec.sample_id,
               "subject": rec.subject, "seq_type": rec.seq_type, "condition": rec.condition,
               "probe_view": rec.view, "n_frames": int(x.shape[1]),
               "eps": config.get("eps", 0), "steps": config.get("steps", 0),
               "clean_rank1": clean["rank1"], "adv_rank1": adv["rank1"],
               "attack_success": clean["rank1"] and not adv["rank1"],
               "clean_rank1_viewavg": clean["rank1_viewavg"], "adv_rank1_viewavg": adv["rank1_viewavg"],
               "clean_margin": clean["margin"], "adv_margin": adv["margin"],
               "embedding_shift_l2": float((e_adv - e_clean).flatten().norm()),
               **perturbation_stats(x, x_adv)}
        rows.append(row)
        if out_csv:
            append_rows(out_csv, [row])
    assert model.digest() == digest_before, "model weights changed during attack run"
    df = pd.DataFrame(rows)
    if out_csv and out_csv.exists():  # include rows from earlier (resumed) sessions
        df = read_results(out_csv)
        df = df[df["run_id"] == run_id].reset_index(drop=True)
    return df


ID_COLUMNS = ["run_id", "attack", "tier", "model", "checkpoint_id", "sample_id", "subject",
              "seq_type", "condition", "probe_view"]


def read_results(path: str | Path) -> pd.DataFrame:
    """Read a results CSV keeping ids as strings (views like "000" must not become 0)."""
    return pd.read_csv(path, dtype={c: str for c in ID_COLUMNS})


def aggregate_rank1(df: pd.DataFrame, col: str = "clean_rank1_viewavg") -> pd.DataFrame:
    """OpenGait-style Rank-1 (%) per condition: mean over probe views of the per-view mean.
    Returns a frame indexed by condition with columns = probe views + 'mean'."""
    per_view = df.groupby(["condition", "probe_view"])[col].mean().unstack() * 100
    per_view["mean"] = per_view.mean(1)
    return per_view


def official_opengait_rank1(model: GaitModel, data: CasiaB) -> dict:
    """Run OpenGait's own `evaluate_indoor_dataset` on embeddings of every test sequence.

    Returns {"fp32": {"NM@R1": ...}, "fp16": {...}}. Embeddings are the same (fp16 autocast
    values); only the distance precision differs. `evaluate_attack` computes distances in fp32,
    so it must equal "fp32" exactly. "fp16" is OpenGait's stock pipeline (it keeps the fp16
    dtype into `cuda_dist`); near-tie rounding makes it differ by a few hundredths of a pp
    (measured 2026-10-16: <= 0.03 pp; docs/deviations.md)."""
    from evaluation.evaluator import evaluate_indoor_dataset  # OpenGait's
    emb = np.concatenate([model.embed(data.load(r)).cpu().numpy() for r in data.records])
    info = {"labels": [r.subject for r in data.records],
            "types": [r.seq_type for r in data.records], "views": [r.view for r in data.records]}
    out = {}
    for name, e in [("fp32", emb), ("fp16", emb.astype(np.float16) if model.fp16_eval else emb)]:
        res = evaluate_indoor_dataset({"embeddings": e, **info}, "CASIA-B", metric="euc")
        out[name] = {k.split("/")[-1]: float(v) for k, v in res.items()}
    return out


def gate_rows(model_name: str, table: pd.DataFrame, official: dict, reported: dict,
              tolerance_pct: float) -> list[dict]:
    """Notebook 01 gate, one row per condition. `table` from `aggregate_rank1`, `official` from
    `official_opengait_rank1`. PASS needs matches_evaluator (exact vs OpenGait's evaluator at
    fp32 distance) and within_tolerance (vs reported numbers)."""
    rows = []
    for cond in ["NM", "BG", "CL"]:
        ours, og, og16, rep = (table.loc[cond, "mean"], official["fp32"][f"{cond}@R1"],
                               official["fp16"][f"{cond}@R1"], reported[cond])
        rows.append(dict(model=model_name, condition=cond, ours=round(ours, 2),
                         opengait_evaluator=round(og, 2), opengait_evaluator_fp16=round(og16, 2),
                         reported=rep, diff_vs_reported=round(ours - rep, 2),
                         matches_evaluator=abs(ours - og) < 0.01,
                         within_tolerance=abs(ours - rep) <= tolerance_pct))
    return rows
