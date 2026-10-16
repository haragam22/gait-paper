"""Shared helpers: seeding, config loading, environment records, append-only CSVs,
OpenGait import setup. No hidden global state beyond what OpenGait itself requires
(a torch.distributed process group and its message manager)."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA); cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "local"


def git_commit(repo_dir: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def env_record(seed: int, opengait_dir: str | Path, extra: Mapping[str, Any] | None = None) -> dict:
    """Metadata every notebook records in its first cell and every run saves."""
    rec = {
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "platform": detect_platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "opengait_commit": git_commit(opengait_dir),
        "repo_commit": git_commit(REPO_ROOT),
    }
    rec.update(extra or {})
    return rec


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, default=str)


def append_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append rows to a CSV, writing the header only if the file is new. Never overwrites.
    Raises if the columns differ from the existing header (silent misalignment is worse)."""
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, newline="", encoding="utf8") as f:
            header = next(csv.reader(f))
        if header != fields:
            raise ValueError(f"{path}: columns {fields} != existing header {header}")
    with open(path, "a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def weights_digest(module: torch.nn.Module) -> str:
    """Hash of every parameter and buffer; used to assert attacks never change the model."""
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def setup_opengait(opengait_dir: str | Path) -> None:
    """Make OpenGait importable (its code uses top-level `utils`, `data`, `modeling`, ...).

    `opengait/modeling/models/__init__.py` imports every model in the repo, pulling in
    kornia, lightly, sklearn, etc. We register an empty package in its place so that only
    the model modules we import explicitly (baseline.py, gaitset.py) are loaded. OpenGait's
    code itself is used unmodified.
    """
    root = Path(opengait_dir).resolve() / "opengait"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "modeling.models" not in sys.modules:
        import modeling  # noqa: F401  (namespace package)
        pkg = types.ModuleType("modeling.models")
        pkg.__path__ = [str(root / "modeling" / "models")]
        sys.modules["modeling.models"] = pkg


def init_single_process_dist() -> None:
    """OpenGait's BaseModel calls torch.distributed.get_rank(); give it a world of size 1."""
    import torch.distributed as dist
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=0, world_size=1)
