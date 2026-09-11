"""Single registered construction path for every trainable v1 model."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from torch import nn

from .acil import ACILBase
from .jobs import PlannedJob, planned_jobs
from .manifest import active_manifest_stages
from .model_assets import load_pinned_gpt2_blocks
from .models import QueryResidualModel
from .oracle_model import TruthQDeepSets
from .registries import seed_bundle as registered_seed_bundle


_ACTIVE_MODEL_HANDLERS = frozenset(
    {
        ("stage0_acil_tune", "acil"),
        ("stage_h", "truth_q_deepsets"),
        ("stage_i", "local_loo"),
        ("stage_i", "global_loo"),
        ("full_tune", "full_u0"),
        ("full_tune", "full_scratch"),
        ("full_tune", "full_gpt2"),
    }
)


def require_active_registered_job(job: PlannedJob) -> PlannedJob:
    """Reject fabricated, future-formal, and otherwise unregistered jobs."""

    if not isinstance(job, PlannedJob):
        raise TypeError("job must be a PlannedJob")
    if job.stage not in active_manifest_stages():
        raise ValueError("job stage is outside the active manifest")
    if (job.stage, job.method) not in _ACTIVE_MODEL_HANDLERS:
        raise ValueError("job has no registered active model architecture")
    if job not in planned_jobs(job.stage):
        raise ValueError("job identity is absent from the active manifest registry")
    return job


def seed_runtime(seed: int, *, include_cuda: bool = True) -> None:
    """Apply the deterministic training seed policy at a model boundary."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("runtime seed must be a nonnegative integer")
    if not isinstance(include_cuda, bool):
        raise TypeError("include_cuda must be boolean")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if include_cuda and hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if include_cuda and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def new_residual_model(
    method: str,
    *,
    acil: ACILBase,
    model_seed: int,
    reseed: Callable[[int], object] | None = None,
) -> tuple[QueryResidualModel, tuple[nn.Parameter, ...]]:
    """Construct one residual architecture using the sole method registry.

    ``reseed`` is injectable only so CPU checkpoint validation can preserve the
    no-GPU boundary. Training uses the complete deterministic runtime seeder.
    """

    if not isinstance(acil, ACILBase):
        raise TypeError("acil must be an ACILBase")
    if isinstance(model_seed, bool) or not isinstance(model_seed, int) or model_seed < 0:
        raise ValueError("model_seed must be a nonnegative integer")
    if method == "local_loo":
        kind = "local"
        blocks = None
    elif method == "global_loo":
        kind = "deepsets"
        blocks = None
    elif method in {"full_u0", "full_gpt2"}:
        kind = "full_u0" if method == "full_u0" else "gpt2_set"
        blocks = load_pinned_gpt2_blocks("pretrained", model_seed)
    elif method == "full_scratch":
        kind = "gpt2_scratch"
        blocks = load_pinned_gpt2_blocks("random", model_seed)
    else:
        raise ValueError(f"unsupported registered residual method {method!r}")

    # Asset construction may consume the shared RNG. All newly trained heads
    # therefore start from the registered bundle independently of asset loading.
    (seed_runtime if reseed is None else reseed)(model_seed)
    model = QueryResidualModel(kind, acil, retained_blocks=blocks)
    if method in {"full_u0", "full_gpt2"}:
        pretrained = model.gpt_backbone_parameters()
    else:
        pretrained = ()
    return model, pretrained


def build_registered_model(job: PlannedJob) -> nn.Module:
    """Build the exact CPU architecture registered for one active job.

    This path never queries or seeds CUDA. It is used for semantic checkpoint
    validation, while the residual sub-factory is shared verbatim with training.
    """

    job = require_active_registered_job(job)
    model_seed = registered_seed_bundle(job.seed_bundle).model
    with torch.random.fork_rng(devices=[]):
        if job.stage == "stage0_acil_tune":
            torch.manual_seed(model_seed)
            return ACILBase()

        acil = ACILBase()
        if job.stage == "stage_h":
            torch.manual_seed(model_seed)
            return TruthQDeepSets(acil)

        model, _ = new_residual_model(
            job.method,
            acil=acil,
            model_seed=model_seed,
            reseed=torch.manual_seed,
        )
        return model


__all__ = [
    "build_registered_model",
    "new_residual_model",
    "require_active_registered_job",
    "seed_runtime",
]
