"""Exactly parameter-matched ACIL feature variants."""

from __future__ import annotations

import torch

from experiments.acil_innovation_v1.acil import ACILBase


_FEATURE_SET = {
    "full": "full",
    "value_only": "value_only",
    "no_anchor": "no_anchor_values",
}


def new_acil(method: str, *, model_seed: int) -> ACILBase:
    """Construct one registered variant with an identical initial state."""

    if method not in _FEATURE_SET:
        raise ValueError("method must be full, value_only, or no_anchor")
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise TypeError("model_seed must be an integer")
    if model_seed < 0:
        raise ValueError("model_seed must be nonnegative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = ACILBase(feature_set=_FEATURE_SET[method])
    return model


__all__ = ["new_acil"]

