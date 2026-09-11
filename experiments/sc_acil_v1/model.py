"""Full SC-ACIL and its exactly matched zero-innovation twin."""

from __future__ import annotations

import torch

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.model_factory import seed_runtime
from experiments.acil_innovation_v1.models import ModelFeatures, QueryResidualModel


class SCACILModel(QueryResidualModel):
    def __init__(self, acil: ACILBase, *, innovation_enabled: bool) -> None:
        if not isinstance(innovation_enabled, bool):
            raise TypeError("innovation_enabled must be boolean")
        super().__init__("local", acil)
        self.innovation_enabled = innovation_enabled

    def token_inputs(self, features: ModelFeatures) -> torch.Tensor:
        tokens = super().token_inputs(features)
        if self.innovation_enabled:
            return tokens
        zeroed = tokens.clone()
        zeroed[..., -2] = 0.0
        return zeroed


def new_sc_acil_model(
    method: str, *, acil: ACILBase, model_seed: int
) -> SCACILModel:
    if method not in {"sc_acil", "sc_acil_u0"}:
        raise ValueError("method must be sc_acil or sc_acil_u0")
    if not isinstance(acil, ACILBase):
        raise TypeError("acil must be ACILBase")
    seed_runtime(model_seed)
    return SCACILModel(acil, innovation_enabled=method == "sc_acil")


__all__ = ["SCACILModel", "new_sc_acil_model"]

