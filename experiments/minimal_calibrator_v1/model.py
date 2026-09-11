"""Exactly matched static and input-conditioned interpolation calibrators."""

from __future__ import annotations

import torch

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.acil_core import (
    ObservationGeometryExtractorV2,
)


_METHODS = ("static_zero", "blinear_only", "value_only")


class CalibrationFeatureExtractor(ObservationGeometryExtractorV2):
    """Apply one fixed experiment-registered mask to the 16 ACIL channels."""

    def __init__(self, method: str) -> None:
        if method not in _METHODS:
            raise ValueError(f"unsupported calibrator method {method!r}")
        super().__init__(feature_set="full")
        self.feature_set = method

    def _apply_feature_set(self, features: torch.Tensor) -> torch.Tensor:
        if self.feature_set == "static_zero":
            return torch.zeros_like(features)
        if self.feature_set == "blinear_only":
            keep_names = {"B_linear"}
        else:
            keep_names = {"B_linear", "x_obs", "m", "local_volatility"}
        keep = {self._feature_index[name] for name in keep_names}
        out = torch.zeros_like(features)
        for index in keep:
            out[..., index] = features[..., index]
        return out


def new_calibrator(method: str, *, model_seed: int) -> ACILBase:
    """Construct one registered arm with a bit-identical initial state."""

    if method not in _METHODS:
        raise ValueError(
            "method must be static_zero, blinear_only, or value_only"
        )
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise TypeError("model_seed must be an integer")
    if model_seed < 0:
        raise ValueError("model_seed must be nonnegative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = ACILBase(feature_set="full")
    model.extractor = CalibrationFeatureExtractor(method)
    return model


__all__ = ["CalibrationFeatureExtractor", "new_calibrator"]

