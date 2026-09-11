"""One frozen tiny-KAN transplant for the minimal interpolation calibrator.

This module is deliberately data-free.  It defines only three construction
arms and does not import a dataset loader, checkpoint bank, result file, or
runner:

``mlp_value``
    The exact existing ``value_only`` model.  This is the matched incumbent.
``kan_value``
    The same feature extractor and constrained interpolation equations, with
    the learned MLP-and-head logit map replaced by one frozen tiny KAN.
``kan_static``
    The identical KAN receiving algebraically zeroed features.  This controls
    for input conditioning without changing the KAN parameterization.

The KAN configuration is intentionally singular rather than tunable:
LayerNorm, 16 -> 32 -> 3, five uniform grid intervals, and cubic B-splines.
The three outputs are logits for the existing delta-r, internal-offset, and
edge-offset equations.  All feasibility logic remains inherited verbatim from
``AnchorConditionedInterpolationLayer.forward``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.acil_core import (
    AnchorConditionedInterpolationLayer,
)
from experiments.minimal_calibrator_v1.model import (
    CalibrationFeatureExtractor,
    new_calibrator as new_existing_calibrator,
)


METHODS = ("mlp_value", "kan_value", "kan_static")
KAN_INPUT_DIM = 16
KAN_HIDDEN_DIM = 32
KAN_OUTPUT_DIM = 3
KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3
KAN_GRID_RANGE = (-1.0, 1.0)


class TinyKANLayer(nn.Module):
    """A compact edge-wise B-spline KAN layer with a SiLU base branch.

    Each input-output edge owns ``grid_size + spline_order`` spline
    coefficients and one base coefficient.  The fixed knot grid is a buffer,
    not a learned parameter.  In line with the reference KAN construction, its
    uniform grid is extended by ``spline_order`` knots on each side rather than
    endpoint-repeated as an open-uniform CAD spline.  ``tanh(x / 2)`` maps
    arbitrary normalized features into the registered spline range without
    prematurely saturating ordinary normalized values; the parallel base
    branch still receives the original value through SiLU.

    Basis recursion and both learned contractions explicitly run in FP32 when
    given FP16/BF16 values.  This is intentionally part of the frozen model,
    not a runner option, so CUDA autocast cannot silently change the spline
    arithmetic.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = KAN_GRID_SIZE,
        spline_order: int = KAN_SPLINE_ORDER,
        grid_range: tuple[float, float] = KAN_GRID_RANGE,
    ) -> None:
        super().__init__()
        if isinstance(in_features, bool) or int(in_features) != in_features:
            raise TypeError("in_features must be an integer")
        if isinstance(out_features, bool) or int(out_features) != out_features:
            raise TypeError("out_features must be an integer")
        if int(in_features) < 1 or int(out_features) < 1:
            raise ValueError("KAN feature dimensions must be positive")
        if isinstance(grid_size, bool) or int(grid_size) != grid_size:
            raise TypeError("grid_size must be an integer")
        if isinstance(spline_order, bool) or int(spline_order) != spline_order:
            raise TypeError("spline_order must be an integer")
        if int(grid_size) < 1 or int(spline_order) < 1:
            raise ValueError("KAN grid size and spline order must be positive")
        if len(grid_range) != 2:
            raise ValueError("grid_range must have two endpoints")
        grid_min, grid_max = (float(value) for value in grid_range)
        if not math.isfinite(grid_min) or not math.isfinite(grid_max):
            raise ValueError("KAN grid endpoints must be finite")
        if grid_min >= grid_max:
            raise ValueError("KAN grid range must be strictly increasing")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_basis = self.grid_size + self.spline_order

        step = (grid_max - grid_min) / float(self.grid_size)
        core = torch.linspace(grid_min, grid_max, self.grid_size + 1)
        extension = torch.arange(
            1, self.spline_order + 1, dtype=core.dtype
        )
        left = core[0] - step * extension.flip(0)
        right = core[-1] + step * extension
        self.register_buffer(
            "knots", torch.cat((left, core, right)), persistent=True
        )

        self.base_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, self.num_basis)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(
            self.spline_weight,
            mean=0.0,
            std=0.02 / math.sqrt(float(self.in_features)),
        )
        nn.init.zeros_(self.bias)

    def _validate_inputs(self, inputs: Tensor) -> None:
        if not isinstance(inputs, Tensor):
            raise TypeError("KAN inputs must be a torch tensor")
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"KAN inputs must end in {self.in_features} features"
            )
        if not inputs.is_floating_point():
            raise TypeError("KAN inputs must use a floating dtype")

    @staticmethod
    def _compute_dtype(inputs: Tensor) -> torch.dtype:
        if inputs.dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return inputs.dtype

    def _basis_unchecked(self, inputs: Tensor) -> Tensor:
        """Evaluate basis from already validated, compute-dtype inputs."""

        # The factor 1/2 keeps common normalized values in the central, better
        # resolved part of the fixed grid while retaining a finite global map.
        spline_inputs = torch.tanh(inputs / 2.0).unsqueeze(-1)
        knots = self.knots.to(device=inputs.device, dtype=inputs.dtype)
        basis = (
            (spline_inputs >= knots[:-1])
            & (spline_inputs < knots[1:])
        ).to(inputs.dtype)

        for degree in range(1, self.spline_order + 1):
            count = knots.numel() - 1 - degree
            left_denominator = knots[degree : degree + count] - knots[:count]
            right_denominator = (
                knots[degree + 1 : degree + 1 + count]
                - knots[1 : 1 + count]
            )
            left = (
                (spline_inputs - knots[:count])
                / left_denominator.clamp_min(torch.finfo(inputs.dtype).eps)
            ) * basis[..., :count]
            right = (
                (knots[degree + 1 : degree + 1 + count] - spline_inputs)
                / right_denominator.clamp_min(torch.finfo(inputs.dtype).eps)
            ) * basis[..., 1 : count + 1]
            basis = left + right

        if basis.shape[-2:] != (self.in_features, self.num_basis):
            raise RuntimeError("KAN B-spline basis shape drifted")
        return basis

    def b_spline_basis(self, inputs: Tensor) -> Tensor:
        """Evaluate the registered B-spline basis on the last input axis."""

        self._validate_inputs(inputs)
        compute_dtype = self._compute_dtype(inputs)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return self._basis_unchecked(inputs.to(dtype=compute_dtype))

    def forward(self, inputs: Tensor) -> Tensor:
        self._validate_inputs(inputs)
        compute_dtype = self._compute_dtype(inputs)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            working = inputs.to(dtype=compute_dtype)
            basis = self._basis_unchecked(working)
            base = F.linear(
                F.silu(working),
                self.base_weight.to(dtype=compute_dtype),
                self.bias.to(dtype=compute_dtype),
            )
            spline = torch.einsum(
                "...ik,oik->...o",
                basis,
                self.spline_weight.to(dtype=compute_dtype),
            )
            return base + spline


class TinyKANLogitNetwork(nn.Module):
    """The sole registered 16 -> 32 -> 3 cubic-spline KAN."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(KAN_INPUT_DIM)
        self.first = TinyKANLayer(KAN_INPUT_DIM, KAN_HIDDEN_DIM)
        self.second = TinyKANLayer(KAN_HIDDEN_DIM, KAN_OUTPUT_DIM)
        # The second KAN is the three-logit output head.  Mirror the incumbent
        # MLP's near-zero head initialization so this one-shot comparison
        # changes the function family without also changing the initial
        # "stay near canonical Linear" prior.
        nn.init.normal_(self.second.base_weight, mean=0.0, std=1.0e-4)
        nn.init.normal_(self.second.spline_weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.second.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != KAN_INPUT_DIM:
            raise ValueError(
                f"tiny KAN expects {KAN_INPUT_DIM} feature channels"
            )
        compute_dtype = TinyKANLayer._compute_dtype(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            normalized = F.layer_norm(
                features.to(dtype=compute_dtype),
                self.input_norm.normalized_shape,
                self.input_norm.weight.to(dtype=compute_dtype),
                self.input_norm.bias.to(dtype=compute_dtype),
                self.input_norm.eps,
            )
        return self.second(self.first(normalized))


class _SelectLogit(nn.Module):
    """Expose one KAN logit through the legacy parameter-free head interface."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = int(index)

    def forward(self, logits: Tensor) -> Tensor:
        if logits.shape[-1] != KAN_OUTPUT_DIM:
            raise ValueError("tiny KAN logit width drifted")
        return logits[..., self.index : self.index + 1]


class KANAnchorConditionedInterpolationLayer(
    AnchorConditionedInterpolationLayer
):
    """KAN logits with the exact inherited bounded interpolation equations."""

    def __init__(
        self,
        *,
        feature_dim: int = KAN_INPUT_DIM,
        beta_r: float = 0.25,
        beta_o: float = 0.10,
        beta_e: float = 0.10,
        use_edge_extrapolation: bool = True,
        eps: float = 1e-6,
    ) -> None:
        # Initialize nn.Module directly so the discarded legacy MLP never
        # affects the model identity or parameter count.  Forward is inherited
        # unchanged from AnchorConditionedInterpolationLayer.
        nn.Module.__init__(self)
        if int(feature_dim) != KAN_INPUT_DIM:
            raise ValueError(f"tiny KAN feature_dim must be {KAN_INPUT_DIM}")
        self.eps = float(eps)
        self.beta_r = float(beta_r)
        self.beta_o = float(beta_o)
        self.beta_e = float(beta_e)
        self.use_edge_extrapolation = bool(use_edge_extrapolation)
        self.trunk = TinyKANLogitNetwork()
        self.delta_r_head = _SelectLogit(0)
        self.offset_head = _SelectLogit(1)
        self.edge_offset_head = _SelectLogit(2)


class TinyKANCalibrator(ACILBase):
    """An ACIL-compatible shell whose only learned logit map is tiny KAN."""

    def __init__(self, *, feature_method: str) -> None:
        if feature_method not in {"value_only", "static_zero"}:
            raise ValueError("unsupported tiny-KAN feature method")
        # ACILBase fixes all preprocessing and raw-space projection semantics.
        # Its interpolator is then wholly replaced before the model is exposed.
        super().__init__(feature_set="full")
        self.extractor = CalibrationFeatureExtractor(feature_method)
        self.interpolator = KANAnchorConditionedInterpolationLayer(
            feature_dim=self.extractor.feature_dim
        )

    def architecture_record(self) -> dict[str, object]:
        """Return the singular, runner-visible architecture identity."""

        return {
            "arithmetic": "fp32-for-fp16-or-bf16",
            "base_branch": "silu",
            "beta_e": self.interpolator.beta_e,
            "beta_o": self.interpolator.beta_o,
            "beta_r": self.interpolator.beta_r,
            "feature_dim": KAN_INPUT_DIM,
            "feature_set": self.extractor.feature_set,
            "grid_range": list(KAN_GRID_RANGE),
            "grid_size": KAN_GRID_SIZE,
            "hidden_dim": KAN_HIDDEN_DIM,
            "knot_vector": "uniform-extended-by-spline-order",
            "layer_widths": [KAN_INPUT_DIM, KAN_HIDDEN_DIM, KAN_OUTPUT_DIM],
            "logit_dim": KAN_OUTPUT_DIM,
            "output_layer_initialization": "base-and-spline-normal-1e-4-bias-zero",
            "spline_basis_count_per_edge": KAN_GRID_SIZE + KAN_SPLINE_ORDER,
            "spline_input_map": "tanh(x/2)",
            "spline_order": KAN_SPLINE_ORDER,
            "trunk": "layernorm-kan-kan",
        }


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Return an exact scalar parameter count without touching any data."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def new_calibrator(method: str, *, model_seed: int) -> ACILBase:
    """Construct one frozen KAN arm or the exact existing MLP control."""

    if method not in METHODS:
        raise ValueError(
            "method must be mlp_value, kan_value, or kan_static"
        )
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise TypeError("model_seed must be an integer")
    if model_seed < 0:
        raise ValueError("model_seed must be nonnegative")
    if method == "mlp_value":
        # Do not reconstruct this arm locally: it must remain bit-identical to
        # the frozen incumbent implementation.
        return new_existing_calibrator("value_only", model_seed=model_seed)

    feature_method = "value_only" if method == "kan_value" else "static_zero"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = TinyKANCalibrator(feature_method=feature_method)
    return model


__all__ = [
    "KAN_GRID_RANGE",
    "KAN_GRID_SIZE",
    "KAN_HIDDEN_DIM",
    "KAN_INPUT_DIM",
    "KAN_OUTPUT_DIM",
    "KAN_SPLINE_ORDER",
    "METHODS",
    "KANAnchorConditionedInterpolationLayer",
    "TinyKANCalibrator",
    "TinyKANLayer",
    "TinyKANLogitNetwork",
    "count_parameters",
    "new_calibrator",
]
