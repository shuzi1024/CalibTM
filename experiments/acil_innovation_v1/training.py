"""Frozen training primitives shared by all ACIL-Innovation stages."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .acil import ACILBase, middle_anchor_leave_one_out
from .config import load_protocol_config
from .preprocessing import FitFallback
from .registries import seed_bundle


_ORDER_DOMAIN = b"acil-innovation-v1:epoch-order:v1\x00"


def epoch_window_order(*, seed_bundle: int, epoch: int) -> np.ndarray:
    bundle = globals()["seed_bundle"](seed_bundle)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < 20:
        raise ValueError("epoch must be a zero-based integer in [0,20)")
    payload = f"{bundle.data_order}:{epoch}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(_ORDER_DOMAIN + payload).digest()[:16], "big")
    order = np.random.Generator(np.random.PCG64DXSM(seed)).permutation(512)
    result = np.ascontiguousarray(order, dtype="<i8")
    result.setflags(write=False)
    return result


def normalized_target_mae(
    prediction: Tensor, truth: Tensor, target: Tensor, scale: Tensor
) -> Tensor:
    if prediction.shape != truth.shape or prediction.shape != target.shape:
        raise ValueError("prediction, truth, and target shapes must match")
    if target.dtype is not torch.bool:
        raise TypeError("target must be boolean")
    if scale.shape != prediction.shape[:-1] + (1,):
        raise ValueError("scale must have shape [B,F,1]")
    if not target.any().item() or (scale <= 0).any().item():
        raise ValueError("target must be nonempty and scale positive")
    error = torch.abs((prediction - truth) / scale)
    selected = error.masked_select(target)
    if not torch.isfinite(selected).all().item():
        raise FloatingPointError("normalized target loss is nonfinite")
    return selected.mean()


@dataclass(frozen=True, slots=True)
class ACILObjectiveDetails:
    k3_loss: Tensor
    k2_loss: Tensor
    k3_target_count: int
    k2_target_count: int


def acil_paired_objective(
    model: ACILBase,
    truth: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> tuple[Tensor, ACILObjectiveDetails]:
    return acil_paired_objective_separated(
        model,
        model_input=truth,
        truth=truth,
        observed=observed,
        fit_fallback=fit_fallback,
    )


def acil_paired_objective_separated(
    model: ACILBase,
    *,
    model_input: Tensor,
    truth: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> tuple[Tensor, ACILObjectiveDetails]:
    """Pair K=3/K=2 losses while keeping missing truth outside model input."""

    if not isinstance(model, ACILBase):
        raise TypeError("model must be ACILBase")
    if model_input.shape != truth.shape or truth.shape != observed.shape:
        raise ValueError("model input, truth, and observed shapes must match")
    if not torch.isfinite(truth).all().item() or (truth < 0).any().item():
        raise ValueError("loss truth must be finite and nonnegative")
    loo = middle_anchor_leave_one_out(model, model_input, observed, fit_fallback)
    k3_target = ~observed
    k2_target = ~loo.submask
    k3_loss = normalized_target_mae(
        loo.full_result.prediction,
        truth,
        k3_target,
        loo.full_result.statistics.std,
    )
    k2_loss = normalized_target_mae(
        loo.submask_result.prediction,
        truth,
        k2_target,
        loo.submask_result.statistics.std,
    )
    details = ACILObjectiveDetails(
        k3_loss=k3_loss,
        k2_loss=k2_loss,
        k3_target_count=int(k3_target.sum().item()),
        k2_target_count=int(k2_target.sum().item()),
    )
    return 0.5 * (k3_loss + k2_loss), details


def choose_checkpoint_epoch(rows: Sequence[Mapping[str, object]]) -> int:
    if not rows:
        raise ValueError("checkpoint table is empty")
    grouped: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        family = str(row["mask_family"])
        ae = float(row["ae"])
        truth = float(row["truth"])
        if family not in {"random", "internal_block", "two_burst"}:
            raise ValueError("checkpoint table has an unknown mask family")
        if not math.isfinite(ae) or not math.isfinite(truth) or ae < 0 or truth <= 0:
            raise ValueError("checkpoint sums must be finite with positive truth")
        grouped.setdefault(epoch, {}).setdefault(family, []).append((ae, truth))
    candidates = []
    required = {"random", "internal_block", "two_burst"}
    for epoch, masks in grouped.items():
        if set(masks) != required:
            raise ValueError("each checkpoint epoch must contain all three masks")
        ae_sum = math.fsum(ae for rows_for_mask in masks.values() for ae, _ in rows_for_mask)
        truth_sum = math.fsum(truth for rows_for_mask in masks.values() for _, truth in rows_for_mask)
        candidates.append((ae_sum / truth_sum, epoch))
    return min(candidates)[1]


_RESIDUAL_KINDS = {
    "local_loo": "local",
    "global_loo": "deepsets",
    "loo_deepsets": "deepsets",
    "full_u0": "full_u0",
    "full_scratch": "gpt2_scratch",
    "full_gpt2": "gpt2_set",
}


def residual_model_kind(method: str) -> str:
    try:
        return _RESIDUAL_KINDS[method]
    except KeyError:
        raise ValueError(f"method {method!r} is not a residual model") from None


def trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    values = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not values:
        raise ValueError("model has no trainable parameters")
    return values


@dataclass(frozen=True, slots=True)
class SourceDevResult:
    """Raw source-dev rows pooled before checkpoint comparison."""

    rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("source-dev rows cannot be empty")
        families = set()
        for row in rows:
            family = str(row["mask_family"])
            ae = float(row["ae"])
            truth = float(row["truth"])
            if family not in {"random", "internal_block", "two_burst"}:
                raise ValueError("source-dev row has an unknown mask family")
            if not math.isfinite(ae) or not math.isfinite(truth) or ae < 0 or truth <= 0:
                raise ValueError("source-dev sums must be finite with positive truth")
            families.add(family)
        if families != {"random", "internal_block", "two_burst"}:
            raise ValueError("source-dev rows must contain all three mask families")
        object.__setattr__(self, "rows", rows)

    @property
    def absolute_error_sum(self) -> float:
        return math.fsum(float(row["ae"]) for row in self.rows)

    @property
    def absolute_truth_sum(self) -> float:
        return math.fsum(float(row["truth"]) for row in self.rows)

    @property
    def nmae(self) -> float:
        return self.absolute_error_sum / self.absolute_truth_sum


@dataclass(frozen=True, slots=True)
class ProtocolEpochRecord:
    epoch: int
    physical_batches: int
    optimizer_updates: int
    mean_loss: float
    source_dev_absolute_error_sum: float
    source_dev_absolute_truth_sum: float
    source_dev_nmae: float
    selected_as_best: bool


@dataclass(frozen=True, slots=True)
class ProtocolTrainingResult:
    epochs_completed: int
    optimizer_updates: int
    best_epoch: int
    best_source_dev_nmae: float
    best_state: Mapping[str, Tensor]
    epochs: tuple[ProtocolEpochRecord, ...]


def _optimizer_specification() -> Mapping[str, object]:
    specification = load_protocol_config()["training"]["optimizer"]
    required = {
        "type": "AdamW",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "lr_schedule": "constant",
        "physical_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 32,
        "optimizer_updates_per_epoch": 16,
        "total_optimizer_updates": 320,
        "learning_rates": {
            "new_modules": 0.0003,
            "scratch_gpt_blocks": 0.0003,
            "pretrained_gpt_blocks": 0.00001,
        },
        "gradient_clip_norm": 1.0,
        "weight_decay": 0.01,
        "no_decay_rules": [
            "parameter_name_endswith_bias",
            "parameter_belongs_to_layernorm_and_name_endswith_weight",
        ],
        "weight_decay_applies_to": "all_other_trainable_parameters",
    }
    for name, expected in required.items():
        if specification.get(name) != expected:
            raise RuntimeError(f"optimizer protocol field {name!r} drifted")
    return specification


def _normalization_parameter_ids(model: nn.Module) -> set[int]:
    result: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            result.update(id(parameter) for parameter in module.parameters(recurse=False))
    return result


def build_protocol_optimizer(
    model: nn.Module,
    *,
    pretrained_parameters: Iterable[nn.Parameter],
) -> tuple[torch.optim.AdamW, Mapping[str, tuple[str, ...]]]:
    """Build the frozen provenance/LR and decay partition exactly once."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    specification = _optimizer_specification()
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named:
        raise ValueError("model has no trainable parameters")
    model_ids = {id(parameter) for _, parameter in named}
    supplied = tuple(pretrained_parameters)
    supplied_ids = [id(parameter) for parameter in supplied]
    if len(supplied_ids) != len(set(supplied_ids)):
        raise ValueError("pretrained parameter list contains duplicates")
    if any(
        not isinstance(parameter, nn.Parameter) or id(parameter) not in model_ids
        for parameter in supplied
    ):
        raise ValueError("pretrained parameters must be trainable members of the model")
    pretrained_ids = set(supplied_ids)
    normalization_ids = _normalization_parameter_ids(model)
    order = (
        "pretrained_decay",
        "pretrained_no_decay",
        "new_decay",
        "new_no_decay",
    )
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {name: [] for name in order}
    for name, parameter in named:
        provenance = "pretrained" if id(parameter) in pretrained_ids else "new"
        no_decay = name.rsplit(".", 1)[-1] == "bias" or id(parameter) in normalization_ids
        grouped[f"{provenance}_{'no_decay' if no_decay else 'decay'}"].append(
            (name, parameter)
        )
    parameter_groups = []
    for group_name in order:
        values = grouped[group_name]
        if not values:
            continue
        pretrained = group_name.startswith("pretrained_")
        no_decay = group_name.endswith("_no_decay")
        parameter_groups.append(
            {
                "params": [parameter for _, parameter in values],
                "group_name": group_name,
                "lr": float(
                    specification["learning_rates"]["pretrained_gpt_blocks"]
                    if pretrained
                    else specification["learning_rates"]["new_modules"]
                ),
                "weight_decay": 0.0
                if no_decay
                else float(specification["weight_decay"]),
            }
        )
    flattened = [parameter for group in parameter_groups for parameter in group["params"]]
    if len(flattened) != len(named) or {id(item) for item in flattened} != model_ids:
        raise RuntimeError("optimizer partition must contain every trainable parameter once")
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=tuple(float(value) for value in specification["betas"]),
        eps=float(specification["epsilon"]),
    )
    audit = {
        name: tuple(parameter_name for parameter_name, _ in grouped[name])
        for name in order
    }
    return optimizer, audit


def _owned_cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def run_protocol_training(
    model: nn.Module,
    *,
    pretrained_parameters: Iterable[nn.Parameter],
    epoch_batches: Callable[[int], Iterable[object]],
    compute_loss: Callable[[nn.Module, object], Tensor],
    evaluate_source_dev: Callable[[nn.Module, int], SourceDevResult],
    device_type: str,
) -> ProtocolTrainingResult:
    """Run the exact 20-epoch, 320-update lane with outcome-free stopping."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("device_type must be cpu or cuda")
    if not all(callable(value) for value in (epoch_batches, compute_loss, evaluate_source_dev)):
        raise TypeError("training callbacks must be callable")
    config = load_protocol_config()["training"]
    if (
        config.get("epochs") != 20
        or config.get("fit_window_count") != 512
        or config.get("outcome_dependent_early_stop") is not False
        or config.get("precision") != "bf16"
    ):
        raise RuntimeError("fixed training lane drifted")
    optimizer, _ = build_protocol_optimizer(
        model, pretrained_parameters=pretrained_parameters
    )
    specification = _optimizer_specification()
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if {parameter.device.type for parameter in trainable} != {device_type}:
        raise ValueError("trainable parameters do not share the declared device type")

    records: list[ProtocolEpochRecord] = []
    updates = 0
    best_epoch = -1
    best_metric = math.inf
    best_state: dict[str, Tensor] | None = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(20):
        model.train()
        losses: list[float] = []
        batch_count = 0
        for batch_count, batch in enumerate(epoch_batches(epoch), start=1):
            if batch_count > 64:
                raise ValueError("epoch produced more than 64 physical batches")
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == "cuda",
            ):
                loss = compute_loss(model, batch)
            if not isinstance(loss, Tensor) or loss.ndim != 0:
                raise TypeError("compute_loss must return one scalar tensor")
            loss_value = float(loss.detach().float().item())
            if not math.isfinite(loss_value):
                raise FloatingPointError("training loss became nonfinite")
            (loss / 4.0).backward()
            losses.append(loss_value)
            if batch_count % 4 == 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, float(specification["gradient_clip_norm"])
                )
                if not math.isfinite(float(torch.as_tensor(gradient_norm).float().item())):
                    raise FloatingPointError("training gradient norm became nonfinite")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        if batch_count != 64:
            raise ValueError(f"epoch produced {batch_count} rather than 64 batches")
        if updates != (epoch + 1) * 16:
            raise RuntimeError("optimizer update count drifted")
        model.eval()
        with torch.no_grad():
            source_dev = evaluate_source_dev(model, epoch)
        if not isinstance(source_dev, SourceDevResult):
            raise TypeError("source-dev callback must return SourceDevResult")
        metric = source_dev.nmae
        selected = metric < best_metric
        if selected:
            best_metric = metric
            best_epoch = epoch
            best_state = _owned_cpu_state(model)
        records.append(
            ProtocolEpochRecord(
                epoch=epoch,
                physical_batches=64,
                optimizer_updates=16,
                mean_loss=math.fsum(losses) / 64.0,
                source_dev_absolute_error_sum=source_dev.absolute_error_sum,
                source_dev_absolute_truth_sum=source_dev.absolute_truth_sum,
                source_dev_nmae=metric,
                selected_as_best=selected,
            )
        )
    if updates != 320 or best_state is None or best_epoch < 0:
        raise RuntimeError("successful training did not produce the exact fixed lane")
    records = [
        replace(row, selected_as_best=row.epoch == best_epoch) for row in records
    ]
    model.load_state_dict(best_state, strict=True)
    return ProtocolTrainingResult(
        epochs_completed=20,
        optimizer_updates=updates,
        best_epoch=best_epoch,
        best_source_dev_nmae=best_metric,
        best_state=best_state,
        epochs=tuple(records),
    )


__all__ = [
    "ACILObjectiveDetails",
    "acil_paired_objective",
    "acil_paired_objective_separated",
    "choose_checkpoint_epoch",
    "build_protocol_optimizer",
    "epoch_window_order",
    "normalized_target_mae",
    "ProtocolEpochRecord",
    "ProtocolTrainingResult",
    "residual_model_kind",
    "run_protocol_training",
    "SourceDevResult",
    "trainable_parameters",
]
