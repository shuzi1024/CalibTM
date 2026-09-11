"""CPU-only assembly of registered windows and scalar mask identities.

This module deliberately owns no optimizer, device, path, split, order, or mask
override.  Hidden payloads are replaced by NaN before they reach a model while
the independent truth tensor remains available to losses and metric code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from .config import load_protocol_config
from .data import RegisteredWindows
from .masks import (
    MaskIdentity,
    build_evaluation_mask,
    build_mixed_training_mask,
    factory_oracle_partition,
    mixed_training_family,
)
from .oracle_model import OracleSupportQ
from .registries import dataset_spec, ordered_window_starts
from .training import epoch_window_order


IdentityGrid = tuple[tuple[MaskIdentity, ...], ...]
HashGrid = tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    dataset: str
    cohort: str
    seed_bundle: int
    epoch: int
    batch_index: int
    epoch_order_positions: tuple[int, ...]
    window_indices: tuple[int, ...]
    absolute_starts: tuple[int, ...]
    families: tuple[str, ...]
    mask_identities: IdentityGrid
    mask_sha256: HashGrid
    oracle_q_sha256: HashGrid
    oracle_e_sha256: HashGrid
    truth: Tensor
    model_input: Tensor
    observed: Tensor
    target: Tensor
    controlled_gap: Tensor
    oracle_q: Tensor
    oracle_e: Tensor
    oracle_support_q: OracleSupportQ


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    dataset: str
    cohort: str
    seed_bundle: int
    family: str
    window_indices: tuple[int, ...]
    absolute_starts: tuple[int, ...]
    mask_identities: IdentityGrid
    mask_sha256: HashGrid
    oracle_q_sha256: HashGrid
    oracle_e_sha256: HashGrid
    truth: Tensor
    model_input: Tensor
    observed: Tensor
    target: Tensor
    controlled_gap: Tensor
    oracle_q: Tensor
    oracle_e: Tensor
    oracle_support_q: OracleSupportQ


def _require_exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _require_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


@lru_cache(maxsize=1)
def _training_layout() -> tuple[int, int, int]:
    config = load_protocol_config()["training"]
    fit_count = _require_exact_int(config["fit_window_count"], "fit window count")
    epochs = _require_exact_int(config["epochs"], "epoch count")
    physical_batch = _require_exact_int(
        config["optimizer"]["physical_batch_size"], "physical batch size"
    )
    if fit_count <= 0 or epochs <= 0 or physical_batch <= 0:
        raise RuntimeError("training layout values must be positive")
    if fit_count % physical_batch:
        raise RuntimeError("fit windows must divide exactly into physical batches")
    return fit_count, physical_batch, epochs


@lru_cache(maxsize=120)
def _cached_epoch_window_order(seed_bundle: int, epoch: int) -> tuple[int, ...]:
    order = epoch_window_order(seed_bundle=seed_bundle, epoch=epoch)
    fit_count, _, _ = _training_layout()
    result = tuple(int(index) for index in order)
    if len(result) != fit_count or set(result) != set(range(fit_count)):
        raise RuntimeError("epoch window order is not the registered permutation")
    return result


def _validated_windows(
    windows: RegisteredWindows,
) -> tuple[str, str, tuple[int, ...], np.ndarray]:
    if not isinstance(windows, RegisteredWindows):
        raise TypeError("windows must be RegisteredWindows")
    dataset = _require_name(windows.dataset, "dataset")
    cohort = _require_name(windows.cohort, "cohort")
    registered_starts = ordered_window_starts(dataset, cohort)
    if not isinstance(windows.absolute_starts, tuple):
        raise TypeError("registered absolute starts must be a tuple")
    if windows.absolute_starts != registered_starts:
        raise ValueError("window schedule differs from the registered schedule")
    values = windows.values
    if not isinstance(values, np.ndarray):
        raise TypeError("registered values must be a NumPy array")
    expected_shape = (len(registered_starts), dataset_spec(dataset).flows, 50)
    if values.shape != expected_shape or values.dtype != np.dtype("<f4"):
        raise ValueError("registered window tensor shape or dtype drifted")
    if values.flags.writeable or not values.flags.c_contiguous:
        raise ValueError("registered window tensor must be read-only C-order")
    return dataset, cohort, registered_starts, values


def _owned_truth(values: np.ndarray, indices: tuple[int, ...]) -> Tensor:
    selected = np.array(values[list(indices)], dtype="<f4", order="C", copy=True)
    if not np.isfinite(selected).all() or np.any(selected < 0.0):
        raise ValueError("selected registered truth must be finite and nonnegative")
    return torch.from_numpy(selected)


def _mask_tensors(
    truth: Tensor,
    observed: np.ndarray,
    target: np.ndarray,
    controlled_gap: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    observed_tensor = torch.from_numpy(np.ascontiguousarray(observed))
    target_tensor = torch.from_numpy(np.ascontiguousarray(target))
    controlled_tensor = torch.from_numpy(np.ascontiguousarray(controlled_gap))
    if not torch.equal(target_tensor, ~observed_tensor):
        raise RuntimeError("batched target is not the observed complement")
    model_input = truth.clone()
    model_input.masked_fill_(~observed_tensor, float("nan"))
    if model_input.data_ptr() == truth.data_ptr():
        raise RuntimeError("truth and model input unexpectedly share storage")
    if not torch.isnan(model_input.masked_select(~observed_tensor)).all().item():
        raise RuntimeError("missing model payload was not replaced by NaN")
    if not torch.equal(
        model_input.masked_select(observed_tensor),
        truth.masked_select(observed_tensor),
    ):
        raise RuntimeError("observed model payload differs from registered truth")
    return model_input, observed_tensor, target_tensor, controlled_tensor


def _oracle_tensors(
    truth: Tensor,
    observed: Tensor,
    target: Tensor,
    oracle_q: np.ndarray,
    oracle_e: np.ndarray,
    q_indices: np.ndarray,
) -> tuple[Tensor, Tensor, OracleSupportQ]:
    oracle_q_tensor = torch.from_numpy(np.ascontiguousarray(oracle_q))
    oracle_e_tensor = torch.from_numpy(np.ascontiguousarray(oracle_e))
    if torch.any(oracle_q_tensor & oracle_e_tensor).item():
        raise RuntimeError("oracle Q and E overlap in the batched registry")
    if not torch.equal(oracle_q_tensor | oracle_e_tensor, target):
        raise RuntimeError("oracle Q/E union differs from the deployable target")
    q_index_tensor = torch.from_numpy(np.ascontiguousarray(q_indices, dtype="<i8"))
    support_q = OracleSupportQ(
        indices=q_index_tensor,
        values=truth.gather(-1, q_index_tensor.unsqueeze(-1)).clone(),
    )
    if not torch.equal(support_q.support_mask(observed), oracle_q_tensor):
        raise RuntimeError("compact oracle support differs from the Q registry")
    return oracle_q_tensor, oracle_e_tensor, support_q


def build_training_batch(
    windows: RegisteredWindows,
    *,
    seed_bundle: int,
    epoch: int,
    batch_index: int,
) -> TrainingBatch:
    """Assemble one frozen physical fit batch in epoch-order position space."""

    seed_bundle = _require_exact_int(seed_bundle, "seed bundle")
    epoch = _require_exact_int(epoch, "epoch")
    batch_index = _require_exact_int(batch_index, "batch index")
    dataset, cohort, registered_starts, values = _validated_windows(windows)
    if cohort != "fit":
        raise ValueError("training batches require the fit cohort")

    fit_count, physical_batch, epochs = _training_layout()
    if len(registered_starts) != fit_count:
        raise RuntimeError("fit window count differs from the frozen training layout")
    if not 0 <= epoch < epochs:
        raise ValueError("epoch is outside the frozen training range")
    batch_count = fit_count // physical_batch
    if not 0 <= batch_index < batch_count:
        raise ValueError("batch index is outside the frozen epoch layout")

    order = _cached_epoch_window_order(seed_bundle, epoch)
    first_position = batch_index * physical_batch
    positions = tuple(range(first_position, first_position + physical_batch))
    window_indices = tuple(order[position] for position in positions)
    absolute_starts = tuple(registered_starts[index] for index in window_indices)
    truth = _owned_truth(values, window_indices)
    batch_size, flows, times = truth.shape
    observed = np.empty((batch_size, flows, times), dtype=np.bool_)
    target = np.empty_like(observed)
    controlled_gap = np.empty_like(observed)
    oracle_q = np.empty_like(observed)
    oracle_e = np.empty_like(observed)
    q_indices = np.empty((batch_size, flows), dtype="<i8")
    identity_rows: list[tuple[MaskIdentity, ...]] = []
    hash_rows: list[tuple[str, ...]] = []
    q_hash_rows: list[tuple[str, ...]] = []
    e_hash_rows: list[tuple[str, ...]] = []
    families: list[str] = []

    for row, (position, absolute_start) in enumerate(
        zip(positions, absolute_starts)
    ):
        expected_family = mixed_training_family(epoch, position, seed_bundle)
        identities: list[MaskIdentity] = []
        hashes: list[str] = []
        q_hashes: list[str] = []
        e_hashes: list[str] = []
        for flow in range(flows):
            bundle = build_mixed_training_mask(
                dataset=dataset,
                window_start=absolute_start,
                flow_index=flow,
                seed_bundle=seed_bundle,
                epoch=epoch,
                epoch_order_position=position,
            )
            if bundle.identity.family != expected_family:
                raise RuntimeError("mixed mask family differs within one window visit")
            partition = factory_oracle_partition(bundle)
            observed[row, flow] = bundle.observed
            target[row, flow] = bundle.target
            controlled_gap[row, flow] = bundle.controlled_gap
            oracle_q[row, flow] = partition.support
            oracle_e[row, flow] = partition.evaluation
            q_indices[row, flow] = int(np.argmax(partition.support))
            identities.append(bundle.identity)
            hashes.append(partition.mask_sha256)
            q_hashes.append(partition.support_sha256)
            e_hashes.append(partition.evaluation_sha256)
        families.append(expected_family)
        identity_rows.append(tuple(identities))
        hash_rows.append(tuple(hashes))
        q_hash_rows.append(tuple(q_hashes))
        e_hash_rows.append(tuple(e_hashes))

    model_input, observed_tensor, target_tensor, controlled_tensor = _mask_tensors(
        truth, observed, target, controlled_gap
    )
    oracle_q_tensor, oracle_e_tensor, support_q = _oracle_tensors(
        truth, observed_tensor, target_tensor, oracle_q, oracle_e, q_indices
    )
    return TrainingBatch(
        dataset=dataset,
        cohort=cohort,
        seed_bundle=seed_bundle,
        epoch=epoch,
        batch_index=batch_index,
        epoch_order_positions=positions,
        window_indices=window_indices,
        absolute_starts=absolute_starts,
        families=tuple(families),
        mask_identities=tuple(identity_rows),
        mask_sha256=tuple(hash_rows),
        oracle_q_sha256=tuple(q_hash_rows),
        oracle_e_sha256=tuple(e_hash_rows),
        truth=truth,
        model_input=model_input,
        observed=observed_tensor,
        target=target_tensor,
        controlled_gap=controlled_tensor,
        oracle_q=oracle_q_tensor,
        oracle_e=oracle_e_tensor,
        oracle_support_q=support_q,
    )


def build_evaluation_batch(
    windows: RegisteredWindows,
    *,
    seed_bundle: int,
    family: str,
) -> EvaluationBatch:
    """Assemble an entire registered evaluation cohort with fixed Q/E."""

    seed_bundle = _require_exact_int(seed_bundle, "seed bundle")
    family = _require_name(family, "mask family")
    dataset, cohort, absolute_starts, values = _validated_windows(windows)
    if cohort not in {"source_dev", "tune", "gate"}:
        raise ValueError("evaluation batches require source_dev, tune, or gate")

    window_indices = tuple(range(len(absolute_starts)))
    truth = _owned_truth(values, window_indices)
    batch_size, flows, times = truth.shape
    observed = np.empty((batch_size, flows, times), dtype=np.bool_)
    target = np.empty_like(observed)
    controlled_gap = np.empty_like(observed)
    oracle_q = np.empty_like(observed)
    oracle_e = np.empty_like(observed)
    q_indices = np.empty((batch_size, flows), dtype="<i8")
    identity_rows: list[tuple[MaskIdentity, ...]] = []
    mask_hash_rows: list[tuple[str, ...]] = []
    q_hash_rows: list[tuple[str, ...]] = []
    e_hash_rows: list[tuple[str, ...]] = []

    for row, absolute_start in enumerate(absolute_starts):
        identities: list[MaskIdentity] = []
        mask_hashes: list[str] = []
        q_hashes: list[str] = []
        e_hashes: list[str] = []
        for flow in range(flows):
            bundle = build_evaluation_mask(
                dataset=dataset,
                cohort=cohort,
                window_start=absolute_start,
                flow_index=flow,
                family=family,
                seed_bundle=seed_bundle,
            )
            partition = factory_oracle_partition(bundle)
            observed[row, flow] = bundle.observed
            target[row, flow] = bundle.target
            controlled_gap[row, flow] = bundle.controlled_gap
            oracle_q[row, flow] = partition.support
            oracle_e[row, flow] = partition.evaluation
            q_indices[row, flow] = int(np.argmax(partition.support))
            identities.append(bundle.identity)
            mask_hashes.append(partition.mask_sha256)
            q_hashes.append(partition.support_sha256)
            e_hashes.append(partition.evaluation_sha256)
        identity_rows.append(tuple(identities))
        mask_hash_rows.append(tuple(mask_hashes))
        q_hash_rows.append(tuple(q_hashes))
        e_hash_rows.append(tuple(e_hashes))

    model_input, observed_tensor, target_tensor, controlled_tensor = _mask_tensors(
        truth, observed, target, controlled_gap
    )
    oracle_q_tensor, oracle_e_tensor, support_q = _oracle_tensors(
        truth, observed_tensor, target_tensor, oracle_q, oracle_e, q_indices
    )

    return EvaluationBatch(
        dataset=dataset,
        cohort=cohort,
        seed_bundle=seed_bundle,
        family=family,
        window_indices=window_indices,
        absolute_starts=absolute_starts,
        mask_identities=tuple(identity_rows),
        mask_sha256=tuple(mask_hash_rows),
        oracle_q_sha256=tuple(q_hash_rows),
        oracle_e_sha256=tuple(e_hash_rows),
        truth=truth,
        model_input=model_input,
        observed=observed_tensor,
        target=target_tensor,
        controlled_gap=controlled_tensor,
        oracle_q=oracle_q_tensor,
        oracle_e=oracle_e_tensor,
        oracle_support_q=support_q,
    )


__all__ = [
    "EvaluationBatch",
    "TrainingBatch",
    "build_evaluation_batch",
    "build_training_batch",
]
