"""Fit-only preprocessing and reproducible masks for Sync-Delta v1.1.

Canonical parents are memory mapped, but only fit/source-dev/tune rows are
read or hashed. Historical fit truth is fully supervised; the 20% observation
budget applies to the input of each imputation window, not to offline fit data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


PROTOCOL = "sync-delta-v1.1"
WINDOW_LENGTH = 50
CONDITIONS = ("uniform", "unequal", "unequal_gap")
DATA_ORDER_SEED = 51001
TRAINING_MASK_SEED = 61001
SELECTION_MASK_SEED = 71001
POSTFREEZE_MASK_SEEDS = (71002, 71003)
DATASET_SPECS = {
    "abilene": {
        "flows": 144, "fit": (0, 33335), "source_dev": (33384, 33884),
        "tune": (33884, 37484), "train_parent": (0, 33884),
        "val_parent": (33884, 41134),
    },
    "geant": {
        "flows": 462, "fit": (0, 7023), "source_dev": (7072, 7572),
        "tune": (7572, 8322), "train_parent": (0, 7572),
        "val_parent": (7572, 9172),
    },
}
_DEFAULT_CANONICAL = Path(__file__).resolve().parents[1] / "acil_innovation_v1" / "canonical"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def stable_seed(domain: str, **identity: Any) -> int:
    """A domain-separated 128-bit seed, independent of Python hash/process state."""
    digest = hashlib.sha256(
        PROTOCOL.encode("ascii") + b"\0" + domain.encode("ascii") + b"\0" + _json_bytes(identity)
    ).digest()
    return int.from_bytes(digest[:16], "big")


def _rng(domain: str, **identity: Any) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(stable_seed(domain, **identity)))


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    metadata = _json_bytes({"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"})
    digest = hashlib.sha256(PROTOCOL.encode("ascii") + b":array\0" + metadata + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    return hashlib.sha256(_json_bytes({name: array_sha256(value)
                                     for name, value in sorted(arrays.items())})).hexdigest()


def _spec(dataset: str) -> dict[str, Any]:
    if dataset not in DATASET_SPECS:
        raise ValueError("dataset must be 'abilene' or 'geant'")
    return DATASET_SPECS[dataset]


def window_starts(dataset: str, split: str) -> np.ndarray:
    spec = _spec(dataset)
    if split in ("fit", "train"):
        start, stop = spec["fit"]
        return start + np.arange(512, dtype=np.int64) * (stop - start - WINDOW_LENGTH) // 511
    if split == "dev":
        return np.concatenate([np.arange(*spec[name], WINDOW_LENGTH, dtype=np.int64)
                               for name in ("source_dev", "tune")])
    if split in ("source_dev", "tune"):
        return np.arange(*spec[split], WINDOW_LENGTH, dtype=np.int64)
    raise ValueError("only fit/train, source_dev, tune, and dev windows are available")


def _validate_traffic(values: np.ndarray, label: str) -> None:
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{label} must be a finite, nonnegative 2D traffic matrix")


def fit_statistics(fit: np.ndarray) -> dict[str, Any]:
    """Population (ddof=0) fit statistics, computed in float64 and used in FP32."""
    values = np.asarray(fit, dtype=np.float64)
    _validate_traffic(values, "fit")
    if min(values.shape) == 0:
        raise ValueError("fit cannot be empty")
    mu = values.mean(axis=0, dtype=np.float64)
    sigma = values.std(axis=0, ddof=0, dtype=np.float64)
    global_scale = float(np.sqrt(np.mean(sigma * sigma, dtype=np.float64)))
    scale = np.maximum(sigma, max(1e-3 * global_scale, 1e-8))
    c = float(np.mean(np.abs(values), dtype=np.float64))
    if not np.isfinite(c) or c <= 0:
        raise ValueError("fit loss constant C=mean(abs(fit)) must be finite and positive")
    result = {
        "mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32),
        "scale": scale.astype(np.float32), "C": c,
        "global_scale": global_scale, "ddof": 0,
    }
    if not all(np.isfinite(result[name]).all() for name in ("mu", "sigma", "scale")):
        raise ValueError("fit statistics are not representable as finite float32")
    return result


def fit_neighbors(fit: np.ndarray, max_neighbors: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Self plus strictly positive Pearson neighbors, ties broken by flow ID."""
    values = np.asarray(fit, dtype=np.float64)
    _validate_traffic(values, "fit")
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be nonnegative")
    centered = values - values.mean(axis=0, dtype=np.float64)
    ss = np.einsum("tf,tf->f", centered, centered)
    denominator = np.sqrt(ss[:, None] * ss[None, :])
    corr = np.zeros_like(denominator)
    np.divide(centered.T @ centered, denominator, out=corr, where=denominator > 0)
    np.clip(corr, -1.0, 1.0, out=corr)
    flows = values.shape[1]
    neighbors = np.full((flows, max_neighbors + 1), -1, dtype=np.int64)
    ids = np.arange(flows, dtype=np.int64)
    for target in range(flows):
        neighbors[target, 0] = target
        if ss[target] <= 0:
            continue
        eligible = ids[(ss > 0) & (corr[target] > 0) & (ids != target)]
        order = np.lexsort((eligible, -corr[target, eligible]))
        selected = eligible[order[:max_neighbors]]
        neighbors[target, 1:1 + len(selected)] = selected
    return neighbors, corr


def _load_cohort(dataset: str, cohort: str, canonical_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    spec = _spec(dataset)
    if cohort not in ("fit", "source_dev", "tune"):
        raise ValueError("unregistered cohort")
    split = "val" if cohort == "tune" else "train"
    parent_bounds = spec[f"{split}_parent"]
    bounds = spec[cohort]
    path = canonical_dir / f"{dataset}_{split}.npy"
    parent = np.load(path, allow_pickle=False, mmap_mode="r")
    expected_shape = (parent_bounds[1] - parent_bounds[0], spec["flows"])
    if parent.shape != expected_shape or parent.dtype != np.dtype("<f4") or not parent.flags.c_contiguous:
        raise ValueError(f"canonical parent header drift: {path}")
    # Deliberately do not materialize/hash the full val parent, which includes gate rows.
    values = np.array(parent[bounds[0] - parent_bounds[0]:bounds[1] - parent_bounds[0]],
                      dtype="<f4", order="C", copy=True)
    _validate_traffic(values, f"{dataset}/{cohort}")
    record = {"cohort": cohort, "bounds": list(bounds), "source": str(path.resolve()),
              "parent_header_shape": list(expected_shape), "shape": list(values.shape),
              "dtype": values.dtype.str, "array_sha256": array_sha256(values),
              "hash_scope": "cohort_rows_only"}
    return values, record


def _take_windows(values: np.ndarray, absolute_starts: np.ndarray, offset: int) -> np.ndarray:
    indices = absolute_starts[:, None] - offset + np.arange(WINDOW_LENGTH, dtype=np.int64)[None, :]
    if indices.min() < 0 or indices.max() >= values.shape[0]:
        raise ValueError("window escapes its registered cohort")
    return np.ascontiguousarray(values[indices], dtype=np.float32)


@dataclass(frozen=True)
class DataBundle:
    dataset: str
    train_windows: np.ndarray
    dev_windows: np.ndarray
    train_starts: np.ndarray
    dev_starts: np.ndarray
    dev_cohorts: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    scale: np.ndarray
    C: float
    neighbors: np.ndarray
    neighbor_valid: np.ndarray
    metadata: dict[str, Any]

    @property
    def flows(self) -> int:
        return int(self.mu.size)

    @property
    def stats(self) -> dict[str, np.ndarray]:
        return {"mu": self.mu, "scale": self.scale}


def load_dataset(dataset: str, canonical_dir: str | Path | None = None) -> DataBundle:
    spec = _spec(dataset)
    canonical_dir = Path(canonical_dir) if canonical_dir is not None else _DEFAULT_CANONICAL
    fit, fit_record = _load_cohort(dataset, "fit", canonical_dir)
    statistics = fit_statistics(fit)
    neighbors, corr = fit_neighbors(fit)
    train_starts = window_starts(dataset, "fit")
    train_windows = _take_windows(fit, train_starts, spec["fit"][0])
    dev_rows, records, labels = [], [fit_record], []
    for cohort in ("source_dev", "tune"):
        values, record = _load_cohort(dataset, cohort, canonical_dir)
        starts = window_starts(dataset, cohort)
        dev_rows.append(_take_windows(values, starts, spec[cohort][0]))
        records.append(record)
        labels.extend([cohort] * len(starts))
    stats_arrays = {name: statistics[name] for name in ("mu", "sigma", "scale")}
    stats_arrays["C"] = np.asarray(statistics["C"], dtype="<f8")
    stats_arrays["global_scale"] = np.asarray(statistics["global_scale"], dtype="<f8")
    metadata = {
        "protocol": PROTOCOL, "dataset": dataset, "window_length": WINDOW_LENGTH,
        "fit_full_truth_for_offline_statistics_and_supervision": True,
        "online_window_observation_fraction": 0.2, "dev_is_selection_not_test": True,
        "cohort_sources": records, "ddof": 0, "global_scale": statistics["global_scale"],
        "stats_sha256": _arrays_sha256(stats_arrays), "neighbors_sha256": array_sha256(neighbors),
        "pearson_sha256": array_sha256(corr), "train_starts_sha256": array_sha256(train_starts),
        "dev_starts_sha256": array_sha256(window_starts(dataset, "dev")),
    }
    metadata["data_sha256"] = hashlib.sha256(_json_bytes({
        "protocol": PROTOCOL, "dataset": dataset,
        "cohorts": [{key: record[key] for key in ("cohort", "bounds", "array_sha256")}
                    for record in records],
        "stats_sha256": metadata["stats_sha256"], "neighbors_sha256": metadata["neighbors_sha256"],
        "train_starts_sha256": metadata["train_starts_sha256"],
        "dev_starts_sha256": metadata["dev_starts_sha256"],
    })).hexdigest()
    return DataBundle(dataset, train_windows, np.concatenate(dev_rows, axis=0), train_starts,
                      window_starts(dataset, "dev"), np.asarray(labels), statistics["mu"],
                      statistics["sigma"], statistics["scale"], statistics["C"],
                      neighbors, neighbors >= 0, metadata)


def load_fit_windows(dataset: str, indices: tuple[int, ...] = (0, 1),
                     canonical_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Fit-only entry point for real-shape preflight; no dev parent is opened."""
    canonical_dir = Path(canonical_dir) if canonical_dir is not None else _DEFAULT_CANONICAL
    fit, _ = _load_cohort(dataset, "fit", canonical_dir)
    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.ndim != 1 or not len(index_array) or np.any((index_array < 0) | (index_array >= 512)):
        raise ValueError("fit window indices must be nonempty and in [0,512)")
    starts = window_starts(dataset, "fit")[index_array]
    windows = _take_windows(fit, starts, DATASET_SPECS[dataset]["fit"][0])
    neighbors, _ = fit_neighbors(fit)
    return windows, starts, fit_statistics(fit), neighbors


@dataclass(frozen=True)
class MaskFamily:
    uniform: np.ndarray
    unequal: np.ndarray
    unequal_gap: np.ndarray
    sparse_group: np.ndarray
    gap_affected: np.ndarray
    gap_start: int
    seed: int
    window_start: int
    epoch: int | None
    dataset: str
    mask_hashes: dict[str, str]
    family_hash: str

    @property
    def masks(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in CONDITIONS}

    def __getitem__(self, condition: str) -> np.ndarray:
        if condition not in CONDITIONS:
            raise KeyError(condition)
        return getattr(self, condition)


def make_mask_family(flows: int, window_start: int, seed: int, *, dataset: str,
                     epoch: int | None = None) -> MaskFamily:
    """Generate all three paired global masks before selecting target chunks."""
    if flows <= 0 or flows % 2 or int(flows) != flows:
        raise ValueError("mask family requires a positive even flow count")
    if epoch is not None and (int(epoch) != epoch or epoch < 0):
        raise ValueError("epoch must be None or a nonnegative integer")
    flows, window_start, seed = int(flows), int(window_start), int(seed)
    identity = {"dataset": str(dataset), "flows": flows, "window_start": window_start,
                "seed": seed, "epoch": None if epoch is None else int(epoch)}
    permutations = _rng("mask-positions", **identity).permuted(
        np.broadcast_to(np.arange(WINDOW_LENGTH, dtype=np.int16), (flows, WINDOW_LENGTH)), axis=1)
    group_order = _rng("sparse-group", **identity).permutation(flows)
    sparse = np.zeros(flows, dtype=bool)
    sparse[group_order[:flows // 2]] = True
    counts = np.where(sparse, 4, 16)
    affected = np.zeros(flows, dtype=bool)
    sparse_count = flows // 4
    if (flows // 2) % 2:
        sparse_count += int(_rng("gap-balance", **identity).integers(0, 2))
    affected_rng = _rng("gap-affected", **identity)
    affected[affected_rng.permutation(np.flatnonzero(sparse))[:sparse_count]] = True
    dense_count = flows // 2 - sparse_count
    affected[affected_rng.permutation(np.flatnonzero(~sparse))[:dense_count]] = True
    gap_start = int(_rng("gap-start", **identity).integers(0, WINDOW_LENGTH - 10 + 1))
    flow_ids = np.broadcast_to(np.arange(flows)[:, None], permutations.shape)
    rank = np.arange(WINDOW_LENGTH)[None, :]
    uniform = np.zeros((WINDOW_LENGTH, flows), dtype=bool)
    uniform[permutations[:, :10], flow_ids[:, :10]] = True
    unequal = np.zeros_like(uniform)
    selected = np.broadcast_to(rank, permutations.shape) < counts[:, None]
    unequal[permutations[selected], flow_ids[selected]] = True
    allowed = (~affected[:, None]) | (permutations < gap_start) | (permutations >= gap_start + 10)
    selected_gap = allowed & (np.cumsum(allowed, axis=1) <= counts[:, None])
    unequal_gap = np.zeros_like(uniform)
    unequal_gap[permutations[selected_gap], flow_ids[selected_gap]] = True
    masks = dict(zip(CONDITIONS, (uniform, unequal, unequal_gap)))
    assert int(sparse.sum()) == flows // 2 and int(affected.sum()) == flows // 2
    assert abs(int((affected & sparse).sum()) - int((affected & ~sparse).sum())) <= 1
    assert np.all(uniform.sum(axis=0) == 10)
    assert np.array_equal(unequal.sum(axis=0), counts)
    assert np.array_equal(unequal_gap.sum(axis=0), counts)
    assert not unequal_gap[gap_start:gap_start + 10, affected].any()
    assert np.array_equal(unequal[:, ~affected], unequal_gap[:, ~affected])
    assert all(int(mask.sum()) == 10 * flows and int((~mask).sum()) == 40 * flows
               for mask in masks.values())
    hashes = {name: array_sha256(np.packbits(mask.reshape(-1), bitorder="little"))
              for name, mask in masks.items()}
    family_hash = hashlib.sha256(_json_bytes({
        **identity, "mask_hashes": hashes, "gap_start": gap_start,
        "sparse_group_sha256": array_sha256(sparse), "gap_affected_sha256": array_sha256(affected),
    })).hexdigest()
    return MaskFamily(uniform, unequal, unequal_gap, sparse, affected, gap_start, seed,
                      window_start, epoch, str(dataset), hashes, family_hash)


@dataclass(frozen=True)
class EpochMasks:
    epoch: int
    order: np.ndarray
    conditions: tuple[str, ...]
    masks: np.ndarray
    families: tuple[MaskFamily, ...]
    data_order_seed: int
    mask_seed: int


def training_epoch(bundle: DataBundle, epoch: int, data_order_seed: int = DATA_ORDER_SEED,
                   mask_seed: int = TRAINING_MASK_SEED) -> EpochMasks:
    if int(epoch) != epoch or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    count = len(bundle.train_starts)
    order = _rng("training-window-order", dataset=bundle.dataset, seed=int(data_order_seed),
                 epoch=int(epoch)).permutation(count).astype(np.int64)
    conditions = tuple(CONDITIONS[(int(epoch) * count + position) % 3] for position in range(count))
    families = tuple(make_mask_family(bundle.flows, int(bundle.train_starts[index]), mask_seed,
                                     dataset=bundle.dataset, epoch=int(epoch)) for index in order)
    masks = np.stack([family[condition] for family, condition in zip(families, conditions)])
    return EpochMasks(int(epoch), order, conditions, masks, families, int(data_order_seed), int(mask_seed))


def linear_fill(values: np.ndarray, observed: np.ndarray, fit_mean: np.ndarray) -> np.ndarray:
    """Piecewise linear interpolation with constant edges, using observations only."""
    values, observed = np.asarray(values), np.asarray(observed)
    if values.shape != observed.shape or values.ndim not in (2, 3) or observed.dtype != np.bool_:
        raise ValueError("values/mask must match as [T,F] or [B,T,F], with a boolean mask")
    fit_mean = np.asarray(fit_mean)
    if fit_mean.shape != (values.shape[-1],) or not np.isfinite(fit_mean).all():
        raise ValueError("fit_mean must be finite with shape [F]")
    if not np.isfinite(values[observed]).all() or np.any(values[observed] < 0) or np.any(fit_mean < 0):
        raise ValueError("observed values and fit means must be finite and nonnegative")
    single = values.ndim == 2
    values_batch = values[None] if single else values
    masks_batch = observed[None] if single else observed
    output = np.empty(values_batch.shape, dtype=np.float32)
    times = np.arange(values.shape[-2])
    for batch in range(len(values_batch)):
        for flow in range(values.shape[-1]):
            positions = np.flatnonzero(masks_batch[batch, :, flow])
            output[batch, :, flow] = (np.interp(times, positions, values_batch[batch, positions, flow])
                                      if len(positions) else fit_mean[flow])
    # np.interp gives a constant for a single observation and constant edge extrapolation.
    output[masks_batch] = values_batch[masks_batch]
    return output[0] if single else output


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def save_data_manifest(outdir: str | Path, bundle: DataBundle) -> dict[str, Any]:
    outdir = Path(outdir)
    arrays = {"mu": bundle.mu, "sigma": bundle.sigma, "scale": bundle.scale,
              "C": np.asarray(bundle.C, dtype="<f8"),
              "global_scale": np.asarray(bundle.metadata["global_scale"], dtype="<f8"),
              "neighbors": bundle.neighbors, "neighbor_valid": bundle.neighbor_valid,
              "train_starts": bundle.train_starts, "dev_starts": bundle.dev_starts,
              "dev_cohorts": bundle.dev_cohorts}
    path = outdir / "data_arrays.npz"
    _atomic_npz(path, arrays)
    payload = {**bundle.metadata, "arrays_path": path.name,
               "arrays_sha256": _arrays_sha256(arrays),
               "npz_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "train_window_count": len(bundle.train_starts), "dev_window_count": len(bundle.dev_starts),
               "flows": bundle.flows, "selection_mask_seed": SELECTION_MASK_SEED,
               "postfreeze_mask_seeds": list(POSTFREEZE_MASK_SEEDS)}
    _atomic_json(outdir / "data_manifest.json", payload)
    return payload


def save_mask_registry(outdir: str | Path, bundle: DataBundle, seed: int = SELECTION_MASK_SEED,
                       split: str = "dev", epoch: int | None = None,
                       epoch_masks: EpochMasks | None = None) -> dict[str, Any]:
    """Save actual packed masks/groups, not just the RNG seed recipe."""
    if split == "dev":
        if epoch is not None or epoch_masks is not None:
            raise ValueError("dev masks have no epoch")
        starts = bundle.dev_starts
        families = tuple(make_mask_family(bundle.flows, int(start), seed, dataset=bundle.dataset)
                         for start in starts)
        order = np.arange(len(starts), dtype=np.int64)
        selected_conditions = np.full(len(starts), "all")
    elif split == "train":
        if epoch is None:
            raise ValueError("training registry requires epoch")
        schedule = epoch_masks if epoch_masks is not None else training_epoch(bundle, epoch, mask_seed=seed)
        if schedule.epoch != epoch or schedule.mask_seed != seed:
            raise ValueError("supplied epoch schedule does not match registry identity")
        order, families = schedule.order, schedule.families
        starts = bundle.train_starts[order]
        selected_conditions = np.asarray(schedule.conditions)
    else:
        raise ValueError("mask registry split must be dev or train")
    packed = np.stack([np.stack([np.packbits(family[name].reshape(-1), bitorder="little")
                                 for name in CONDITIONS]) for family in families])
    arrays = {
        "window_starts": starts, "window_order": order, "selected_conditions": selected_conditions,
        "condition_names": np.asarray(CONDITIONS), "packed_masks": packed,
        "packed_sparse_group": np.stack([np.packbits(f.sparse_group, bitorder="little") for f in families]),
        "packed_gap_affected": np.stack([np.packbits(f.gap_affected, bitorder="little") for f in families]),
        "gap_starts": np.asarray([f.gap_start for f in families], dtype=np.int16),
        "mask_sha256": np.asarray([[f.mask_hashes[name] for name in CONDITIONS] for f in families]),
        "family_sha256": np.asarray([f.family_hash for f in families]),
        "seed": np.asarray(seed, dtype=np.int64), "epoch": np.asarray(-1 if epoch is None else epoch, dtype=np.int64),
    }
    suffix = f"{split}_seed{seed}" + ("" if epoch is None else f"_epoch{epoch:03d}")
    path = Path(outdir) / f"masks_{suffix}.npz"
    _atomic_npz(path, arrays)
    payload = {
        "protocol": PROTOCOL, "dataset": bundle.dataset, "split": split, "seed": int(seed), "epoch": epoch,
        "rng": "PCG64DXSM", "seed_derivation": "domain-separated SHA256 first 128 bits big endian",
        "bitorder": "little", "mask_unpacked_shape": [WINDOW_LENGTH, bundle.flows],
        "mask_flatten_order": "C (time,flow)", "condition_order": list(CONDITIONS),
        "window_count": len(starts), "arrays_path": path.name, "arrays_sha256": _arrays_sha256(arrays),
        "npz_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "data_sha256": bundle.metadata["data_sha256"],
        "postfreeze_only": split == "dev" and seed in POSTFREEZE_MASK_SEEDS,
        "data_order_seed": None if split == "dev" else schedule.data_order_seed,
    }
    _atomic_json(path.with_suffix(".json"), payload)
    return payload


__all__ = [
    "PROTOCOL", "WINDOW_LENGTH", "CONDITIONS", "DATASET_SPECS", "DATA_ORDER_SEED",
    "TRAINING_MASK_SEED", "SELECTION_MASK_SEED", "POSTFREEZE_MASK_SEEDS",
    "DataBundle", "MaskFamily", "EpochMasks", "load_dataset", "load_fit_windows",
    "window_starts", "fit_statistics", "fit_neighbors", "make_mask_family", "training_epoch",
    "linear_fill", "stable_seed", "array_sha256", "save_data_manifest", "save_mask_registry",
]
