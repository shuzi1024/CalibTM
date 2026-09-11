from __future__ import annotations

from dataclasses import asdict
import hashlib

from experiments.acil_innovation_v1 import config as config_module
from experiments.acil_innovation_v1 import masks as masks_module
from experiments.acil_innovation_v1 import registries as registries_module
from experiments.acil_innovation_v1.config import canonical_json_bytes
from experiments.acil_innovation_v1.masks import build_mixed_training_mask


def _clear_optional_runtime_caches() -> None:
    """Cold-start private caches without making them part of the public API."""

    for module, names in (
        (config_module, ("_cached_fixed_config",)),
        (
            registries_module,
            (
                "_cached_dataset_spec",
                "_cached_cohort_spec",
                "_cached_ordered_window_starts",
                "_cached_window_schedule_sha256",
                "_cached_seed_bundle",
                "_cached_stage_spec",
            ),
        ),
        (masks_module, ("_cached_mask_context",)),
    ):
        for name in names:
            candidate = getattr(module, name, None)
            cache_clear = getattr(candidate, "cache_clear", None)
            if cache_clear is not None:
                cache_clear()


def _bundle_regression_sha256(bundle) -> str:
    payload = {
        "controlled_gap": bundle.controlled_gap.tolist(),
        "identity": asdict(bundle.identity),
        "observed": bundle.observed.tolist(),
        "target": bundle.target.tolist(),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def test_many_mixed_training_masks_read_the_fixed_config_once(monkeypatch):
    _clear_optional_runtime_caches()
    real_read = config_module._read_fixed_config
    read_count = 0

    def counting_read():
        nonlocal read_count
        read_count += 1
        return real_read()

    monkeypatch.setattr(config_module, "_read_fixed_config", counting_read)
    geant_fit_starts = tuple(
        (index * (7023 - 50)) // (512 - 1) for index in range(512)
    )

    for index in range(128):
        bundle = build_mixed_training_mask(
            dataset="geant",
            window_start=geant_fit_starts[index],
            flow_index=index % 462,
            seed_bundle=index % 6 + 1,
            epoch=index % 20,
            epoch_order_position=index,
        )
        assert int(bundle.observed.sum()) == 3

    assert read_count == 1


def test_cached_hot_path_preserves_frozen_scalar_mask_semantics():
    cases = (
        (
            ("abilene", 0, 0, 1, 0, 0),
            "e21cfed8e1a261c6aa04688a6279eb6c52c87f01e16eb1251d5ae45024bdbd8c",
        ),
        (
            ("abilene", 8272, 143, 2, 7, 11),
            "fd5fe689e50a7f4348c7976738197e521096b251dd8c60f553767032615d9e6f",
        ),
        (
            ("abilene", 33285, 10, 6, 19, 511),
            "333cb9ab6bb5d539a1fb4d4afb58d933260ab753392930c27854c7fed9124e31",
        ),
        (
            ("geant", 0, 0, 3, 0, 2),
            "224e2f624c85c17095db7516ff0fd6698b199ca1f290da504cd3be2b29b430ce",
        ),
        (
            ("geant", 3506, 461, 4, 8, 256),
            "8f26ec2dc79e3437ee5a4d3b1aea52cd419b35e1b3af2fdad07d9e108166d0a6",
        ),
        (
            ("geant", 6973, 173, 6, 19, 510),
            "cde5e80b7630baa1ee20475bd990bc71237a3d029b99d7daf4fcfaf7c16de0ff",
        ),
    )

    for args, expected_sha256 in cases:
        bundle = build_mixed_training_mask(*args)
        assert _bundle_regression_sha256(bundle) == expected_sha256
