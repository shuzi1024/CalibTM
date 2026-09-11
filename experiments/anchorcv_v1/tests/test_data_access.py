from __future__ import annotations

import inspect

import pytest

from experiments.anchorcv_v1.data_access import (
    canonical_data_identity,
    load_permitted_windows,
    permitted_fit_fallback,
)


def test_loader_public_contract_exposes_no_path_split_or_test_override() -> None:
    parameters = tuple(inspect.signature(load_permitted_windows).parameters)

    assert parameters == ("dataset", "cohort")
    assert all(token not in " ".join(parameters).lower() for token in ("path", "split", "test", "override"))
    import experiments.anchorcv_v1.data_access as module

    assert not hasattr(module, "load_confirmation_windows")


@pytest.mark.parametrize("cohort", ["gate", "test", "validation_gap", "train_gap"])
def test_loader_rejects_every_unregistered_discovery_cohort(cohort: str) -> None:
    with pytest.raises(ValueError, match="fit, source_dev, or tune"):
        load_permitted_windows("geant", cohort)


def test_loader_returns_only_registered_permitted_window() -> None:
    windows = load_permitted_windows("geant", "source_dev")

    assert windows.dataset == "geant"
    assert windows.cohort == "source_dev"
    assert windows.values.shape == (10, 462, 50)
    assert windows.values.dtype.str == "<f4"
    assert not windows.values.flags.writeable


def test_data_identity_is_pinned_to_parsed_permitted_arrays() -> None:
    identity = canonical_data_identity("geant")

    assert identity["dataset"] == "geant"
    assert set(identity["parsed_array_sha256"]) == {"train", "val"}
    assert all(len(value) == 64 for value in identity["parsed_array_sha256"].values())
    assert len(identity["data_sha256"]) == 64
    assert identity["hash_scope"] == "canonical_parsed_permitted_split_arrays_only"
    assert "path" not in identity


def test_fit_fallback_is_finite_positive_and_dataset_bound() -> None:
    fallback = permitted_fit_fallback("abilene")

    assert fallback.mean >= 0.0
    assert fallback.std > 0.0


@pytest.mark.parametrize("dataset", ["wsdream", "GEANT", "", "../geant"])
def test_unknown_dataset_is_rejected(dataset: str) -> None:
    with pytest.raises(ValueError, match="abilene or geant"):
        canonical_data_identity(dataset)
