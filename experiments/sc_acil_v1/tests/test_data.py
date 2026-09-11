from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_loader_exposes_only_registered_cohorts_and_exact_gate_windows() -> None:
    from experiments.sc_acil_v1.data import load_windows

    assert tuple(inspect.signature(load_windows).parameters) == ("dataset", "cohort")
    abilene = load_windows("abilene", "gate")
    geant = load_windows("geant", "gate")
    assert abilene.absolute_starts == tuple(range(37534, 41134, 50))
    assert geant.absolute_starts == tuple(range(8372, 9172, 50))
    assert abilene.values.shape == (72, 144, 50)
    assert geant.values.shape == (16, 462, 50)
    assert abilene.values.dtype == np.dtype("<f4")
    assert geant.values.dtype == np.dtype("<f4")
    assert not abilene.values.flags.writeable
    assert not geant.values.flags.writeable

    with pytest.raises(ValueError, match="registered"):
        load_windows("geant", "test")


def test_permitted_parent_semantic_hashes_are_pinned_without_raw_csv_access() -> None:
    from experiments.sc_acil_v1.data import permitted_parent_hashes

    assert permitted_parent_hashes() == {
        "abilene/train": "5ec5799c6a57c3a3971ab8b33fe95a8ea7c93a01955bec92440a75f38115d8d2",
        "abilene/val": "561be18adfff24be226c23e5a8937b9bceed5a5804bdc8c56c530ef8afe5626e",
        "geant/train": "233cb96106d01c6ef483ae01041d71188a0a37c65aa73fb1074b635f75e796f2",
        "geant/val": "337d1a0b6d1583d94332b80e22d2ecaf297827e3450666e3cc4f678eb45e05dc",
    }

