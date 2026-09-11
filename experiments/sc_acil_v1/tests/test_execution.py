from __future__ import annotations

import inspect

import pytest


def test_execution_dispatch_and_carriers_are_frozen() -> None:
    from experiments.sc_acil_v1.execution import carrier_labels

    assert carrier_labels("fit_acil", "acil_only") == ()
    assert carrier_labels("formal_gate", "sc_acil_u0") == (
        "linear_interpolation",
        "acil_only",
        "sc_acil_u0",
    )
    assert carrier_labels("formal_gate", "sc_acil") == ("sc_acil",)
    with pytest.raises(ValueError, match="handler"):
        carrier_labels("formal_gate", "fabricated")


def test_execute_job_has_no_scientific_override_surface() -> None:
    from experiments.sc_acil_v1.execution import execute_job

    assert tuple(inspect.signature(execute_job).parameters) == (
        "stage",
        "job_id",
        "output_root",
    )


def test_formal_gate_requires_all_exact_acil_dependencies(tmp_path) -> None:
    from experiments.sc_acil_v1.execution import require_all_acil_dependencies

    with pytest.raises(RuntimeError, match="six succeeded"):
        require_all_acil_dependencies(tmp_path, manifest_sha256="a" * 64)

