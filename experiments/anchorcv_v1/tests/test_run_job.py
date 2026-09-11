from __future__ import annotations

import pytest

from experiments.anchorcv_v1.run_job import build_parser


def test_parser_exposes_only_frozen_job_axes_and_no_data_authority() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}

    assert destinations == {
        "help",
        "dataset",
        "seed_bundle",
        "stage",
        "output_root",
        "freeze_record",
    }
    assert not any(token in destinations for token in ("cohort", "mask", "path", "split", "test"))


def test_parser_rejects_unregistered_seed_and_stage() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset",
                "geant",
                "--seed-bundle",
                "4",
                "--stage",
                "prototype",
                "--output-root",
                "/tmp/x",
                "--freeze-record",
                "/tmp/freeze.json",
            ]
        )
