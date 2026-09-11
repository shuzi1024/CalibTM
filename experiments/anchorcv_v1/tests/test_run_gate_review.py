from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

import pytest

from experiments.anchorcv_v1 import run_gate_review


def _argv(tmp_path: Path) -> list[str]:
    return [
        "run_gate_review",
        "--output-root",
        str(tmp_path / "gate"),
        "--authority-record",
        str(tmp_path / "authority.json"),
        "--extension-report",
        str(tmp_path / "extension.json"),
        "--freeze-record",
        str(tmp_path / "freeze.json"),
        "--prototype-output-root",
        str(tmp_path / "prototype"),
        "--extension-output-root",
        str(tmp_path / "extension"),
        "--output",
        str(tmp_path / "final-review.json"),
    ]


def test_parser_exposes_only_operational_artifact_paths() -> None:
    destinations = {
        action.dest for action in run_gate_review.build_parser()._actions
    }
    assert destinations == {
        "help",
        "output_root",
        "authority_record",
        "extension_report",
        "freeze_record",
        "prototype_output_root",
        "extension_output_root",
        "output",
    }
    assert destinations.isdisjoint(
        {
            "dataset",
            "bundle",
            "seed_bundle",
            "cohort",
            "split",
            "test",
            "data",
            "device",
        }
    )


def test_cli_writes_exclusive_readonly_finite_json_and_prints_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {"verdict": "proceed", "finite": 1.0}
    monkeypatch.setattr(
        run_gate_review,
        "review_final_gate",
        lambda **kwargs: report,
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert run_gate_review.main() == 0
    output = tmp_path / "final-review.json"
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert output.stat().st_mode & stat.S_IWUSR == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "output": str(output),
        "verdict": "proceed",
    }

    with pytest.raises(FileExistsError):
        run_gate_review.main()


def test_cli_rejects_nonfinite_report_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_gate_review,
        "review_final_gate",
        lambda **kwargs: {"verdict": "revise", "bad": float("nan")},
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    with pytest.raises(ValueError, match="finite JSON"):
        run_gate_review.main()
    assert not (tmp_path / "final-review.json").exists()
