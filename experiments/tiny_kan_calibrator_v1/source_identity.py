"""Small source binding for the one-shot tiny-KAN run."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .jobs import canonical_json


PACKAGE = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE.parents[1]

_RUNTIME_FILES = (
    "experiments/tiny_kan_calibrator_v1/__init__.py",
    "experiments/tiny_kan_calibrator_v1/model.py",
    "experiments/tiny_kan_calibrator_v1/jobs.py",
    "experiments/tiny_kan_calibrator_v1/result_io.py",
    "experiments/tiny_kan_calibrator_v1/source_identity.py",
    "experiments/tiny_kan_calibrator_v1/run_job.py",
    "experiments/tiny_kan_calibrator_v1/queue.py",
    "experiments/tiny_kan_calibrator_v1/smoke.py",
    "experiments/tiny_kan_calibrator_v1/adjudicate.py",
    "experiments/tiny_kan_calibrator_v1/research_card.json",
    "experiments/minimal_calibrator_v1/model.py",
    "experiments/acil_innovation_v1/acil.py",
    "experiments/acil_innovation_v1/acil_core.py",
    "experiments/acil_innovation_v1/batching.py",
    "experiments/acil_innovation_v1/config.py",
    "experiments/acil_innovation_v1/configs/protocol_v1.json",
    "experiments/acil_innovation_v1/data.py",
    "experiments/acil_innovation_v1/masks.py",
    "experiments/acil_innovation_v1/models.py",
    "experiments/acil_innovation_v1/oracle_model.py",
    "experiments/acil_innovation_v1/preprocessing.py",
    "experiments/acil_innovation_v1/registries.py",
    "experiments/acil_innovation_v1/training.py",
    "experiments/acil_mechanism_v1/model.py",
    "experiments/acil_mechanism_v1/run_job.py",
    "experiments/anchorcv_v1/data_access.py",
    "experiments/sc_acil_v1/checkpoint.py",
    "experiments/sc_acil_v1/protocol.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record() -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for relative in _RUNTIME_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required runtime source is missing: {relative}")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    aggregate = hashlib.sha256(
        b"tiny-kan-calibrator-v1:source:v1\x00" + canonical_json(files)
    ).hexdigest()
    return {"aggregate_sha256": aggregate, "files": files}


def source_tree_sha256() -> str:
    return str(source_record()["aggregate_sha256"])


__all__ = [
    "PACKAGE",
    "PROJECT_ROOT",
    "file_sha256",
    "source_record",
    "source_tree_sha256",
]
