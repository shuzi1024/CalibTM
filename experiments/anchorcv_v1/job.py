"""Content-addressed scientific identity for one AnchorCV training job."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from experiments.acil_innovation_v1.registries import dataset_spec, seed_bundle
from experiments.sc2_ari_v1.initialization import acil_checkpoint_record

from .data_access import canonical_data_identity
from .protocol import fingerprint, load_protocol


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOB_DOMAIN = b"anchorcv-v1:job:v1\x00"


@dataclass(frozen=True, slots=True)
class JobSpec:
    dataset: str
    seed_bundle: int
    stage: str
    output_root: Path
    device: str
    source_tree_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        protocol = load_protocol()
        if self.dataset not in protocol.datasets:
            raise ValueError("dataset must be exactly abilene or geant")
        if self.stage not in {"prototype", "extension"}:
            raise ValueError("stage must be exactly prototype or extension")
        expected_bundles = (
            protocol.prototype_bundles
            if self.stage == "prototype"
            else protocol.extension_bundles
        )
        if (
            isinstance(self.seed_bundle, bool)
            or not isinstance(self.seed_bundle, int)
            or self.seed_bundle not in expected_bundles
        ):
            raise ValueError("seed bundle does not belong to the requested stage")
        if self.device != "cuda":
            raise ValueError("formal device must be exactly the registered CUDA lane")
        if (
            not isinstance(self.source_tree_sha256, str)
            or _SHA256.fullmatch(self.source_tree_sha256) is None
        ):
            raise ValueError("source_tree_sha256 must be a lowercase SHA-256")
        if self.config_sha256 != fingerprint():
            raise ValueError("config_sha256 differs from the frozen protocol")
        output_root = Path(self.output_root)
        if output_root == Path("/"):
            raise ValueError("output_root cannot be the filesystem root")
        object.__setattr__(self, "output_root", output_root.resolve())

    @property
    def scientific_identity(self) -> dict[str, object]:
        protocol = load_protocol()
        data = canonical_data_identity(self.dataset)
        acil = acil_checkpoint_record(self.dataset, self.seed_bundle)
        seeds = seed_bundle(self.seed_bundle)
        return {
            "acil_checkpoint_file_sha256": acil.file_sha256,
            "config_sha256": self.config_sha256,
            "compute_lane": "cuda_bf16_math_sdp",
            "data_sha256": data["data_sha256"],
            "dataset": self.dataset,
            "evaluation_cohort": "tune",
            "flows": dataset_spec(self.dataset).flows,
            "git_available": False,
            "git_commit": None,
            "mask_families": list(protocol.mask_families),
            "model_seed": seeds.model,
            "parsed_array_sha256": data["parsed_array_sha256"],
            "protocol": protocol.protocol_id,
            "seed_bundle": self.seed_bundle,
            "source_tree_sha256": self.source_tree_sha256,
            "stage": self.stage,
            "test_access": False,
            "training_cohort": "fit",
            "checkpoint_selection_cohort": "source_dev",
        }

    @property
    def identity_json(self) -> str:
        return json.dumps(
            self.scientific_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def job_id(self) -> str:
        return hashlib.sha256(
            _JOB_DOMAIN + self.identity_json.encode("ascii")
        ).hexdigest()

    @property
    def job_directory(self) -> Path:
        return self.output_root / self.job_id


__all__ = ["JobSpec"]
