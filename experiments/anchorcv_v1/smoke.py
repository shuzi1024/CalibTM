"""Frozen one-update CUDA/CPU smoke for the complete AnchorCV model path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from experiments.acil_innovation_v1.batching import (
    build_evaluation_batch,
    build_training_batch,
)
from experiments.acil_innovation_v1.registries import seed_bundle
from experiments.acil_innovation_v1.training import build_protocol_optimizer
from experiments.sc2_ari_v1.initialization import load_frozen_acil

from .data_access import (
    canonical_data_identity,
    load_permitted_windows,
    permitted_fit_fallback,
)
from .evaluation import evaluate_tensor_batch, summarize_case_evidence
from .freeze import verify_freeze_record
from .job_runtime import (
    mask_identity_sha256,
    scientific_array_sha256,
    scientific_tensor_sha256,
)
from .protocol import fingerprint
from .training_runtime import (
    build_neural_expert,
    compute_batch_loss,
    model_parameter_count,
    seed_everything,
)


def run_smoke(
    *,
    dataset: str,
    seed_bundle_id: int,
    device: torch.device,
    freeze: dict[str, object],
) -> dict[str, object]:
    if dataset not in {"abilene", "geant"}:
        raise ValueError("dataset must be exactly abilene or geant")
    if seed_bundle_id not in {1, 2, 3}:
        raise ValueError("seed bundle must be exactly 1, 2, or 3")
    if freeze.get("config_sha256") != fingerprint():
        raise ValueError("smoke freeze config identity mismatch")
    if freeze.get("git_available") is not False or freeze.get("git_commit") is not None:
        raise ValueError("smoke freeze must record unavailable git identity")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")

    seeds = seed_bundle(seed_bundle_id)
    seed_everything(seeds.model, deterministic=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_neural_expert(dataset).to(device)
    initial_tensor_sha = scientific_tensor_sha256(model.state_dict())
    optimizer, optimizer_audit = build_protocol_optimizer(
        model, pretrained_parameters=()
    )
    fit = load_permitted_windows(dataset, "fit")
    fallback = permitted_fit_fallback(dataset)
    optimizer.zero_grad(set_to_none=True)
    loss_values = []
    started = time.monotonic()
    for batch_index in range(4):
        batch = build_training_batch(
            fit,
            seed_bundle=seed_bundle_id,
            epoch=0,
            batch_index=batch_index,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, details = compute_batch_loss(
                model,
                batch,
                fit_fallback=fallback,
                device=device,
            )
        (loss / 4.0).backward()
        loss_values.append(float(loss.detach().float().item()))
        if details.drop_rank != 1:
            raise RuntimeError("smoke middle-anchor contract drifted")
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if not gradients or any(
        not torch.isfinite(gradient).all().item()
        for gradient in gradients.values()
    ):
        raise FloatingPointError("smoke gradients are absent or nonfinite")
    gradient_sha = scientific_tensor_sha256(gradients)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        tuple(model.parameters()), 1.0
    )
    if not torch.isfinite(torch.as_tensor(gradient_norm)).item():
        raise FloatingPointError("smoke gradient norm is nonfinite")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updated_tensor_sha = scientific_tensor_sha256(model.state_dict())
    if updated_tensor_sha == initial_tensor_sha:
        raise RuntimeError("smoke optimizer update did not change model tensors")

    model.eval()
    prior, prior_record = load_frozen_acil(
        dataset, seed_bundle_id, device=device
    )
    source_windows = load_permitted_windows(dataset, "source_dev")
    evaluation_batch = build_evaluation_batch(
        source_windows,
        seed_bundle=seed_bundle_id,
        family="random",
    )
    evidence = evaluate_tensor_batch(
        prior=prior,
        neural=model,
        model_input=evaluation_batch.model_input[:2].to(device),
        truth=evaluation_batch.truth[:2].to(device),
        observed=evaluation_batch.observed[:2].to(device),
        fit_fallback=fallback,
        window_offset=0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    scientific = {
        "protocol": "anchorcv-v1",
        "source_tree_sha256": freeze["source_tree_sha256"],
        "config_sha256": freeze["config_sha256"],
        "data_sha256": canonical_data_identity(dataset)["data_sha256"],
        "dataset": dataset,
        "seed_bundle": seed_bundle_id,
        "model_seed": seeds.model,
        "parameter_count": model_parameter_count(model),
        "initial_tensor_sha256": initial_tensor_sha,
        "gradient_tensor_sha256": gradient_sha,
        "updated_tensor_sha256": updated_tensor_sha,
        "loss_values": loss_values,
        "gradient_norm_before_clip": float(
            torch.as_tensor(gradient_norm).detach().float().cpu().item()
        ),
        "optimizer_parameter_names": {
            name: list(values) for name, values in optimizer_audit.items()
        },
        "prior_checkpoint_file_sha256": prior_record.file_sha256,
        "evidence_content_sha256": scientific_array_sha256(
            evidence.as_npz_dict()
        ),
        "mask_sha256": mask_identity_sha256(
            evaluation_batch.mask_sha256[:2]
        ),
        "evaluation_summary": summarize_case_evidence(evidence),
        "test_access": False,
    }
    return {
        "status": "succeeded",
        "scientific_smoke": scientific,
        "operational": {
            "device_type": device.type,
            "elapsed_seconds": elapsed,
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("abilene", "geant"))
    parser.add_argument("--seed-bundle", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    freeze = verify_freeze_record(args.freeze_record)
    result = run_smoke(
        dataset=args.dataset,
        seed_bundle_id=args.seed_bundle,
        device=torch.device(args.device),
        freeze=freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    print(json.dumps({"status": "succeeded", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "run_smoke"]
