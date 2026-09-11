"""Fixed offline GPT-2 asset boundary for six retained set-attention blocks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import stat
from typing import Final, Mapping

import torch
from torch import Tensor, nn


GPT2_REVISION: Final = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPT2_ASSET_ROOT: Path = (
    Path(__file__).resolve().parent / "assets" / "gpt2" / GPT2_REVISION
)
EXPECTED_ASSET_SHA256: Final = {
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
}
_DEPTH: Final = 6
_INITIALIZATIONS: Final = ("pretrained", "random")
_TENSOR_HASH_DOMAIN: Final = b"acil-innovation-v1:gpt2-tensor-state:v1\x00"


class RetainedGPT2Block(nn.Module):
    """Only submodules exercised by the custom noncausal set forward."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        if not isinstance(source, nn.Module) or not hasattr(source, "attn"):
            raise TypeError("source must be one Hugging Face GPT-2 block")
        self.ln_1 = copy.deepcopy(source.ln_1)
        self.c_attn = copy.deepcopy(source.attn.c_attn)
        self.c_proj = copy.deepcopy(source.attn.c_proj)
        self.attn_dropout = copy.deepcopy(source.attn.attn_dropout)
        self.resid_dropout = copy.deepcopy(source.attn.resid_dropout)
        self.ln_2 = copy.deepcopy(source.ln_2)
        self.mlp = copy.deepcopy(source.mlp)


def _sha256_regular_file(path: Path) -> tuple[int, str]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"GPT-2 asset must be a regular non-symlink: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"GPT-2 asset mutated while hashing: {path.name}")
    return int(before.st_size), digest.hexdigest()


def pinned_asset_record() -> dict[str, object]:
    """Validate the sole local GPT-2 directory and its exact two files."""

    root = GPT2_ASSET_ROOT
    try:
        metadata = root.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError("fixed GPT-2 asset root is absent") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("fixed GPT-2 asset root must be a non-symlink directory")
    children = sorted(root.iterdir(), key=lambda item: item.name)
    if [item.name for item in children] != sorted(EXPECTED_ASSET_SHA256):
        raise RuntimeError(
            "fixed GPT-2 asset root must contain exactly config.json and model.safetensors"
        )
    files = []
    for path in children:
        size, digest = _sha256_regular_file(path)
        if digest != EXPECTED_ASSET_SHA256[path.name]:
            raise RuntimeError(f"fixed GPT-2 asset hash drifted: {path.name}")
        files.append({"name": path.name, "sha256": digest, "size": size})
    return {
        "files": files,
        "repository": "openai-community/gpt2",
        "revision": GPT2_REVISION,
        "root": str(root),
        "status": "validated_offline_asset",
    }


def _retained_blocks(model: nn.Module) -> nn.ModuleList:
    blocks = getattr(model, "h", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) < _DEPTH:
        raise RuntimeError("deserialized GPT-2 model has fewer than six blocks")
    return nn.ModuleList(RetainedGPT2Block(blocks[index]) for index in range(_DEPTH))


def _load_pretrained(root: Path) -> nn.ModuleList:
    try:
        from transformers import GPT2Model
    except ImportError as exc:  # pragma: no cover - formal runtime owns dependency
        raise RuntimeError("fixed runtime is missing transformers") from exc
    temporary = GPT2Model.from_pretrained(
        str(root),
        local_files_only=True,
        use_safetensors=True,
    )
    retained = _retained_blocks(temporary)
    del temporary
    return retained


def _load_random(root: Path, seed: int) -> nn.ModuleList:
    try:
        from transformers import GPT2Config, GPT2Model
    except ImportError as exc:  # pragma: no cover - formal runtime owns dependency
        raise RuntimeError("fixed runtime is missing transformers") from exc
    configuration = GPT2Config.from_json_file(str(root / "config.json"))
    configuration.n_layer = _DEPTH
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        temporary = GPT2Model(configuration)
    retained = _retained_blocks(temporary)
    del temporary
    return retained


def load_pinned_gpt2_blocks(init: str, seed: int) -> nn.ModuleList:
    """Load six retained GPT blocks without any path, revision, or network option."""

    if init not in _INITIALIZATIONS:
        raise ValueError(f"init must be exactly one of {_INITIALIZATIONS!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an explicit nonnegative integer")
    if seed < 0:
        raise ValueError("seed must be an explicit nonnegative integer")
    record = pinned_asset_record()
    if record.get("status") != "validated_offline_asset":
        raise RuntimeError("GPT-2 asset validation did not complete")
    if init == "pretrained":
        blocks = _load_pretrained(GPT2_ASSET_ROOT)
    else:
        blocks = _load_random(GPT2_ASSET_ROOT, seed)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != _DEPTH:
        raise RuntimeError("GPT-2 loader did not return exactly six retained blocks")
    return blocks


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Stable semantic hash for a named retained tensor state."""

    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    digest = hashlib.sha256(_TENSOR_HASH_DOMAIN)
    for name in sorted(state):
        if not isinstance(name, str) or not name:
            raise TypeError("tensor state names must be nonempty strings")
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"tensor state value {name!r} is not a tensor")
        canonical = tensor.detach().cpu().contiguous()
        fields = (
            name.encode("utf-8"),
            str(canonical.dtype).encode("ascii"),
            json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"),
            canonical.view(torch.uint8).reshape(-1).numpy().tobytes(),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


__all__ = [
    "EXPECTED_ASSET_SHA256",
    "GPT2_ASSET_ROOT",
    "GPT2_REVISION",
    "RetainedGPT2Block",
    "load_pinned_gpt2_blocks",
    "pinned_asset_record",
    "tensor_state_sha256",
]
