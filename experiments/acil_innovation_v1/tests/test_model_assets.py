from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.acil_innovation_v1 import model_assets


REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


class _FakeAttention(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.c_attn = nn.Linear(width, 3 * width)
        self.c_proj = nn.Linear(width, width)
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)


class _FakeBlock(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = _FakeAttention(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 8), nn.GELU(), nn.Linear(8, width))


class _FakeGPT2(nn.Module):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.h = nn.ModuleList([_FakeBlock() for _ in range(6)])
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.add_(offset)
        self.wte = nn.Embedding(3, 4)
        self.wpe = nn.Embedding(3, 4)
        self.ln_f = nn.LayerNorm(4)


def _schema(module: nn.Module):
    return tuple((name, tuple(value.shape)) for name, value in module.named_parameters())


def test_asset_constants_inventory_and_real_hashes_are_exact() -> None:
    expected_root = (
        Path(model_assets.__file__).resolve().parent / "assets" / "gpt2" / REVISION
    )
    assert model_assets.GPT2_REVISION == REVISION
    assert model_assets.GPT2_ASSET_ROOT == expected_root
    assert model_assets.EXPECTED_ASSET_SHA256 == {
        "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
        "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
    }
    record = model_assets.pinned_asset_record()
    assert [item["name"] for item in record["files"]] == [
        "config.json",
        "model.safetensors",
    ]
    assert record["status"] == "validated_offline_asset"
    assert record["revision"] == REVISION


def test_loader_public_api_has_no_path_revision_depth_or_network_override() -> None:
    assert tuple(inspect.signature(model_assets.load_pinned_gpt2_blocks).parameters) == (
        "init",
        "seed",
    )
    forbidden = ("path", "revision", "depth", "cache", "remote", "network")
    assert not any(
        token in name.lower()
        for name in inspect.signature(model_assets.load_pinned_gpt2_blocks).parameters
        for token in forbidden
    )


def test_loader_forces_local_safetensors_and_retains_only_six_allowed_blocks(
    monkeypatch,
) -> None:
    captured = {}

    def fake_pretrained(root):
        captured["root"] = root
        captured["local_files_only"] = True
        captured["use_safetensors"] = True
        return model_assets._retained_blocks(_FakeGPT2())

    monkeypatch.setattr(model_assets, "pinned_asset_record", lambda: {"status": "validated_offline_asset"})
    monkeypatch.setattr(model_assets, "_load_pretrained", fake_pretrained)
    blocks = model_assets.load_pinned_gpt2_blocks("pretrained", 41001)

    assert captured == {
        "root": model_assets.GPT2_ASSET_ROOT,
        "local_files_only": True,
        "use_safetensors": True,
    }
    assert isinstance(blocks, nn.ModuleList)
    assert len(blocks) == 6
    assert set(vars(blocks[0])["_modules"]) == {
        "attn_dropout",
        "c_attn",
        "c_proj",
        "ln_1",
        "ln_2",
        "mlp",
        "resid_dropout",
    }
    forbidden = ("wte", "wpe", "tokenizer", "prompt", "ln_f", "lm_head")
    assert not any(any(token in name for token in forbidden) for name in blocks.state_dict())


def test_pretrained_and_random_are_shape_isomorphic_and_random_seed_is_explicit(
    monkeypatch,
) -> None:
    seen = []

    def fake_pretrained(root):
        return model_assets._retained_blocks(_FakeGPT2(offset=0.0))

    def fake_random(root, seed):
        seen.append(seed)
        return model_assets._retained_blocks(_FakeGPT2(offset=float(seed) / 1e6))

    monkeypatch.setattr(model_assets, "pinned_asset_record", lambda: {"status": "validated_offline_asset"})
    monkeypatch.setattr(model_assets, "_load_pretrained", fake_pretrained)
    monkeypatch.setattr(model_assets, "_load_random", fake_random)
    pretrained = model_assets.load_pinned_gpt2_blocks("pretrained", 41001)
    first = model_assets.load_pinned_gpt2_blocks("random", 41001)
    second = model_assets.load_pinned_gpt2_blocks("random", 41002)

    assert seen == [41001, 41002]
    assert _schema(pretrained) == _schema(first) == _schema(second)
    assert model_assets.tensor_state_sha256(first.state_dict()) != model_assets.tensor_state_sha256(
        second.state_dict()
    )
    for bad in (None, True, -1, 1.5, "41001"):
        with pytest.raises((TypeError, ValueError), match="seed"):
            model_assets.load_pinned_gpt2_blocks("random", bad)


@pytest.mark.parametrize("init", ["latest", "cache", "randomly", "pretrained/main"])
def test_only_exact_pretrained_or_random_init_is_accepted(init, monkeypatch) -> None:
    monkeypatch.setattr(model_assets, "pinned_asset_record", lambda: {})
    with pytest.raises(ValueError, match="init"):
        model_assets.load_pinned_gpt2_blocks(init, 41001)
