from __future__ import annotations

from pathlib import PureWindowsPath

import torch

from gai1.quantization import dequantize_state_dict, pack_int4, quantize_state_dict, unpack_int4


def test_int4_pack_roundtrip() -> None:
    values = torch.tensor([-8, -7, -1, 0, 1, 6, 7], dtype=torch.int8)
    packed = pack_int4(values)
    restored = unpack_int4(packed, tuple(values.shape))
    assert torch.equal(restored, values)


def test_quantized_state_dict_roundtrip_shape() -> None:
    state = {
        "linear.weight": torch.randn(8, 8),
        "norm.weight": torch.ones(8),
    }
    records, stats = quantize_state_dict(state, bits=4)
    restored = dequantize_state_dict(records, dtype=torch.float32)
    assert restored["linear.weight"].shape == state["linear.weight"].shape
    assert restored["norm.weight"].shape == state["norm.weight"].shape
    assert stats.quantized_tensors == 1


def test_tied_lm_head_is_aliased() -> None:
    emb = torch.randn(16, 8)
    state = {
        "token_embedding.weight": emb,
        "lm_head.weight": emb,
    }
    records, stats = quantize_state_dict(state, bits=4)
    restored = dequantize_state_dict(records, dtype=torch.float32)
    assert records["lm_head.weight"]["alias"] == "token_embedding.weight"
    assert restored["lm_head.weight"] is restored["token_embedding.weight"]
    assert stats.aliased_tensors == 1


def test_lm_head_alias_requires_equal_weights() -> None:
    state = {
        "token_embedding.weight": torch.randn(16, 8),
        "lm_head.weight": torch.randn(16, 8),
    }
    records, stats = quantize_state_dict(state, bits=4)
    assert "alias" not in records["lm_head.weight"]
    assert stats.aliased_tensors == 0


def test_router_gate_kept_for_int4() -> None:
    state = {
        "blocks.0.ffn.gate.weight": torch.randn(4, 8),
        "blocks.0.ffn.experts.0.w1.weight": torch.randn(16, 8),
    }
    records, stats = quantize_state_dict(state, bits=4)
    assert records["blocks.0.ffn.gate.weight"]["quantized"] is False
    assert records["blocks.0.ffn.experts.0.w1.weight"]["quantized"] is True
    assert stats.quantized_tensors == 1


def test_release_metadata_source_path_is_relative() -> None:
    metadata = {"source_checkpoint": "outputs/gai1_train_gpu/last.pt"}
    source = metadata["source_checkpoint"]
    assert not PureWindowsPath(source).is_absolute()
    assert not source.startswith("/")
