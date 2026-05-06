from __future__ import annotations

from pathlib import PureWindowsPath

import pytest
import torch

from gai1.quantization import dequantize_state_dict, pack_int4, quantize_state_dict, unpack_int4, validate_quantized_records
from scripts.export_quantized import default_generation_config


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


def test_router_gate_can_be_quantized_for_int4_override() -> None:
    state = {
        "blocks.0.ffn.gate.weight": torch.randn(4, 8),
    }
    records, stats = quantize_state_dict(state, bits=4, keep_router_fp16=False)
    assert records["blocks.0.ffn.gate.weight"]["quantized"] is True
    assert stats.quantized_tensors == 1


def test_router_gate_quantized_for_int8() -> None:
    state = {
        "blocks.0.ffn.gate.weight": torch.randn(4, 8),
    }
    records, stats = quantize_state_dict(state, bits=8)
    assert records["blocks.0.ffn.gate.weight"]["quantized"] is True
    assert stats.quantized_tensors == 1


def test_dequantize_alias_can_point_forward() -> None:
    emb = torch.randn(16, 8)
    records, _stats = quantize_state_dict({"token_embedding.weight": emb}, bits=4)
    records = {"lm_head.weight": {"alias": "token_embedding.weight"}, **records}
    restored = dequantize_state_dict(records, dtype=torch.float32)
    assert restored["lm_head.weight"] is restored["token_embedding.weight"]


def test_dequantize_rejects_missing_alias_target() -> None:
    with pytest.raises(ValueError, match="Alias target not found"):
        dequantize_state_dict({"lm_head.weight": {"alias": "token_embedding.weight"}}, dtype=torch.float32)


def test_validate_quantized_records_rejects_bits_mismatch() -> None:
    records, _stats = quantize_state_dict({"linear.weight": torch.randn(4, 4)}, bits=4)
    with pytest.raises(ValueError, match="does not match checkpoint bits"):
        validate_quantized_records(records, checkpoint_bits=8)


def test_release_metadata_source_path_is_relative() -> None:
    metadata = {"source_checkpoint": "outputs/gai1_train_gpu/last.pt"}
    source = metadata["source_checkpoint"]
    assert not PureWindowsPath(source).is_absolute()
    assert not source.startswith("/")


def test_default_generation_config_chat_template_is_utf8() -> None:
    config = default_generation_config({})
    assert config["chat_template"] == "Пользователь: {prompt}\nАссистент:"
    assert "Ð" not in config["chat_template"]
