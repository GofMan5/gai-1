from __future__ import annotations

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

