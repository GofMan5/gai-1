from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class QuantizationStats:
    original_bytes: int
    quantized_bytes: int
    quantized_tensors: int
    kept_tensors: int
    aliased_tensors: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.quantized_bytes == 0:
            return 0.0
        return self.original_bytes / self.quantized_bytes


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def is_router_tensor(name: str) -> bool:
    lowered = name.casefold()
    return ".gate.weight" in lowered or "router" in lowered


def should_quantize_tensor(
    name: str,
    tensor: torch.Tensor,
    keep_norm_fp16: bool = True,
    bits: int = 8,
    keep_router_fp16: bool = False,
) -> bool:
    if not tensor.is_floating_point():
        return False
    if tensor.ndim < 2:
        return False
    lowered = name.casefold()
    if keep_norm_fp16 and ("norm" in lowered or "bias" in lowered):
        return False
    if keep_router_fp16 and bits == 4 and is_router_tensor(name):
        return False
    return True


def quantize_tensor_symmetric(tensor: torch.Tensor, bits: int) -> dict[str, Any]:
    if bits not in {4, 8}:
        raise ValueError("Only 4-bit and 8-bit quantization are supported")
    source = tensor.detach().cpu().float().contiguous()
    qmax = (2 ** (bits - 1)) - 1
    scale = source.abs().max().clamp_min(1e-8) / qmax
    q = torch.round(source / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    if bits == 8:
        payload = q
    else:
        payload = pack_int4(q)
    return {
        "quantized": True,
        "bits": bits,
        "shape": tuple(source.shape),
        "scale": scale.to(torch.float16),
        "payload": payload,
    }


def dequantize_tensor_symmetric(record: dict[str, Any], dtype: torch.dtype = torch.float16) -> torch.Tensor:
    bits = int(record["bits"])
    shape = tuple(record["shape"])
    scale = record["scale"].float()
    if bits == 8:
        q = record["payload"].to(torch.float32)
    elif bits == 4:
        q = unpack_int4(record["payload"], shape).to(torch.float32)
    else:
        raise ValueError(f"Unsupported quantization bits: {bits}")
    return (q * scale).reshape(shape).to(dtype)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    flat = q.reshape(-1).to(torch.int16)
    flat = (flat + 8).clamp(0, 15).to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat((flat, torch.zeros(1, dtype=torch.uint8)))
    low = flat[0::2]
    high = flat[1::2] << 4
    return low | high


def unpack_int4(payload: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    packed = payload.reshape(-1).to(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values = torch.empty(packed.numel() * 2, dtype=torch.int16)
    values[0::2] = low.to(torch.int16)
    values[1::2] = high.to(torch.int16)
    values = values[:total] - 8
    return values.to(torch.int8).reshape(shape)


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    bits: int,
    keep_norm_fp16: bool = True,
    keep_router_fp16: bool = True,
    tie_lm_head: bool = True,
) -> tuple[dict[str, Any], QuantizationStats]:
    records: dict[str, Any] = {}
    original_bytes = 0
    quantized_bytes = 0
    quantized_tensors = 0
    kept_tensors = 0
    aliased_tensors = 0
    token_embedding = state_dict.get("token_embedding.weight")
    for name, tensor in state_dict.items():
        original_bytes += tensor_nbytes(tensor)
        can_alias_lm_head = (
            tie_lm_head
            and name == "lm_head.weight"
            and "token_embedding.weight" in records
            and token_embedding is not None
            and tuple(tensor.shape) == tuple(token_embedding.shape)
            and torch.equal(tensor.detach().cpu(), token_embedding.detach().cpu())
        )
        if can_alias_lm_head:
            records[name] = {"alias": "token_embedding.weight"}
            aliased_tensors += 1
            continue
        if should_quantize_tensor(name, tensor, keep_norm_fp16=keep_norm_fp16, bits=bits, keep_router_fp16=keep_router_fp16):
            record = quantize_tensor_symmetric(tensor, bits=bits)
            records[name] = record
            quantized_bytes += tensor_nbytes(record["payload"]) + tensor_nbytes(record["scale"])
            quantized_tensors += 1
        else:
            kept = tensor.detach().cpu()
            if kept.is_floating_point():
                kept = kept.to(torch.float16)
            records[name] = {"quantized": False, "payload": kept}
            quantized_bytes += tensor_nbytes(kept)
            kept_tensors += 1
    return records, QuantizationStats(
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
        quantized_tensors=quantized_tensors,
        kept_tensors=kept_tensors,
        aliased_tensors=aliased_tensors,
    )


def dequantize_state_dict(records: dict[str, Any], dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, record in records.items():
        if "alias" in record:
            state[name] = state[record["alias"]]
        elif record.get("quantized"):
            state[name] = dequantize_tensor_symmetric(record, dtype=dtype)
        else:
            payload = record["payload"]
            state[name] = payload.to(dtype) if payload.is_floating_point() else payload
    return state
