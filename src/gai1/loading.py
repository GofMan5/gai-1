from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gai1.config import ModelConfig
from gai1.lora import LoRAConfig, inject_lora
from gai1.model import GAIModel
from gai1.quantization import dequantize_state_dict


@dataclass(frozen=True)
class LoadOptions:
    checkpoint_path: str | Path
    device: str = "auto"
    dtype: str = "auto"
    adapter_path: str | Path | None = None
    context_length: int | None = None
    rope_scaling: str | None = None
    rope_scaling_factor: float | None = None
    rope_original_context: int | None = None


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is not available in this Python environment.")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def make_model_config(raw: dict[str, Any]) -> ModelConfig:
    allowed = ModelConfig.__dataclass_fields__.keys()
    return ModelConfig(**{key: value for key, value in raw.items() if key in allowed})


def load_model(options: LoadOptions) -> tuple[GAIModel, dict[str, Any]]:
    checkpoint_path = Path(options.checkpoint_path)
    device = resolve_device(options.device)
    dtype = resolve_dtype(options.dtype, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    fmt = checkpoint.get("format")

    checkpoint_metadata = checkpoint.get("metadata", {})

    if fmt == "gai1_checkpoint_v1":
        cfg = make_model_config(checkpoint["model_config"])
        state = checkpoint["model_state"]
        quantization = "none"
    elif fmt in {"gai1_quantized_checkpoint_v1", "gai1_quantized_checkpoint_v2"}:
        cfg = make_model_config(checkpoint["model_config"])
        state = dequantize_state_dict(checkpoint["model_state_quantized"], dtype=dtype)
        quantization = f"int{checkpoint.get('bits')}"
    else:
        raise ValueError(f"Unsupported checkpoint format: {fmt}")

    source_block_size = cfg.block_size
    if options.context_length is not None:
        if options.context_length < 1:
            raise ValueError("context_length must be positive")
        cfg.block_size = int(options.context_length)
        cfg.rope_original_context = options.rope_original_context or source_block_size
        if cfg.block_size > source_block_size and options.rope_scaling is None and cfg.rope_scaling == "none":
            cfg.rope_scaling = "linear"
            cfg.rope_scaling_factor = cfg.block_size / max(1, source_block_size)
    if options.rope_scaling is not None:
        cfg.rope_scaling = options.rope_scaling
    if options.rope_scaling_factor is not None:
        cfg.rope_scaling_factor = float(options.rope_scaling_factor)
    if options.rope_original_context is not None:
        cfg.rope_original_context = int(options.rope_original_context)

    model = GAIModel(cfg)
    if dtype in {torch.float16, torch.bfloat16}:
        model = model.to(dtype=dtype)
    model.load_state_dict(state)

    if options.adapter_path is not None:
        adapter_path = Path(options.adapter_path)
        adapter = torch.load(adapter_path, map_location="cpu")
        if adapter.get("format") != "gai1_lora_adapter_v1":
            raise ValueError(f"Unsupported adapter format: {adapter.get('format')}")
        rank = int(adapter.get("rank", 8))
        alpha = float(adapter.get("alpha", max(1, rank) * 2))
        dropout = float(adapter.get("dropout", 0.0))
        inject_lora(model, LoRAConfig(rank=rank, alpha=alpha, dropout=dropout))
        missing, unexpected = model.load_state_dict(adapter["state"], strict=False)
        unexpected_nonempty = [key for key in unexpected if key]
        if unexpected_nonempty:
            raise ValueError(f"Unexpected adapter keys: {unexpected_nonempty[:8]}")

    model.to(device)
    model.eval()
    return model, {
        "format": fmt,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "quantization": quantization,
        "checkpoint_path": str(checkpoint_path),
        "adapter_path": str(options.adapter_path) if options.adapter_path is not None else None,
        "context_length": cfg.block_size,
        "source_context_length": source_block_size,
        "trained_context_length": checkpoint_metadata.get("trained_context_length", source_block_size),
        "tested_context_length": checkpoint_metadata.get("tested_context_length", source_block_size),
        "rope_scaling": cfg.rope_scaling,
        "rope_scaling_factor": cfg.rope_scaling_factor,
        "rope_original_context": cfg.rope_original_context,
        "tokenizer": checkpoint.get("tokenizer"),
        "generation_config": checkpoint_metadata.get("generation_config") if isinstance(checkpoint_metadata, dict) else None,
        "quantization_policy": checkpoint_metadata.get("quantization_policy") if isinstance(checkpoint_metadata, dict) else None,
    }
