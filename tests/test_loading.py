from __future__ import annotations

import pytest
import torch

from gai1.config import ModelConfig
from gai1.loading import LoadOptions, load_model
from gai1.lora import LoRAConfig, inject_lora, lora_state_dict
from gai1.model import GAIModel
from gai1.quantization import quantize_state_dict


def test_load_plain_checkpoint(tmp_path) -> None:
    cfg = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False)
    model = GAIModel(cfg)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format": "gai1_checkpoint_v1",
            "step": 0,
            "model_config": model.config_dict(),
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    loaded, metadata = load_model(LoadOptions(checkpoint_path=checkpoint, device="cpu", dtype="fp32"))
    assert metadata["format"] == "gai1_checkpoint_v1"
    assert next(loaded.parameters()).device.type == "cpu"


def test_load_lora_adapter(tmp_path) -> None:
    cfg = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False)
    base = GAIModel(cfg)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format": "gai1_checkpoint_v1",
            "step": 0,
            "model_config": base.config_dict(),
            "model_state": base.state_dict(),
        },
        checkpoint,
    )
    adapter_model = GAIModel(cfg)
    inject_lora(adapter_model, LoRAConfig(rank=2, alpha=4.0, dropout=0.0))
    adapter = tmp_path / "adapter.pt"
    torch.save(
        {
            "format": "gai1_lora_adapter_v1",
            "state": lora_state_dict(adapter_model),
            "rank": 2,
            "alpha": 4.0,
            "dropout": 0.0,
        },
        adapter,
    )
    loaded, metadata = load_model(LoadOptions(checkpoint_path=checkpoint, adapter_path=adapter, device="cpu", dtype="fp32"))
    assert metadata["adapter_path"] == str(adapter)
    assert any("lora_" in name for name, _param in loaded.named_parameters())


def test_load_checkpoint_with_extended_context(tmp_path) -> None:
    cfg = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False)
    model = GAIModel(cfg)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format": "gai1_checkpoint_v1",
            "step": 0,
            "model_config": model.config_dict(),
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    loaded, metadata = load_model(LoadOptions(checkpoint_path=checkpoint, context_length=32, device="cpu", dtype="fp32"))
    assert loaded.cfg.block_size == 32
    assert metadata["source_context_length"] == 8
    assert metadata["context_length"] == 32
    assert metadata["rope_scaling"] == "linear"


def test_load_checkpoint_exposes_tokenizer_metadata(tmp_path) -> None:
    cfg = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False)
    model = GAIModel(cfg)
    tokenizer_metadata = {"kind": "byte", "vocab_size": 260}
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format": "gai1_checkpoint_v1",
            "step": 0,
            "model_config": model.config_dict(),
            "model_state": model.state_dict(),
            "tokenizer": tokenizer_metadata,
        },
        checkpoint,
    )

    _loaded, metadata = load_model(LoadOptions(checkpoint_path=checkpoint, device="cpu", dtype="fp32"))

    assert metadata["tokenizer"] == tokenizer_metadata


def test_load_quantized_checkpoint_exposes_release_metadata(tmp_path) -> None:
    cfg = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False)
    model = GAIModel(cfg)
    records, _stats = quantize_state_dict(model.state_dict(), bits=8)
    tokenizer_metadata = {"kind": "byte", "vocab_size": 260}
    generation_config = {"max_new_tokens": 32, "chat_template": "Пользователь: {prompt}\nАссистент:"}
    quantization_policy = {"bits": 8, "scheme": "symmetric_per_tensor_storage"}
    checkpoint = tmp_path / "model_int8.pt"
    torch.save(
        {
            "format": "gai1_quantized_checkpoint_v2",
            "bits": 8,
            "model_config": model.config_dict(),
            "model_state_quantized": records,
            "tokenizer": tokenizer_metadata,
            "generation_config": generation_config,
            "quantization_policy": quantization_policy,
        },
        checkpoint,
    )

    _loaded, metadata = load_model(LoadOptions(checkpoint_path=checkpoint, device="cpu", dtype="fp32"))

    assert metadata["quantization"] == "int8"
    assert metadata["tokenizer"] == tokenizer_metadata
    assert metadata["generation_config"] == generation_config
    assert metadata["quantization_policy"] == quantization_policy


def test_load_quantized_v2_rejects_missing_tokenizer(tmp_path) -> None:
    cfg = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False)
    model = GAIModel(cfg)
    records, _stats = quantize_state_dict(model.state_dict(), bits=8)
    checkpoint = tmp_path / "model_int8.pt"
    torch.save(
        {
            "format": "gai1_quantized_checkpoint_v2",
            "bits": 8,
            "model_config": model.config_dict(),
            "model_state_quantized": records,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="must include tokenizer metadata"):
        load_model(LoadOptions(checkpoint_path=checkpoint, device="cpu", dtype="fp32"))


def test_load_quantized_rejects_unsupported_bits(tmp_path) -> None:
    cfg = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False)
    model = GAIModel(cfg)
    records, _stats = quantize_state_dict(model.state_dict(), bits=8)
    checkpoint = tmp_path / "model_int3.pt"
    torch.save(
        {
            "format": "gai1_quantized_checkpoint_v2",
            "bits": 3,
            "model_config": model.config_dict(),
            "model_state_quantized": records,
            "tokenizer": {"kind": "byte", "vocab_size": 260},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="Unsupported quantized checkpoint bits"):
        load_model(LoadOptions(checkpoint_path=checkpoint, device="cpu", dtype="fp32"))


def test_load_quantized_rejects_malformed_record(tmp_path) -> None:
    cfg = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False)
    model = GAIModel(cfg)
    records, _stats = quantize_state_dict(model.state_dict(), bits=8)
    first_key = next(iter(records))
    records[first_key] = {"quantized": True, "bits": 8, "shape": (1,)}
    checkpoint = tmp_path / "model_bad.pt"
    torch.save(
        {
            "format": "gai1_quantized_checkpoint_v2",
            "bits": 8,
            "model_config": model.config_dict(),
            "model_state_quantized": records,
            "tokenizer": {"kind": "byte", "vocab_size": 260},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="Missing quantization scale tensor"):
        load_model(LoadOptions(checkpoint_path=checkpoint, device="cpu", dtype="fp32"))
