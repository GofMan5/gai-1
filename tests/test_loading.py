from __future__ import annotations

import torch

from gai1.config import ModelConfig
from gai1.loading import LoadOptions, load_model
from gai1.lora import LoRAConfig, inject_lora, lora_state_dict
from gai1.model import GAIModel


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
