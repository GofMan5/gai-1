from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_keywords: tuple[str, ...] = ("qkv", "proj", "w1", "w2", "w3")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(1, rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=torch.float32)
        self.lora_b.to(device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = self.dropout(x).to(dtype=self.lora_a.weight.dtype)
        lora_out = self.lora_b(self.lora_a(lora_in)).to(dtype=base_out.dtype)
        return base_out + lora_out * self.scaling


def freeze_non_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)


def _replace_child(module: nn.Module, child_name: str, cfg: LoRAConfig) -> bool:
    child = getattr(module, child_name)
    if not isinstance(child, nn.Linear):
        return False
    setattr(module, child_name, LoRALinear(child, rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout))
    return True


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    targets: list[tuple[str, nn.Module, str]] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and any(keyword in full_name for keyword in cfg.target_keywords):
                targets.append((full_name, module, child_name))
    replaced: list[str] = []
    for full_name, module, child_name in targets:
        if _replace_child(module, child_name, cfg):
            replaced.append(full_name)
    freeze_non_lora(model)
    return replaced


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if "lora_" in name}


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
