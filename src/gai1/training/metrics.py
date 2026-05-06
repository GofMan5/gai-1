from __future__ import annotations

from typing import Any

import torch


SCALAR_MOE_KEYS = (
    "moe_aux_loss_per_layer",
    "moe_router_z_loss",
    "moe_router_z_loss_per_layer",
    "moe_router_entropy",
    "moe_router_confidence",
    "moe_load_entropy",
    "moe_load_std",
    "moe_load_cv",
    "moe_load_min",
    "moe_load_max",
    "moe_load_imbalance",
    "moe_dead_expert_fraction",
    "moe_dead_experts",
)

VECTOR_MOE_KEYS = (
    "moe_importance",
    "moe_dispatch_load",
    "moe_primary_load",
)


def _to_python(value: Any) -> float | list[float]:
    if torch.is_tensor(value):
        detached = value.detach().float().cpu()
        if detached.ndim == 0:
            return float(detached.item())
        return [float(item) for item in detached.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return float(value)


def flatten_moe_metrics(info: dict[str, Any], prefix: str = "") -> dict[str, float | list[float]]:
    row: dict[str, float | list[float]] = {}
    for key in SCALAR_MOE_KEYS:
        if key in info:
            value = _to_python(info[key])
            if isinstance(value, list):
                continue
            row[f"{prefix}{key}"] = value
    for key in VECTOR_MOE_KEYS:
        if key in info:
            value = _to_python(info[key])
            if isinstance(value, list):
                row[f"{prefix}{key}"] = value
    return row
