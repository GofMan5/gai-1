from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    rope_base: float = 10000.0
    use_moe: bool = False
    n_experts: int = 4
    n_experts_per_token: int = 2
    moe_aux_loss_weight: float = 0.01


@dataclass
class DataConfig:
    train_path: str = "data/raw/sample_ru_chat.jsonl"
    field: str = "text"
    block_size: int = 256


@dataclass
class TokenizerConfig:
    kind: str = "bpe"
    path: str = "data/tokenizer/gai1_tokenizer.json"
    vocab_size: int = 32000
    min_frequency: int = 2
    byte_fallback: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 100
    output_dir: str = "outputs/gai1_tiny"
    resume: bool = True
    save_optimizer_state: bool = False
    checkpoint_dtype: str = "fp16"
    device: str = "auto"
    precision: str = "auto"
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    allow_tf32: bool = True
    matmul_precision: str = "high"
    compile: bool = False


@dataclass
class GAIConfig:
    run_name: str = "gai1"
    seed: int = 1337
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _dataclass_from_dict(cls: type[Any], values: dict[str, Any]) -> Any:
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_config(path: str | Path) -> GAIConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return GAIConfig(
        run_name=raw.get("run_name", "gai1"),
        seed=int(raw.get("seed", 1337)),
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
        tokenizer=_dataclass_from_dict(TokenizerConfig, raw.get("tokenizer", {})),
        data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
        train=_dataclass_from_dict(TrainConfig, raw.get("train", {})),
    )


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
