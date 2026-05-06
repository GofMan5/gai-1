from __future__ import annotations

import json
from pathlib import Path

from gai1.config import TrainConfig


def test_train_config_has_fused_optimizer_default() -> None:
    cfg = TrainConfig()
    assert cfg.fused_optimizer is True


def test_autotune_config_writer_sets_batch_and_accumulation(tmp_path) -> None:
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gai1_autotune_for_tests", root / "scripts" / "autotune_rtx3060.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    base = {
        "data": {"train_path": "old.jsonl"},
        "train": {"batch_size": 1, "gradient_accumulation_steps": 1, "output_dir": "outputs/old"},
    }
    out = tmp_path / "config.json"
    module.write_config(base, out, batch_size=4, target_effective_batch=32, dataset="data/raw/x.jsonl")
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["data"]["streaming"] is True
    assert payload["train"]["batch_size"] == 4
    assert payload["train"]["gradient_accumulation_steps"] == 8
    assert payload["train"]["fused_optimizer"] is True
