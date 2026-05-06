from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_eval_module():
    spec = importlib.util.spec_from_file_location("gai1_eval_gates_for_tests", ROOT / "scripts" / "eval_gates.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_eval_text_quality_heuristics() -> None:
    gates = load_eval_module()
    text = "\u042d\u0442\u043e \u043d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u044b\u0439 \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u043e\u0442\u0432\u0435\u0442 \u0431\u0435\u0437 \u043b\u0438\u0448\u043d\u0438\u0445 \u043f\u043e\u0432\u0442\u043e\u0440\u043e\u0432."

    assert gates.cyrillic_ratio(text) > 0.8
    assert gates.repetition_ratio(text) < 0.2
    assert gates.prompt_echo_ratio("\u0417\u0430\u0447\u0435\u043c \u043d\u0443\u0436\u0435\u043d tokenizer?", text) < 0.5
    assert gates.has_mojibake(text) is False
    assert gates.has_mojibake("\u00c3\u0090\u00c2\u009f\u00c3\u0091\u00e2\u0082\u00ac\u00c3\u0090\u00c2\u00b8") is True


def test_eval_gates_fail_closed_without_holdout() -> None:
    gates = load_eval_module()
    args = SimpleNamespace(data=None)
    cfg = SimpleNamespace(data=SimpleNamespace(train_path="data/raw/train.jsonl", val_path=""))

    with pytest.raises(ValueError, match="refusing to fall back"):
        gates.resolve_eval_data(args, cfg)


def test_eval_gates_reject_train_data_as_eval() -> None:
    gates = load_eval_module()
    args = SimpleNamespace(data="data/raw/train.jsonl")
    cfg = SimpleNamespace(data=SimpleNamespace(train_path="data/raw/train.jsonl", val_path="data/raw/val.jsonl"))

    with pytest.raises(ValueError, match="held out"):
        gates.resolve_eval_data(args, cfg)


def test_eval_gates_require_sample_prompts() -> None:
    gates = load_eval_module()

    with pytest.raises(ValueError, match="sample_prompts"):
        gates.validate_gates({"sample_prompts": []})
