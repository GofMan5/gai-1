from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
    text = "Это нормальный русский ответ без лишних повторов."

    assert gates.cyrillic_ratio(text) > 0.8
    assert gates.repetition_ratio(text) < 0.2
    assert gates.prompt_echo_ratio("Зачем нужен tokenizer?", text) < 0.5
    assert gates.has_mojibake(text) is False
    assert gates.has_mojibake("ÐŸÑ€Ð¸Ð²ÐµÑ‚") is True
