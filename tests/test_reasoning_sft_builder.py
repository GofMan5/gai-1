from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_builder_module():
    spec = importlib.util.spec_from_file_location("gai1_reasoning_sft_builder_for_tests", ROOT / "scripts" / "build_reasoning_sft.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reasoning_sft_visible_response_contains_trace() -> None:
    builder = load_builder_module()
    response = builder.format_visible_response(
        ["Определить цель.", "Проверить ответ."],
        ["pass:0:basic_consistency"],
        "Итоговый ответ.",
    )

    assert "Краткое рассуждение:" in response
    assert "1. Определить цель." in response
    assert "Проверка: pass:0:basic_consistency" in response
    assert response.endswith("Итоговый ответ.")


def test_reasoning_sft_prompt_response_reader_supports_instruction() -> None:
    builder = load_builder_module()
    prompt, response = builder.read_prompt_response({"instruction": "Сделай план", "input": "для модели", "output": "Готово"})

    assert prompt == "Сделай план\nдля модели"
    assert response == "Готово"


def test_reasoning_sft_output_is_valid_jsonl(tmp_path) -> None:
    builder = load_builder_module()
    source = tmp_path / "input.jsonl"
    target = tmp_path / "out.jsonl"
    source.write_text(
        json.dumps({"instruction": "Объясни токенизацию", "output": "Токенизация делит текст."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_reasoning_sft.py",
            "--input",
            str(source),
            "--output",
            str(target),
            "--max-records",
            "1",
        ]
        assert builder.main() == 0
    finally:
        sys.argv = old_argv

    row = json.loads(target.read_text(encoding="utf-8").strip())
    assert row["prompt"] == "Объясни токенизацию"
    assert "Краткое рассуждение:" in row["response"]
    assert row["metadata"]["reasoning_level"] == "high"
