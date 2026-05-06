from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from gai1.reasoning import GAIReasoningRuntime


ROOT = Path(__file__).resolve().parents[1]


def load_tui_module():
    spec = importlib.util.spec_from_file_location("gai1_tui_for_tests", ROOT / "scripts" / "tui.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tui_prompt_includes_reasoning_controller() -> None:
    tui = load_tui_module()
    trace = GAIReasoningRuntime(level="high").run("Проверь план ответа")
    prompt = tui.format_model_prompt([], "Проверь план ответа", 4, trace)

    assert "Internal reasoning controller:" in prompt
    assert "- effort: high" in prompt
    assert "Определить тип задачи" in prompt
    assert f"{tui.ASSISTANT_LABEL}:" in prompt


def test_tui_reasoning_view_command_changes_view_only() -> None:
    tui = load_tui_module()
    handled, should_exit, level, view = tui.handle_command(
        "/reasoning compact",
        tui.Console(),
        GAIReasoningRuntime(level="medium"),
        [],
        {},
        tui.TurnStats(),
        "medium",
        "full",
        tui.argparse.Namespace(),
    )

    assert handled is True
    assert should_exit is False
    assert level == "medium"
    assert view == "compact"
