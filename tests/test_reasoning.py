from __future__ import annotations

import pytest

from gai1.reasoning import GAIReasoningRuntime, get_reasoning_profile, load_reasoning_profiles


def test_reasoning_profiles_ordered_by_budget() -> None:
    low = get_reasoning_profile("low")
    medium = get_reasoning_profile("medium")
    high = get_reasoning_profile("high")
    max_profile = get_reasoning_profile("max")
    assert low.private_token_budget < medium.private_token_budget < high.private_token_budget < max_profile.private_token_budget
    assert low.draft_count <= medium.draft_count <= high.draft_count <= max_profile.draft_count


def test_reasoning_runtime_level_trace() -> None:
    runtime = GAIReasoningRuntime(level="high")
    trace = runtime.run("Объясни архитектуру GAI-1")
    assert trace.level == "high"
    assert trace.profile["critic_passes"] == 2
    assert trace.drafts
    assert trace.final


def test_unknown_reasoning_level_rejected() -> None:
    with pytest.raises(ValueError):
        get_reasoning_profile("ultra")


def test_load_custom_reasoning_profile(tmp_path) -> None:
    config = tmp_path / "reasoning.json"
    config.write_text(
        """
{
  "levels": {
    "extreme": {
      "planning_depth": 5,
      "draft_count": 4,
      "critic_passes": 4,
      "verifier_passes": 4,
      "rollback_limit": 4,
      "tool_budget": 10,
      "private_token_budget": 8192,
      "self_consistency": true,
      "temperature": 0.9
    }
  }
}
""",
        encoding="utf-8",
    )
    profiles = load_reasoning_profiles(config)
    runtime = GAIReasoningRuntime(level="extreme", profiles=profiles)
    assert runtime.profile.private_token_budget == 8192


def test_reasoning_runtime_uses_generator_for_drafts() -> None:
    class DummyGenerator:
        def complete(self, prompt: str) -> str:
            assert "Reasoning level: medium" in prompt
            return "Сгенерированный моделью черновик с нормальной структурой."

    runtime = GAIReasoningRuntime(generator=DummyGenerator(), level="medium")
    trace = runtime.run("Проверь генератор")

    assert trace.drafts == ["Сгенерированный моделью черновик с нормальной структурой."]
    assert trace.final.startswith("Сгенерированный моделью")
