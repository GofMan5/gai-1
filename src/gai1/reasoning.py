from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


ReasoningLevel = str


class TextGenerator(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class ReasoningProfile:
    level: str
    planning_depth: int
    draft_count: int
    critic_passes: int
    verifier_passes: int
    rollback_limit: int
    tool_budget: int
    private_token_budget: int
    self_consistency: bool
    temperature: float


REASONING_PROFILES: dict[str, ReasoningProfile] = {
    "low": ReasoningProfile("low", 1, 1, 0, 0, 0, 0, 256, False, 0.4),
    "medium": ReasoningProfile("medium", 2, 1, 1, 1, 1, 2, 768, False, 0.6),
    "high": ReasoningProfile("high", 3, 2, 2, 2, 2, 5, 2048, True, 0.7),
    "max": ReasoningProfile("max", 4, 3, 3, 3, 3, 8, 4096, True, 0.8),
}


@dataclass
class ReasoningTrace:
    task: str
    level: ReasoningLevel
    profile: dict[str, object]
    plan: list[str]
    drafts: list[str]
    critiques: list[list[str]]
    verifier_results: list[str]
    rollbacks: int
    final: str

    def to_record(self, include_private: bool = False) -> dict[str, object]:
        public: dict[str, object] = {
            "task": self.task,
            "level": self.level,
            "profile": self.profile,
            "final": self.final,
        }
        if include_private:
            public.update(
                {
                    "plan": self.plan,
                    "drafts": self.drafts,
                    "critiques": self.critiques,
                    "verifier_results": self.verifier_results,
                    "rollbacks": self.rollbacks,
                }
            )
        return public


def profile_from_dict(level: str, raw: dict[str, object]) -> ReasoningProfile:
    return ReasoningProfile(
        level=level,
        planning_depth=int(raw["planning_depth"]),
        draft_count=int(raw["draft_count"]),
        critic_passes=int(raw["critic_passes"]),
        verifier_passes=int(raw["verifier_passes"]),
        rollback_limit=int(raw["rollback_limit"]),
        tool_budget=int(raw["tool_budget"]),
        private_token_budget=int(raw["private_token_budget"]),
        self_consistency=bool(raw["self_consistency"]),
        temperature=float(raw.get("temperature", 0.6)),
    )


def load_reasoning_profiles(path: str | Path) -> dict[str, ReasoningProfile]:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    levels = raw.get("levels")
    if not isinstance(levels, dict):
        raise ValueError(f"Reasoning config must contain a 'levels' object: {path}")
    return {str(level): profile_from_dict(str(level), values) for level, values in levels.items()}


def get_reasoning_profile(level: str, profiles: dict[str, ReasoningProfile] | None = None) -> ReasoningProfile:
    source = profiles or REASONING_PROFILES
    if level not in source:
        available = ", ".join(source)
        raise ValueError(f"Unknown reasoning level '{level}'. Available: {available}")
    return source[level]


class GAIReasoningRuntime:
    """Planner -> draft -> critic -> verifier -> rollback -> final renderer."""

    def __init__(
        self,
        generator: TextGenerator | None = None,
        level: str = "medium",
        profiles: dict[str, ReasoningProfile] | None = None,
    ) -> None:
        self.generator = generator
        self.profiles = profiles or REASONING_PROFILES
        self.profile = get_reasoning_profile(level, self.profiles)

    def set_level(self, level: str) -> None:
        self.profile = get_reasoning_profile(level, self.profiles)

    def plan(self, task: str) -> list[str]:
        base = [
            "Определить тип задачи и критерий успеха.",
            "Выделить факты, ограничения и неизвестные места.",
            "Собрать черновой ответ.",
            "Проверить слабые места и противоречия.",
            "Сжать результат в финальный ответ без лишней воды.",
        ]
        if self.profile.planning_depth >= 3:
            base.insert(2, "Разбить задачу на независимые проверки.")
        if self.profile.planning_depth >= 4:
            base.insert(3, "Построить альтернативную ветку решения для self-consistency.")
        return base[: max(1, self.profile.planning_depth + 2)]

    def _prompt_for_draft(self, task: str, plan: list[str], draft_index: int) -> str:
        plan_text = "\n".join(f"- {item}" for item in plan)
        return (
            f"Reasoning level: {self.profile.level}\n"
            f"Private token budget: {self.profile.private_token_budget}\n"
            f"Draft index: {draft_index}\n\n"
            f"Задача:\n{task}\n\n"
            f"План:\n{plan_text}\n\n"
            "Собери черновой ответ. Не раскрывай скрытую цепочку рассуждений, верни только полезный черновик:"
        )

    def draft(self, task: str, plan: list[str], draft_index: int = 0) -> str:
        if self.generator is None:
            return (
                f"Черновик {draft_index + 1} [{self.profile.level}] для задачи: {task}\n"
                f"Проверочный план: {' | '.join(plan)}"
            )
        return self.generator.complete(self._prompt_for_draft(task, plan, draft_index))

    def critique(self, draft: str, pass_index: int = 0) -> list[str]:
        issues: list[str] = []
        text = draft.strip()
        lowered = text.casefold()
        if len(text) < 40:
            issues.append("Черновик слишком короткий для выбранного reasoning уровня.")
        if "не знаю" in lowered and self.profile.level in {"high", "max"}:
            issues.append("Нужно отделить реальный пробел знания от слабой формулировки.")
        if self.profile.level in {"high", "max"} and not any(mark in text for mark in (":", ".", "\n")):
            issues.append("Ответ выглядит как сырая фраза без структуры.")
        if pass_index > 0 and issues:
            issues.append(f"Повторная критика #{pass_index + 1}: требуется более строгая проверка.")
        return issues

    def verify(self, draft: str, pass_index: int = 0) -> str:
        if self.profile.verifier_passes == 0:
            return "skipped"
        if len(draft.strip()) < 20:
            return f"fail:{pass_index}:too_short"
        return f"pass:{pass_index}:basic_consistency"

    def render_final(self, draft: str, critiques: list[list[str]], verifier_results: list[str]) -> str:
        final = draft.strip()
        open_issues = [issue for batch in critiques for issue in batch]
        hard_fail = any(result.startswith("fail") for result in verifier_results)
        if hard_fail:
            return final + "\n\nСтатус проверки: есть риск, нужен повторный прогон или внешний verifier."
        if open_issues and self.profile.level in {"high", "max"}:
            return final + "\n\nПроверено: слабые места отмечены во внутреннем trace."
        return final

    def run(self, task: str, level: str | None = None) -> ReasoningTrace:
        if level is not None:
            self.set_level(level)

        plan = self.plan(task)
        drafts: list[str] = []
        critiques: list[list[str]] = []
        verifier_results: list[str] = []
        rollbacks = 0
        best_draft = ""
        best_score: tuple[int, int] | None = None

        for draft_index in range(self.profile.draft_count):
            candidate = self.draft(task, plan, draft_index)
            candidate_critiques: list[str] = []
            for critic_pass in range(self.profile.critic_passes):
                candidate_critiques.extend(self.critique(candidate, critic_pass))
            candidate_verifier = [self.verify(candidate, verifier_pass) for verifier_pass in range(self.profile.verifier_passes)]

            drafts.append(candidate)
            critiques.append(candidate_critiques)
            verifier_results.extend(candidate_verifier)

            failures = sum(1 for result in candidate_verifier if result.startswith("fail"))
            score = (failures, len(candidate_critiques))
            if best_score is None or score < best_score:
                best_score = score
                best_draft = candidate
            if score == (0, 0):
                break
            if rollbacks < self.profile.rollback_limit:
                rollbacks += 1

        final = self.render_final(best_draft, critiques, verifier_results)
        return ReasoningTrace(
            task=task,
            level=self.profile.level,
            profile=asdict(self.profile),
            plan=plan,
            drafts=drafts,
            critiques=critiques,
            verifier_results=verifier_results,
            rollbacks=rollbacks,
            final=final,
        )
