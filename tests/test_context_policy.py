from __future__ import annotations

from pathlib import Path

from gai1.config import GAI1_CONTEXT_EXTENSION_STAGES, GAI1_TARGET_CONTEXT_LENGTH, resolve_context_length, validate_context_length


ROOT = Path(__file__).resolve().parents[1]


def test_project_context_target_is_fixed_at_256k() -> None:
    assert GAI1_TARGET_CONTEXT_LENGTH == 262_144
    assert GAI1_CONTEXT_EXTENSION_STAGES[-1] == GAI1_TARGET_CONTEXT_LENGTH
    assert resolve_context_length(None) == GAI1_TARGET_CONTEXT_LENGTH


def test_context_length_cannot_exceed_project_target() -> None:
    assert validate_context_length(32_768) == 32_768
    try:
        validate_context_length(GAI1_TARGET_CONTEXT_LENGTH + 1)
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("context above target must fail")


def test_legacy_128k_context_artifacts_do_not_return() -> None:
    assert not (ROOT / "configs" / "train_128k_experimental.json").exists()
    offenders: list[str] = []
    for path in [
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "MODEL_CARD.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "configs").glob("*.json"),
        *(ROOT / "scripts").glob("*.py"),
    ]:
        text = path.read_text(encoding="utf-8")
        if "128k" in text.casefold() or "131072" in text:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
