from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "scripts", "tests", "configs", "evals")
SKIP_NAMES = {"test_encoding_integrity.py"}
MOJIBAKE_MARKERS = (chr(0x00D0), chr(0x00D1), chr(0x00E2), chr(0xFFFD))


def test_source_text_has_no_mojibake_markers() -> None:
    offenders: list[str] = []
    for dirname in SCAN_DIRS:
        for path in (ROOT / dirname).rglob("*"):
            if not path.is_file() or path.name in SKIP_NAMES or path.suffix.lower() not in {".py", ".json", ".md", ".txt", ".bat"}:
                continue
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in MOJIBAKE_MARKERS):
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []
