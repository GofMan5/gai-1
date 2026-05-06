from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    checks: list[str]
    issues: list[str]


class BasicVerifier:
    def verify_text(self, text: str) -> VerificationResult:
        issues: list[str] = []
        checks = ["non_empty", "not_repeated", "has_sentence_end"]
        stripped = text.strip()
        if not stripped:
            issues.append("empty_output")
        if len(stripped) > 40:
            chunks = [stripped[i : i + 20] for i in range(0, min(len(stripped), 200), 20)]
            if len(chunks) != len(set(chunks)):
                issues.append("repetitive_output")
        if stripped and not re.search(r"[.!?。]|$", stripped):
            issues.append("no_sentence_end")
        return VerificationResult(passed=not issues, checks=checks, issues=issues)

    def verify_math(self, prompt: str, answer: str) -> VerificationResult:
        checks = ["basic_arithmetic_pattern"]
        issues: list[str] = []
        match = re.search(r"(\d+)\s*([+\-*])\s*(\d+)", prompt)
        if match:
            left = int(match.group(1))
            right = int(match.group(3))
            op = match.group(2)
            expected = {"+": left + right, "-": left - right, "*": left * right}[op]
            if str(expected) not in answer:
                issues.append(f"expected_{expected}_not_found")
        return VerificationResult(passed=not issues, checks=checks, issues=issues)

