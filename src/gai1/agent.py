from __future__ import annotations

from dataclasses import dataclass

from gai1.reasoning import GAIReasoningRuntime, ReasoningTrace
from gai1.verifier import BasicVerifier, VerificationResult


@dataclass
class AgentResult:
    trace: ReasoningTrace
    verification: VerificationResult
    final: str


class GAI1Agent:
    def __init__(self, reasoning: GAIReasoningRuntime | None = None, verifier: BasicVerifier | None = None) -> None:
        self.reasoning = reasoning or GAIReasoningRuntime(level="medium")
        self.verifier = verifier or BasicVerifier()

    def run(self, task: str, level: str = "medium") -> AgentResult:
        trace = self.reasoning.run(task, level=level)
        verification = self.verifier.verify_text(trace.final)
        final = trace.final
        if not verification.passed:
            final += "\n\nVerifier: output needs review: " + ", ".join(verification.issues)
        return AgentResult(trace=trace, verification=verification, final=final)

