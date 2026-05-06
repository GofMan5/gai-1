from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.data import ASSISTANT_LABEL, USER_LABEL
from gai1.reasoning import GAIReasoningRuntime, load_reasoning_profiles


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Russian reasoning-SFT dataset from instruction JSONL.")
    parser.add_argument("--input", default="data/raw/ru_turbo_alpaca_sample.jsonl")
    parser.add_argument("--output", default="data/sft/reasoning_ru.jsonl")
    parser.add_argument("--profiles", default="configs/reasoning_modes.json")
    parser.add_argument("--level", default="high")
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records.")
    parser.add_argument("--style", choices=("visible", "controller"), default="visible")
    return parser.parse_args()


def read_prompt_response(row: dict[str, object]) -> tuple[str, str] | None:
    if "prompt" in row and "response" in row:
        return str(row["prompt"]).strip(), str(row["response"]).strip()
    if "instruction" in row and "output" in row:
        prompt = str(row["instruction"]).strip()
        extra = str(row.get("input", "")).strip()
        if extra:
            prompt = f"{prompt}\n{extra}"
        return prompt, str(row["output"]).strip()
    if "messages" in row and isinstance(row["messages"], list):
        messages = [message for message in row["messages"] if isinstance(message, dict)]
        for idx in range(len(messages) - 1, -1, -1):
            if str(messages[idx].get("role", "")) != "assistant":
                continue
            prompt_parts: list[str] = []
            for message in messages[:idx]:
                role = str(message.get("role", "user"))
                label = ASSISTANT_LABEL if role == "assistant" else USER_LABEL
                prompt_parts.append(f"{label}: {message.get('content', '')}")
            return "\n".join(prompt_parts).strip(), str(messages[idx].get("content", "")).strip()
    return None


def format_visible_response(plan: list[str], verifier: list[str], answer: str) -> str:
    plan_text = "\n".join(f"{idx}. {item}" for idx, item in enumerate(plan, start=1))
    verifier_text = ", ".join(verifier[:3]) if verifier else "skipped"
    return (
        "Краткое рассуждение:\n"
        f"{plan_text}\n\n"
        f"Проверка: {verifier_text}\n\n"
        "Ответ:\n"
        f"{answer.strip()}"
    )


def format_controller_prompt(prompt: str, plan: list[str], level: str) -> str:
    plan_text = "\n".join(f"- {item}" for item in plan)
    return (
        f"Reasoning effort: {level}\n"
        "Используй внутренний controller plan, но в ответе не раскрывай скрытую цепочку рассуждений.\n"
        f"{plan_text}\n\n"
        f"{prompt.strip()}"
    )


def main() -> int:
    configure_console()
    args = parse_args()
    input_path = ROOT / args.input
    output_path = ROOT / args.output
    profiles = load_reasoning_profiles(ROOT / args.profiles)
    runtime = GAIReasoningRuntime(level=args.level, profiles=profiles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_no, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {input_path} line {line_no}")
            pair = read_prompt_response(row)
            if pair is None:
                continue
            prompt, answer = pair
            if not prompt or not answer:
                continue
            trace = runtime.run(prompt, level=args.level)
            if args.style == "visible":
                out_prompt = prompt
                out_response = format_visible_response(trace.plan, trace.verifier_results, answer)
            else:
                out_prompt = format_controller_prompt(prompt, trace.plan, args.level)
                out_response = answer
            target.write(
                json.dumps(
                    {
                        "prompt": out_prompt,
                        "response": out_response,
                        "metadata": {
                            "source": args.input,
                            "source_line": line_no,
                            "reasoning_level": trace.level,
                            "reasoning_style": args.style,
                            "verifier": trace.verifier_results,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            seen += 1
            if args.max_records and seen >= args.max_records:
                break
    print(f"Built reasoning SFT dataset: {output_path} records={written} style={args.style} level={args.level}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
