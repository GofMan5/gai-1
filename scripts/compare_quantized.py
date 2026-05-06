from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FP checkpoint and quantized checkpoint on fixed prompts.")
    parser.add_argument("--fp-checkpoint", default="outputs/gai1_train_gpu/last.pt")
    parser.add_argument("--quantized-checkpoint", required=True)
    parser.add_argument("--prompts", default="evals/progress_prompts_ru.txt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate(checkpoint: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
    cmd = [
        sys.executable,
        "scripts/chat.py",
        "--checkpoint",
        checkpoint,
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.stdout


def main() -> int:
    configure_console()
    args = parse_args()
    prompts = load_prompts(ROOT / args.prompts)
    out_path = ROOT / args.out if args.out else ROOT / "reports" / f"compare_quantized_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for prompt in prompts:
            fp_output = generate(args.fp_checkpoint, prompt, args.max_new_tokens, args.temperature)
            q_output = generate(args.quantized_checkpoint, prompt, args.max_new_tokens, args.temperature)
            record = {
                "prompt": prompt,
                "fp_checkpoint": args.fp_checkpoint,
                "quantized_checkpoint": args.quantized_checkpoint,
                "fp_output": fp_output,
                "quantized_output": q_output,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("\n=== PROMPT ===")
            print(prompt)
            print("--- FP ---")
            print(fp_output.strip())
            print("--- QUANTIZED ---")
            print(q_output.strip())
    print(f"Comparison report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
