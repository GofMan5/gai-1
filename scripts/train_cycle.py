from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an incremental GAI-1 training cycle and fixed progress prompts.")
    parser.add_argument("--stage", choices=("pretrain", "sft"), default="sft")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--checkpoint", default="outputs/gai1_train_gpu/last.pt")
    parser.add_argument("--sft-data", default="data/raw/ru_turbo_alpaca_sample.jsonl")
    parser.add_argument("--sft-out", default="outputs/gai1_sft_lora")
    parser.add_argument("--prompts", default="evals/progress_prompts_ru.txt")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def checkpoint_step(path: Path) -> int:
    if not path.exists():
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    return int(checkpoint.get("step", 0))


def train_pretrain(args: argparse.Namespace) -> Path:
    cfg = load_config(ROOT / args.config)
    checkpoint = ROOT / cfg.train.output_dir / "last.pt"
    current = checkpoint_step(checkpoint)
    target = current + args.steps
    run([sys.executable, "scripts/train_pretrain.py", "--config", args.config, "--max-steps", str(target)])
    return checkpoint


def train_sft(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = ROOT / args.checkpoint
    adapter = ROOT / args.sft_out / "adapter.pt"
    cmd = [
        sys.executable,
        "scripts/train_sft.py",
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--data",
        args.sft_data,
        "--out",
        args.sft_out,
        "--lora",
        "--max-steps",
        str(args.steps),
    ]
    if adapter.exists():
        cmd.extend(["--resume-adapter", str(adapter.relative_to(ROOT))])
    run(cmd)
    return checkpoint, adapter


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_prompts(args: argparse.Namespace, checkpoint: Path, adapter: Path | None) -> Path:
    prompts = load_prompts(ROOT / args.prompts)
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"progress_{args.stage}_{stamp}.jsonl"
    with report_path.open("w", encoding="utf-8", newline="\n") as fh:
        for prompt in prompts:
            cmd = [
                sys.executable,
                "scripts/chat.py",
                "--checkpoint",
                str(checkpoint.relative_to(ROOT)),
                "--prompt",
                prompt,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
            ]
            if adapter is not None and adapter.exists():
                cmd.extend(["--adapter", str(adapter.relative_to(ROOT))])
            print("\n=== PROMPT ===")
            print(prompt)
            result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(result.stdout.strip())
            fh.write(json.dumps({"prompt": prompt, "output": result.stdout, "adapter": str(adapter) if adapter else None}, ensure_ascii=False) + "\n")
    print(f"Progress report: {report_path}")
    return report_path


def main() -> int:
    configure_console()
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.stage == "pretrain":
        checkpoint = train_pretrain(args)
        adapter = None
    else:
        checkpoint, adapter = train_sft(args)
    if not args.skip_eval:
        eval_prompts(args, checkpoint, adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
