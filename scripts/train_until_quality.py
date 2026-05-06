from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged GAI-1 training toward usable RU chat/reasoning.")
    parser.add_argument("--target-pretrain-step", type=int, default=4070)
    parser.add_argument("--pretrain-cycle", type=int, default=100)
    parser.add_argument("--chat-steps", type=int, default=1500)
    parser.add_argument("--reasoning-steps", type=int, default=1000)
    parser.add_argument("--fineweb-docs", type=int, default=50000)
    parser.add_argument("--sft-records", type=int, default=30000)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--skip-reasoning", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def checkpoint_step(path: Path) -> int:
    if not path.exists():
        return 0
    payload = torch.load(path, map_location="cpu")
    return int(payload.get("step", 0))


def write_report(row: dict[str, object]) -> None:
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "train_until_quality.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_large_config() -> str:
    source = ROOT / "configs" / "train_gpu.json"
    target = ROOT / "configs" / "train_gpu_large.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["data"]["train_path"] = "data/raw/fineweb2_ru_large.jsonl"
    payload["data"]["streaming"] = True
    payload["train"]["resume"] = True
    payload["train"]["fused_optimizer"] = True
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target.relative_to(ROOT))


def main() -> int:
    configure_console()
    args = parse_args()
    started = datetime.now(timezone.utc).isoformat()
    write_report({"event": "start", "created_at": started, "args": vars(args)})

    if not args.skip_data:
        run(
            [
                sys.executable,
                "scripts/prepare_big_ru_data.py",
                "--fineweb-docs",
                str(args.fineweb_docs),
                "--sft-records",
                str(args.sft_records),
            ]
        )

    checkpoint = ROOT / "outputs/gai1_train_gpu/last.pt"
    pretrain_config = write_large_config()
    if not args.skip_pretrain:
        while checkpoint_step(checkpoint) < args.target_pretrain_step:
            current = checkpoint_step(checkpoint)
            next_target = min(args.target_pretrain_step, current + args.pretrain_cycle)
            run([sys.executable, "scripts/train_pretrain.py", "--config", pretrain_config, "--max-steps", str(next_target)])
            write_report(
                {
                    "event": "pretrain_cycle",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "from_step": current,
                    "to_step": checkpoint_step(checkpoint),
                    "target": args.target_pretrain_step,
                }
            )

    if not args.skip_chat:
        run(
            [
                sys.executable,
                "scripts/train_sft.py",
                "--checkpoint",
                "outputs/gai1_train_gpu/last.pt",
                "--data",
                "data/raw/ru_turbo_alpaca_large.jsonl",
                "--out",
                "outputs/gai1_sft_lora",
                "--lora",
                "--max-steps",
                str(args.chat_steps),
            ]
        )

    if not args.skip_reasoning:
        run(
            [
                sys.executable,
                "scripts/train_sft.py",
                "--checkpoint",
                "outputs/gai1_train_gpu/last.pt",
                "--data",
                "data/sft/reasoning_ru.jsonl",
                "--out",
                "outputs/gai1_reasoning_lora",
                "--lora",
                "--max-steps",
                str(args.reasoning_steps),
            ]
        )

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            "scripts/eval_gates.py",
            "--checkpoint",
            "outputs/gai1_train_gpu/last.pt",
            "--adapter",
            "outputs/gai1_reasoning_lora/adapter.pt",
            "--data",
            "data/raw/fineweb2_ru_large.jsonl",
        ]
        result = run_capture(eval_cmd)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        write_report(
            {
                "event": "eval_gates",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "returncode": result.returncode,
                "stdout": result.stdout[-8000:],
            }
        )
        if result.returncode != 0:
            print("Eval gates did not pass yet. Keep training; artifacts were not promoted as release-ready.")

    write_report({"event": "done", "created_at": datetime.now(timezone.utc).isoformat()})
    print("Training pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
