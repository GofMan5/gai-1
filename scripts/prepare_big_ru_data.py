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
    parser = argparse.ArgumentParser(description="Prepare larger Russian pretrain, chat SFT, and reasoning SFT data.")
    parser.add_argument("--fineweb-docs", type=int, default=50000)
    parser.add_argument("--sft-records", type=int, default=30000)
    parser.add_argument("--reasoning-records", type=int, default=0, help="0 means all available SFT records.")
    parser.add_argument("--reasoning-level", default="high")
    parser.add_argument("--reasoning-style", choices=("visible", "controller"), default="visible")
    parser.add_argument("--pretrain-out", default="data/raw/fineweb2_ru_large.jsonl")
    parser.add_argument("--sft-out", default="data/raw/ru_turbo_alpaca_large.jsonl")
    parser.add_argument("--reasoning-out", default="data/sft/reasoning_ru.jsonl")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    configure_console()
    args = parse_args()
    pretrain_path = ROOT / args.pretrain_out
    sft_path = ROOT / args.sft_out
    reasoning_path = ROOT / args.reasoning_out

    if jsonl_count(pretrain_path) < args.fineweb_docs:
        run(
            [
                sys.executable,
                "scripts/download_fineweb2_ru.py",
                "--out",
                args.pretrain_out,
                "--max-docs",
                str(args.fineweb_docs),
            ]
        )
    else:
        print(f"skip pretrain download: {pretrain_path} already has {jsonl_count(pretrain_path)} records")

    if jsonl_count(sft_path) < args.sft_records:
        run(
            [
                sys.executable,
                "scripts/download_ru_turbo_alpaca.py",
                "--out",
                args.sft_out,
                "--max-records",
                str(args.sft_records),
            ]
        )
    else:
        print(f"skip SFT download: {sft_path} already has {jsonl_count(sft_path)} records")

    build_cmd = [
        sys.executable,
        "scripts/build_reasoning_sft.py",
        "--input",
        args.sft_out,
        "--output",
        args.reasoning_out,
        "--style",
        args.reasoning_style,
        "--level",
        args.reasoning_level,
    ]
    if args.reasoning_records > 0:
        build_cmd.extend(["--max-records", str(args.reasoning_records)])
    run(build_cmd)

    write_manifest(
        ROOT / "data" / "sft" / "big_ru_data_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pretrain": str(pretrain_path.relative_to(ROOT)),
            "pretrain_records": jsonl_count(pretrain_path),
            "sft": str(sft_path.relative_to(ROOT)),
            "sft_records": jsonl_count(sft_path),
            "reasoning": str(reasoning_path.relative_to(ROOT)),
            "reasoning_records": jsonl_count(reasoning_path),
            "sources": [
                "HuggingFaceFW/fineweb-2 rus_Cyrl",
                "IlyaGusev/ru_turbo_alpaca",
            ],
            "reasoning_level": args.reasoning_level,
            "reasoning_style": args.reasoning_style,
        },
    )
    print("Prepared larger RU data pack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
