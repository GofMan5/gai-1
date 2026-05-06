from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a local Russian sample from HuggingFaceFW/fineweb-2.")
    parser.add_argument("--out", default="data/raw/fineweb2_ru_sample.jsonl")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    parser.add_argument("--config", default="rus_Cyrl")
    return parser.parse_args()


def clean_text(text: str, max_chars: int) -> str:
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    compact = "\n".join(lines)
    if len(compact) > max_chars:
        compact = compact[:max_chars].rsplit(" ", 1)[0]
    return compact.strip()


def main() -> int:
    configure_console()
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: .\\.venv\\Scripts\\python.exe -m pip install datasets") from exc

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(args.dataset, args.config, split="train", streaming=True)
    saved = 0
    seen = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in stream:
            seen += 1
            text = clean_text(str(row.get("text", "")), args.max_chars)
            if len(text) < args.min_chars:
                continue
            fh.write(
                json.dumps(
                    {
                        "text": text,
                        "source": args.dataset,
                        "subset": args.config,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            fh.write("\n")
            saved += 1
            if saved % 500 == 0:
                print(f"saved={saved} seen={seen}", flush=True)
            if saved >= args.max_docs:
                break
    print(f"Saved {saved} Russian FineWeb2 docs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
