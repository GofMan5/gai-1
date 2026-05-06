from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
from gai1.data import read_jsonl_formatted, read_jsonl_texts
from gai1.tokenizer import train_bpe_tokenizer


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RU-first GAI-1 BPE tokenizer.")
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    configure_console()
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    data_path = ROOT / (args.data or cfg.data.train_path)
    out = ROOT / (args.out or cfg.tokenizer.path)
    vocab_size = args.vocab_size or cfg.tokenizer.vocab_size
    try:
        texts = read_jsonl_formatted(data_path)
    except ValueError:
        texts = read_jsonl_texts(data_path, field=cfg.data.field)
    train_bpe_tokenizer(texts=texts, output_path=out, vocab_size=vocab_size, min_frequency=cfg.tokenizer.min_frequency)
    print(f"Saved tokenizer: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
