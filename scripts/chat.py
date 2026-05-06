from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
from gai1.data import format_chat_prompt
from gai1.loading import LoadOptions, load_model
from gai1.tokenizer import BPETokenizer, ByteTokenizer


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a local GAI-1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--config", default="configs/train_gpu.json")
    return parser.parse_args()


def load_chat_tokenizer(config_path: str):
    cfg = load_config(ROOT / config_path)
    if cfg.tokenizer.kind == "byte":
        return ByteTokenizer()
    return BPETokenizer(ROOT / cfg.tokenizer.path)


def main() -> int:
    configure_console()
    args = parse_args()
    adapter_path = ROOT / args.adapter if args.adapter else None
    model, metadata = load_model(
        LoadOptions(
            checkpoint_path=ROOT / args.checkpoint,
            device=args.device,
            dtype=args.dtype,
            adapter_path=adapter_path,
        )
    )

    tokenizer = load_chat_tokenizer(args.config)
    tokens = tokenizer.encode(format_chat_prompt(args.prompt), add_bos=True)
    idx = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(f"[loaded {metadata['format']} dtype={metadata['dtype']} quant={metadata['quantization']} device={metadata['device']}]")
    print(tokenizer.decode(out[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
