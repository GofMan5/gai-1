from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
from gai1.data import format_chat_prompt
from gai1.inference import GenerationConfig, generate_text
from gai1.loading import LoadOptions, load_model
from gai1.tokenizer import BPETokenizer, ByteTokenizer, assert_tokenizer_compatible


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
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--full-text", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--rope-scaling", default=None, choices=("none", "linear", "dynamic_ntk"))
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true")
    parser.add_argument("--strict-tokenizer-path", action="store_true")
    return parser.parse_args()


def load_chat_tokenizer(config_path: str):
    cfg = load_config(ROOT / config_path)
    if cfg.tokenizer.kind == "byte":
        return cfg, ByteTokenizer()
    return cfg, BPETokenizer(ROOT / cfg.tokenizer.path)


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
            context_length=args.context_length,
            rope_scaling=args.rope_scaling,
        )
    )

    cfg, tokenizer = load_chat_tokenizer(args.config)
    compatibility = assert_tokenizer_compatible(
        metadata,
        cfg,
        ROOT,
        tokenizer=tokenizer,
        allow_mismatch=args.allow_tokenizer_mismatch,
        strict_path=args.strict_tokenizer_path,
    )
    if compatibility["issues"]:
        print(f"[warning] tokenizer mismatch ignored: {compatibility['issues']}", file=sys.stderr)
    result = generate_text(
        model,
        tokenizer,
        format_chat_prompt(args.prompt),
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_strings=tuple(args.stop),
            return_full_text=args.full_text,
        ),
    )
    print(f"[loaded {metadata['format']} dtype={metadata['dtype']} quant={metadata['quantization']} device={metadata['device']}]")
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
