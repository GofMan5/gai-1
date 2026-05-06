from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.quantization import quantize_state_dict


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a GAI-1 checkpoint to int8/int4 quantized storage.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--quantize-norm", action="store_true", help="Also quantize norm/bias-like tensors.")
    return parser.parse_args()


def fmt_mb(value: int) -> str:
    return f"{value / 1024**2:.2f}MB"


def main() -> int:
    configure_console()
    args = parse_args()
    checkpoint_path = ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != "gai1_checkpoint_v1":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format')}")

    records, stats = quantize_state_dict(
        checkpoint["model_state"],
        bits=args.bits,
        keep_norm_fp16=not args.quantize_norm,
    )
    output = Path(args.out) if args.out else ROOT / "outputs" / "quantized" / f"{checkpoint_path.stem}_int{args.bits}.pt"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "gai1_quantized_checkpoint_v1",
            "source_checkpoint": str(checkpoint_path),
            "bits": args.bits,
            "model_config": checkpoint["model_config"],
            "model_state_quantized": records,
            "stats": {
                "original_bytes": stats.original_bytes,
                "quantized_bytes_estimate": stats.quantized_bytes,
                "compression_ratio_estimate": stats.compression_ratio,
                "quantized_tensors": stats.quantized_tensors,
                "kept_tensors": stats.kept_tensors,
            },
        },
        output,
    )
    actual = output.stat().st_size
    print(f"Saved quantized checkpoint: {output}")
    print(f"source_file={fmt_mb(checkpoint_path.stat().st_size)}")
    print(f"tensor_payload_estimate={fmt_mb(stats.quantized_bytes)} ratio={stats.compression_ratio:.2f}x")
    print(f"actual_file={fmt_mb(actual)} bits={args.bits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

