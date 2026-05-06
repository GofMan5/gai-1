from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
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
    parser.add_argument("--quantize-router", action="store_true", help="Allow INT4 quantization for MoE router/gate tensors.")
    parser.add_argument("--no-tie-dedupe", action="store_true", help="Store lm_head.weight separately instead of aliasing token embeddings.")
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--tokenizer", default=None)
    return parser.parse_args()


def fmt_mb(value: int) -> str:
    return f"{value / 1024**2:.2f}MB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_tokenizer_artifact(args: argparse.Namespace, cfg) -> dict[str, object] | None:
    tokenizer_path = ROOT / (args.tokenizer or cfg.tokenizer.path)
    if not tokenizer_path.exists():
        return None
    data = tokenizer_path.read_bytes()
    return {
        "path": tokenizer_path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "kind": cfg.tokenizer.kind,
        "vocab_size": cfg.tokenizer.vocab_size,
        "json": json.loads(data.decode("utf-8")),
    }


def main() -> int:
    configure_console()
    args = parse_args()
    checkpoint_path = ROOT / args.checkpoint
    cfg = load_config(ROOT / args.config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != "gai1_checkpoint_v1":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format')}")

    records, stats = quantize_state_dict(
        checkpoint["model_state"],
        bits=args.bits,
        keep_norm_fp16=not args.quantize_norm,
        keep_router_fp16=not args.quantize_router,
        tie_lm_head=not args.no_tie_dedupe,
    )
    output = Path(args.out) if args.out else ROOT / "outputs" / "quantized" / f"{checkpoint_path.stem}_int{args.bits}.pt"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    source_rel = checkpoint_path.relative_to(ROOT).as_posix() if checkpoint_path.is_relative_to(ROOT) else checkpoint_path.name
    metadata = checkpoint.get("metadata", {})
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    artifact_metadata = {
        "source_checkpoint": source_rel,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_metadata": metadata,
        "run_manifest": read_json_if_exists(manifest_path),
        "generation_config": {
            "max_new_tokens": 120,
            "temperature": 0.8,
            "top_k": 50,
            "chat_template": "Пользователь: {prompt}\\nАссистент:",
        },
        "quantization_policy": {
            "bits": args.bits,
            "keep_norm_fp16": not args.quantize_norm,
            "keep_router_fp16": not args.quantize_router,
            "tie_lm_head_to_token_embedding": not args.no_tie_dedupe,
            "scheme": "symmetric_per_tensor_storage",
        },
    }
    torch.save(
        {
            "format": "gai1_quantized_checkpoint_v2",
            "bits": args.bits,
            "model_config": checkpoint["model_config"],
            "model_state_quantized": records,
            "tokenizer": load_tokenizer_artifact(args, cfg),
            "metadata": artifact_metadata,
            "stats": {
                "original_bytes": stats.original_bytes,
                "quantized_bytes_estimate": stats.quantized_bytes,
                "compression_ratio_estimate": stats.compression_ratio,
                "quantized_tensors": stats.quantized_tensors,
                "kept_tensors": stats.kept_tensors,
                "aliased_tensors": stats.aliased_tensors,
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
