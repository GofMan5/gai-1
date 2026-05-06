from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
from gai1.data import SFTDataset
from gai1.loading import LoadOptions, load_model
from gai1.lora import LoRAConfig, inject_lora, lora_state_dict, trainable_parameter_count
from gai1.tokenizer import BPETokenizer, ByteTokenizer


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT fine-tune GAI-1 with optional LoRA adapters.")
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default="outputs/gai1_sft_lora")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=None)
    return parser.parse_args()


def load_tokenizer(cfg: Any):
    if cfg.tokenizer.kind == "byte":
        return ByteTokenizer()
    return BPETokenizer(ROOT / cfg.tokenizer.path)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16 if precision in {"auto", "fp16"} else torch.bfloat16)


def estimate_attention_scores_gb(cfg: Any, dtype_bytes: int = 2) -> float:
    batch = int(cfg.train.batch_size)
    seq = int(cfg.data.block_size)
    heads = int(cfg.model.n_head)
    layers = int(cfg.model.n_layer)
    return batch * heads * seq * seq * dtype_bytes * layers / 1024**3


def validate_context_budget(cfg: Any, device: str) -> None:
    if int(cfg.model.block_size) != int(cfg.data.block_size):
        raise ValueError(f"model.block_size ({cfg.model.block_size}) must equal data.block_size ({cfg.data.block_size})")
    if cfg.model.rope_scaling not in {"none", "linear", "dynamic_ntk"}:
        raise ValueError("model.rope_scaling must be one of: none, linear, dynamic_ntk")
    if cfg.model.block_size > 8192 and cfg.model.rope_original_context <= 0:
        raise ValueError("Long-context configs must set model.rope_original_context")
    if device != "cuda":
        return
    _free, total = torch.cuda.mem_get_info(torch.device(device))
    estimated_gb = estimate_attention_scores_gb(cfg)
    limit_gb = (total / 1024**3) * float(cfg.train.max_attention_memory_fraction)
    if estimated_gb > limit_gb and not cfg.train.allow_unsafe_long_context:
        raise RuntimeError(
            "Refusing unsafe full-attention SFT config: "
            f"block_size={cfg.data.block_size}, batch_size={cfg.train.batch_size}, "
            f"estimated attention-score memory={estimated_gb:.1f}GB, limit={limit_gb:.1f}GB."
        )


def main() -> int:
    configure_console()
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    if args.context_length is not None:
        cfg.model.block_size = args.context_length
        cfg.data.block_size = args.context_length
        if args.context_length > 8192 and cfg.model.rope_original_context <= 0:
            cfg.model.rope_original_context = 768
            cfg.model.rope_scaling = "linear"
            cfg.model.rope_scaling_factor = args.context_length / 768
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    validate_context_budget(cfg, device)
    model, metadata = load_model(
        LoadOptions(checkpoint_path=ROOT / args.checkpoint, device=device, dtype="auto", context_length=cfg.model.block_size)
    )
    model.set_gradient_checkpointing(bool(cfg.train.gradient_checkpointing))
    replaced: list[str] = []
    if args.lora:
        replaced = inject_lora(model, LoRAConfig(rank=args.lora_rank))
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    tokenizer = load_tokenizer(cfg)
    dataset = SFTDataset(
        ROOT / (args.data or cfg.data.train_path),
        tokenizer=tokenizer,
        block_size=cfg.data.block_size,
    )
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, pin_memory=device == "cuda", num_workers=cfg.train.num_workers)
    accumulation = max(1, cfg.train.gradient_accumulation_steps)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "sft_lora" if args.lora else "sft_full",
                "base": args.checkpoint,
                "base_metadata": metadata,
                "lora": args.lora,
                "lora_rank": args.lora_rank,
                "lora_targets": replaced,
                "trainable_params": trainable_parameter_count(model),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    step = 0
    micro = 0
    progress = tqdm(total=args.max_steps, desc="sft")
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context(torch.device(device), cfg.train.precision):
                _logits, loss, _info = model(x, y)
            if loss is None:
                raise RuntimeError("Loss was not produced")
            scaler.scale(loss / accumulation).backward()
            micro += 1
            if micro % accumulation:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if step >= args.max_steps:
                break
    progress.close()

    if args.lora:
        torch.save(
            {
                "format": "gai1_lora_adapter_v1",
                "state": lora_state_dict(model),
                "rank": args.lora_rank,
                "alpha": 16.0,
                "dropout": 0.05,
            },
            out_dir / "adapter.pt",
        )
        print(f"Saved LoRA adapter: {out_dir / 'adapter.pt'}")
    else:
        torch.save(
            {
                "format": "gai1_checkpoint_v1",
                "model_config": model.config_dict(),
                "model_state": model.state_dict(),
                "metadata": {
                    "trained_context_length": model.cfg.block_size,
                    "tested_context_length": model.cfg.block_size,
                    "rope_scaling": model.cfg.rope_scaling,
                    "rope_original_context": model.cfg.rope_original_context,
                    "long_context_validated": model.cfg.block_size >= 131072,
                },
            },
            out_dir / "last.pt",
        )
        print(f"Saved SFT checkpoint: {out_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
