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
    return parser.parse_args()


def load_tokenizer(cfg: Any):
    if cfg.tokenizer.kind == "byte":
        return ByteTokenizer()
    return BPETokenizer(ROOT / cfg.tokenizer.path)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16 if precision in {"auto", "fp16"} else torch.bfloat16)


def main() -> int:
    configure_console()
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_model(LoadOptions(checkpoint_path=ROOT / args.checkpoint, device=device, dtype="auto"))
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
        torch.save({"format": "gai1_checkpoint_v1", "model_config": model.config_dict(), "model_state": model.state_dict()}, out_dir / "last.pt")
        print(f"Saved SFT checkpoint: {out_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
