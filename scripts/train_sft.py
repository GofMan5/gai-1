from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import random
import sys
import time
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
from gai1.training import flatten_moe_metrics


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
    parser.add_argument("--resume-adapter", default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--sft-stride", type=int, default=None)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_record_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def make_optimizer(parameters, cfg: Any, device: str) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
    }
    if device == "cuda" and bool(getattr(cfg.train, "fused_optimizer", True)):
        try:
            if "fused" in inspect.signature(torch.optim.AdamW).parameters:
                kwargs["fused"] = True
        except (TypeError, ValueError):
            pass
    return torch.optim.AdamW(parameters, **kwargs)


def should_drop_last(dataset_len: int, batch_size: int) -> bool:
    return dataset_len >= batch_size


def learning_rate_at_step(cfg: Any, step: int, max_steps: int | None = None) -> float:
    scheduler = str(getattr(cfg.train, "lr_scheduler", "cosine")).lower()
    base_lr = float(cfg.train.learning_rate)
    min_lr = float(getattr(cfg.train, "min_learning_rate", 0.0))
    warmup_steps = max(0, int(getattr(cfg.train, "warmup_steps", 0)))
    total_steps = max(1, int(max_steps if max_steps is not None else cfg.train.max_steps))
    if scheduler not in {"constant", "cosine"}:
        raise ValueError("train.lr_scheduler must be 'constant' or 'cosine'")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if scheduler == "constant":
        return base_lr
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> float:
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def load_training_state(path: Path, optimizer: torch.optim.Optimizer, scaler: Any, device: str) -> tuple[int, float]:
    if not path.exists():
        return 0, float("inf")
    state = torch.load(path, map_location=device)
    if state.get("format") != "gai1_sft_training_state_v1":
        return 0, float("inf")
    if "optimizer_state" in state:
        try:
            optimizer.load_state_dict(state["optimizer_state"])
        except ValueError as exc:
            print(f"SFT optimizer resume skipped: {exc}")
    if "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if device == "cuda" and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(state.get("step", 0)), float(state.get("best_loss", float("inf")))


def save_training_state(
    path: Path,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    best_loss: float,
) -> None:
    payload: dict[str, Any] = {
        "format": "gai1_sft_training_state_v1",
        "step": step,
        "best_loss": best_loss,
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    atomic_torch_save(payload, path)


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
    checkpoint_path = ROOT / args.checkpoint
    data_path = ROOT / (args.data or cfg.data.train_path)
    resume_adapter = ROOT / args.resume_adapter if args.resume_adapter else None
    if resume_adapter is not None and not resume_adapter.exists():
        resume_adapter = None
    model, metadata = load_model(
        LoadOptions(
            checkpoint_path=checkpoint_path,
            device=device,
            dtype="auto",
            context_length=cfg.model.block_size,
            adapter_path=resume_adapter,
        )
    )
    model.set_gradient_checkpointing(bool(cfg.train.gradient_checkpointing))
    replaced: list[str] = []
    if args.lora and resume_adapter is None:
        replaced = inject_lora(model, LoRAConfig(rank=args.lora_rank))
    elif args.lora:
        replaced = [f"resumed:{resume_adapter}"]
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(trainable_params, cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "training_state.pt"
    previous_step = 0
    best_loss = float("inf")
    if resume_adapter is not None:
        previous_step, best_loss = load_training_state(state_path, optimizer, scaler, device)
    total_target_step = previous_step + int(args.max_steps)
    current_lr = set_optimizer_lr(optimizer, learning_rate_at_step(cfg, previous_step, total_target_step))

    tokenizer = load_tokenizer(cfg)
    dataset = SFTDataset(
        data_path,
        tokenizer=tokenizer,
        block_size=cfg.data.block_size,
        stride=args.sft_stride,
    )
    drop_last = should_drop_last(len(dataset), int(cfg.train.batch_size))
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=device == "cuda",
        num_workers=cfg.train.num_workers,
    )
    accumulation = max(1, cfg.train.gradient_accumulation_steps)
    dataset_records = jsonl_record_count(data_path)
    base_sha256 = file_sha256(checkpoint_path)
    dataset_sha256 = file_sha256(data_path)
    run_started_at = utc_now()
    log_path = out_dir / "train_log.jsonl"
    if log_path.exists() and resume_adapter is None:
        log_path.unlink()
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "sft_lora" if args.lora else "sft_full",
                "base": args.checkpoint,
                "base_sha256": base_sha256,
                "base_metadata": metadata,
                "data": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
                "data_sha256": dataset_sha256,
                "data_records": dataset_records,
                "dataset_items": len(dataset),
                "sft_stride": args.sft_stride,
                "drop_last": drop_last,
                "supervised_tokens": dataset.supervised_tokens,
                "ignored_tokens": dataset.ignored_tokens,
                "lora": args.lora,
                "lora_rank": args.lora_rank,
                "max_steps": args.max_steps,
                "previous_step": previous_step,
                "target_step": total_target_step,
                "context_length": cfg.model.block_size,
                "started_at": run_started_at,
                "fused_optimizer": any(group.get("fused", False) for group in optimizer.param_groups),
                "lr_scheduler": cfg.train.lr_scheduler,
                "warmup_steps": cfg.train.warmup_steps,
                "min_learning_rate": cfg.train.min_learning_rate,
                "resume_adapter": str(resume_adapter) if resume_adapter is not None else None,
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
    last_loss = None
    progress = tqdm(total=args.max_steps, desc="sft")
    optimizer.zero_grad(set_to_none=True)
    last_fetch_end = time.perf_counter()
    window_start = time.perf_counter()
    window_tokens = 0
    window_data_wait_s = 0.0
    while step < args.max_steps:
        for x, y in loader:
            batch_ready = time.perf_counter()
            window_data_wait_s += batch_ready - last_fetch_end
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context(torch.device(device), cfg.train.precision):
                _logits, loss, info = model(x, y)
            if loss is None:
                raise RuntimeError("Loss was not produced")
            scaler.scale(loss / accumulation).backward()
            micro += 1
            if micro % accumulation:
                last_fetch_end = time.perf_counter()
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            global_step = previous_step + step
            current_lr = set_optimizer_lr(optimizer, learning_rate_at_step(cfg, global_step, total_target_step))
            last_loss = float(loss.detach())
            window_tokens += int(cfg.train.batch_size) * int(cfg.data.block_size) * accumulation
            progress.update(1)
            now = time.perf_counter()
            elapsed = max(now - window_start, 1e-6)
            tokens_per_s = window_tokens / elapsed
            progress.set_postfix(loss=f"{last_loss:.4f}", **{"tok/s": f"{tokens_per_s:.0f}"})
            if step == 1 or step % int(cfg.train.log_every) == 0 or step == args.max_steps:
                log_row = {
                    "step": global_step,
                    "run_step": step,
                    "loss": last_loss,
                    "lr": current_lr,
                    "examples_seen": global_step * cfg.train.batch_size * accumulation,
                    "tokens_seen": global_step * cfg.train.batch_size * accumulation * cfg.data.block_size,
                    "tokens_per_s": tokens_per_s,
                    "data_wait_s": window_data_wait_s,
                    "window_s": elapsed,
                    "created_at": utc_now(),
                }
                log_row.update(flatten_moe_metrics(info))
                append_jsonl(
                    log_path,
                    log_row,
                )
                window_start = now
                window_tokens = 0
                window_data_wait_s = 0.0
            if step % int(cfg.train.save_every) == 0:
                if last_loss is not None and last_loss < best_loss:
                    best_loss = last_loss
                    if args.lora:
                        atomic_torch_save(
                            {
                                "format": "gai1_lora_adapter_v1",
                                "state": lora_state_dict(model),
                                "rank": args.lora_rank,
                                "alpha": 16.0,
                                "dropout": 0.05,
                                "step": global_step,
                                "metadata": {
                                    "step": global_step,
                                    "best_loss": best_loss,
                                    "base_checkpoint": args.checkpoint,
                                    "data": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
                                },
                            },
                            out_dir / "best_adapter.pt",
                        )
                save_training_state(state_path, optimizer, scaler, global_step, best_loss)
            if step >= args.max_steps:
                break
            last_fetch_end = time.perf_counter()
    progress.close()

    final_step = previous_step + step
    if last_loss is not None and last_loss < best_loss:
        best_loss = last_loss
        if args.lora:
            atomic_torch_save(
                {
                    "format": "gai1_lora_adapter_v1",
                    "state": lora_state_dict(model),
                    "rank": args.lora_rank,
                    "alpha": 16.0,
                    "dropout": 0.05,
                    "step": final_step,
                    "metadata": {
                        "step": final_step,
                        "best_loss": best_loss,
                        "base_checkpoint": args.checkpoint,
                        "data": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
                    },
                },
                out_dir / "best_adapter.pt",
            )
    save_training_state(state_path, optimizer, scaler, final_step, best_loss)

    final_metadata = {
        "step": final_step,
        "run_steps": step,
        "previous_step": previous_step,
        "max_steps": args.max_steps,
        "target_step": total_target_step,
        "final_loss": last_loss,
        "best_loss": best_loss,
        "created_at": utc_now(),
        "started_at": run_started_at,
        "config_path": args.config,
        "base_checkpoint": args.checkpoint,
        "base_sha256": base_sha256,
        "data": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
        "data_sha256": dataset_sha256,
        "data_records": dataset_records,
        "dataset_items": len(dataset),
        "sft_stride": args.sft_stride,
        "drop_last": drop_last,
        "supervised_tokens": dataset.supervised_tokens,
        "ignored_tokens": dataset.ignored_tokens,
        "context_length": cfg.model.block_size,
        "fused_optimizer": any(group.get("fused", False) for group in optimizer.param_groups),
        "lr_scheduler": cfg.train.lr_scheduler,
        "warmup_steps": cfg.train.warmup_steps,
        "min_learning_rate": cfg.train.min_learning_rate,
        "trainable_params": trainable_parameter_count(model),
        "resume_adapter": str(resume_adapter) if resume_adapter is not None else None,
        "train_log": str(log_path.relative_to(ROOT)) if log_path.is_relative_to(ROOT) else str(log_path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "sft_lora" if args.lora else "sft_full",
                "base": args.checkpoint,
                "base_metadata": metadata,
                "lora": args.lora,
                "lora_rank": args.lora_rank,
                "lora_targets": replaced,
                **final_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.lora:
        atomic_torch_save(
            {
                "format": "gai1_lora_adapter_v1",
                "state": lora_state_dict(model),
                "rank": args.lora_rank,
                "alpha": 16.0,
                "dropout": 0.05,
                "step": final_step,
                "metadata": final_metadata,
            },
            out_dir / "adapter.pt",
        )
        print(f"Saved LoRA adapter: {out_dir / 'adapter.pt'}")
    else:
        atomic_torch_save(
            {
                "format": "gai1_checkpoint_v1",
                "step": final_step,
                "model_config": model.config_dict(),
                "model_state": model.state_dict(),
                "metadata": {
                    **final_metadata,
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
