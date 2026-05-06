from __future__ import annotations

import argparse
from contextlib import nullcontext
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

from gai1.config import load_config, save_json
from gai1.data import PackedTextDataset, StreamingPackedTextDataset
from gai1.model import GAIModel, format_param_count
from gai1.tokenizer import BPETokenizer, ByteTokenizer


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the local GAI-1 tiny language model.")
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests CUDA, but this Python has CPU-only PyTorch or cannot see the GPU. "
            "Run scripts/setup_rtx3060_windows.ps1, then use .\\.venv\\Scripts\\python.exe."
        )
    return device


def configure_acceleration(cfg: Any, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.train.allow_tf32)
    torch.set_float32_matmul_precision(cfg.train.matmul_precision)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    chosen = precision
    if chosen == "auto":
        chosen = "fp16"
    if chosen == "fp32":
        return nullcontext()
    if chosen == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if chosen == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported precision: {precision}")


def make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"auto", "fp16"}
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loader(dataset: torch.utils.data.Dataset, cfg: Any, device: torch.device, streaming: bool = False) -> DataLoader:
    num_workers = int(cfg.train.num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": cfg.train.batch_size,
        "shuffle": False if streaming else True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda" and bool(cfg.train.pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.train.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.train.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def make_eval_loader(dataset: torch.utils.data.Dataset, cfg: Any, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda" and bool(cfg.train.pin_memory),
    )


def make_optimizer(parameters, cfg: Any, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
    }
    if device.type == "cuda" and bool(getattr(cfg.train, "fused_optimizer", True)):
        try:
            if "fused" in inspect.signature(torch.optim.AdamW).parameters:
                kwargs["fused"] = True
        except (TypeError, ValueError):
            pass
    return torch.optim.AdamW(parameters, **kwargs)


def learning_rate_at_step(cfg: Any, step: int) -> float:
    scheduler = str(getattr(cfg.train, "lr_scheduler", "cosine")).lower()
    base_lr = float(cfg.train.learning_rate)
    min_lr = float(getattr(cfg.train, "min_learning_rate", 0.0))
    warmup_steps = max(0, int(getattr(cfg.train, "warmup_steps", 0)))
    max_steps = max(1, int(cfg.train.max_steps))
    if scheduler not in {"constant", "cosine"}:
        raise ValueError("train.lr_scheduler must be 'constant' or 'cosine'")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if scheduler == "constant":
        return base_lr
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> float:
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def cuda_memory_summary(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "vram_total_gb": round(total / 1024**3, 3),
        "vram_free_gb": round(free / 1024**3, 3),
        "vram_used_gb": round((total - free) / 1024**3, 3),
    }


def estimate_attention_scores_gb(cfg: Any, dtype_bytes: int = 2) -> float:
    batch = int(cfg.train.batch_size)
    seq = int(cfg.data.block_size)
    heads = int(cfg.model.n_head)
    layers = int(cfg.model.n_layer)
    return batch * heads * seq * seq * dtype_bytes * layers / 1024**3


def validate_context_budget(cfg: Any, device: torch.device) -> None:
    if int(cfg.model.block_size) != int(cfg.data.block_size):
        raise ValueError(f"model.block_size ({cfg.model.block_size}) must equal data.block_size ({cfg.data.block_size})")
    if cfg.model.rope_scaling not in {"none", "linear", "dynamic_ntk"}:
        raise ValueError("model.rope_scaling must be one of: none, linear, dynamic_ntk")
    if cfg.model.block_size > 8192 and cfg.model.rope_original_context <= 0:
        raise ValueError("Long-context configs must set model.rope_original_context")
    if device.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info(device)
    estimated_gb = estimate_attention_scores_gb(cfg)
    limit_gb = (total / 1024**3) * float(cfg.train.max_attention_memory_fraction)
    if estimated_gb > limit_gb and not cfg.train.allow_unsafe_long_context:
        raise RuntimeError(
            "Refusing unsafe full-attention training config: "
            f"block_size={cfg.data.block_size}, batch_size={cfg.train.batch_size}, "
            f"estimated attention-score memory={estimated_gb:.1f}GB, limit={limit_gb:.1f}GB. "
            "For 128k context, train in stages with shorter chunks/context-extension data or use a distributed "
            "long-context stack with sequence parallelism. Set train.allow_unsafe_long_context=true only if you "
            "know this machine can handle it."
        )


def load_training_tokenizer(cfg: Any):
    if cfg.tokenizer.kind == "byte":
        return ByteTokenizer()
    if cfg.tokenizer.kind == "bpe":
        tokenizer_path = ROOT / cfg.tokenizer.path
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}. Run: "
                ".\\.venv\\Scripts\\python.exe .\\scripts\\train_tokenizer.py"
            )
        return BPETokenizer(tokenizer_path)
    raise ValueError(f"Unsupported tokenizer kind: {cfg.tokenizer.kind}")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def checkpoint_state_dict(model: torch.nn.Module, checkpoint_dtype: str) -> dict[str, torch.Tensor]:
    state = unwrap_model(model).state_dict()
    if checkpoint_dtype == "fp32":
        return {key: value.detach().cpu() for key, value in state.items()}
    if checkpoint_dtype == "fp16":
        return {
            key: value.detach().cpu().to(torch.float16) if value.is_floating_point() else value.detach().cpu()
            for key, value in state.items()
        }
    raise ValueError(f"Unsupported checkpoint_dtype: {checkpoint_dtype}")


def tokenizer_metadata(cfg: Any) -> dict[str, Any]:
    tokenizer_path = ROOT / cfg.tokenizer.path
    return {
        "kind": cfg.tokenizer.kind,
        "path": cfg.tokenizer.path,
        "sha256": file_sha256(tokenizer_path),
        "vocab_size": cfg.tokenizer.vocab_size,
        "byte_fallback": cfg.tokenizer.byte_fallback,
    }


def data_metadata(cfg: Any) -> dict[str, Any]:
    train_path = ROOT / cfg.data.train_path
    val_path = ROOT / cfg.data.val_path if cfg.data.val_path else None
    return {
        "train_path": cfg.data.train_path,
        "train_sha256": file_sha256(train_path),
        "val_path": cfg.data.val_path,
        "val_sha256": file_sha256(val_path) if val_path is not None else None,
        "field": cfg.data.field,
        "streaming": bool(getattr(cfg.data, "streaming", False)),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    config_path: str,
    cfg: Any,
) -> None:
    raw_model = unwrap_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "gai1_checkpoint_v1",
        "step": step,
        "model_config": raw_model.config_dict(),
        "model_state": checkpoint_state_dict(raw_model, cfg.train.checkpoint_dtype),
        "tokenizer": tokenizer_metadata(cfg),
        "data": data_metadata(cfg),
        "config_path": config_path,
        "checkpoint_dtype": cfg.train.checkpoint_dtype,
        "metadata": {
            "learning_rate": cfg.train.learning_rate,
            "lr_scheduler": cfg.train.lr_scheduler,
            "warmup_steps": cfg.train.warmup_steps,
            "min_learning_rate": cfg.train.min_learning_rate,
            "trained_context_length": cfg.model.block_size,
            "tested_context_length": cfg.model.block_size,
            "rope_scaling": cfg.model.rope_scaling,
            "rope_original_context": cfg.model.rope_original_context,
            "long_context_validated": cfg.model.block_size >= 131072,
        },
    }
    if cfg.train.save_optimizer_state:
        payload["optimizer_state"] = optimizer.state_dict()
        payload["scaler_state"] = scaler.state_dict()
    atomic_torch_save(payload, path)


def save_training_state(
    path: Path,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    best_metric: float,
    best_loss: float | None,
) -> None:
    payload: dict[str, Any] = {
        "format": "gai1_training_state_v1",
        "step": step,
        "best_metric": best_metric,
        "best_loss": best_loss,
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    atomic_torch_save(payload, path)


def load_training_state(path: Path, optimizer: torch.optim.Optimizer, scaler: Any, device: torch.device) -> tuple[int, float, float | None]:
    if not path.exists():
        return 0, float("inf"), None
    state = torch.load(path, map_location=device)
    if state.get("format") != "gai1_training_state_v1":
        return 0, float("inf"), None
    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(state.get("step", 0)), float(state.get("best_metric", float("inf"))), state.get("best_loss")


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    checkpoint_path: Path,
    device: torch.device,
    load_optimizer: bool,
) -> int:
    if not checkpoint_path.exists():
        return 0
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("format") != "gai1_checkpoint_v1":
        raise ValueError(f"Unsupported checkpoint format in resume file: {checkpoint.get('format')}")
    try:
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        print(f"Resume skipped: checkpoint is incompatible with current config ({exc})")
        return 0
    if load_optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if load_optimizer and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return int(checkpoint.get("step", 0))


def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Any,
) -> dict[str, float]:
    raw_was_training = model.training
    model.eval()
    losses: list[float] = []
    moe_losses: list[float] = []
    max_batches = max(1, int(cfg.train.eval_batches))
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context(device, cfg.train.precision):
                _logits, loss, info = model(x, y)
            if loss is not None:
                losses.append(float(loss.detach()))
            moe_losses.append(float(info["moe_aux_loss"].detach()))
    if raw_was_training:
        model.train()
    if not losses:
        return {"val_loss": float("inf"), "val_ppl": float("inf"), "val_moe_aux_loss": 0.0}
    val_loss = sum(losses) / len(losses)
    return {
        "val_loss": val_loss,
        "val_ppl": math.exp(min(20.0, val_loss)),
        "val_moe_aux_loss": sum(moe_losses) / max(1, len(moe_losses)),
    }


def main() -> int:
    configure_console()
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = resolve_device(cfg.train.device)
    configure_acceleration(cfg, device)
    validate_context_budget(cfg, device)
    tokenizer = load_training_tokenizer(cfg)
    dataset_cls = StreamingPackedTextDataset if bool(getattr(cfg.data, "streaming", False)) else PackedTextDataset
    dataset = dataset_cls(
        path=ROOT / cfg.data.train_path,
        tokenizer=tokenizer,
        block_size=cfg.data.block_size,
        field=cfg.data.field,
    )
    loader = make_loader(dataset, cfg, device, streaming=bool(getattr(cfg.data, "streaming", False)))
    val_loader = None
    if cfg.data.val_path:
        val_dataset = PackedTextDataset(
            path=ROOT / cfg.data.val_path,
            tokenizer=tokenizer,
            block_size=cfg.data.block_size,
            field=cfg.data.field,
        )
        val_loader = make_eval_loader(val_dataset, cfg, device)
    raw_model = GAIModel(cfg.model).to(device)
    raw_model.set_gradient_checkpointing(bool(cfg.train.gradient_checkpointing))
    model: torch.nn.Module = raw_model
    if cfg.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = make_optimizer(model.parameters(), cfg, device)
    scaler = make_grad_scaler(device, cfg.train.precision)
    accumulation = max(1, int(cfg.train.gradient_accumulation_steps))

    out_dir = ROOT / cfg.train.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = out_dir / "train_log.jsonl"
    save_json(
        out_dir / "run_manifest.json",
        {
            "run_name": cfg.run_name,
            "config": args.config,
            "params": raw_model.parameter_count(),
            "device": str(device),
            "precision": cfg.train.precision,
            "batch_size": cfg.train.batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": cfg.train.batch_size * accumulation,
            "num_workers": cfg.train.num_workers,
            "pin_memory": cfg.train.pin_memory,
            "persistent_workers": cfg.train.persistent_workers,
            "prefetch_factor": cfg.train.prefetch_factor,
            "fused_optimizer": any(group.get("fused", False) for group in optimizer.param_groups),
            "lr_scheduler": cfg.train.lr_scheduler,
            "warmup_steps": cfg.train.warmup_steps,
            "min_learning_rate": cfg.train.min_learning_rate,
            "torch_compile": bool(cfg.train.compile),
            "gradient_checkpointing": bool(cfg.train.gradient_checkpointing),
            "cuda_memory_at_start": cuda_memory_summary(device),
            "dataset": data_metadata(cfg),
            "tokenizer": tokenizer_metadata(cfg),
            "eval_every": cfg.train.eval_every,
            "eval_batches": cfg.train.eval_batches,
        },
    )

    step = 0
    resume_path = out_dir / "last.pt"
    state_path = out_dir / "training_state.pt"
    best_metric = float("inf")
    best_loss = None
    if cfg.train.resume:
        step = maybe_resume(model, optimizer, scaler, resume_path, device, cfg.train.save_optimizer_state)
        state_step, best_metric, best_loss = load_training_state(state_path, optimizer, scaler, device)
        step = max(step, state_step)
    current_lr = set_optimizer_lr(optimizer, learning_rate_at_step(cfg, step))

    print(
        "GAI-1 train start: "
        f"params={format_param_count(raw_model.parameter_count())} device={device} "
        f"precision={cfg.train.precision} micro_batch={cfg.train.batch_size} "
        f"accum={accumulation} effective_batch={cfg.train.batch_size * accumulation} "
        f"resume_step={step}"
    )
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} VRAM={props.total_memory / 1024**3:.2f}GB start={cuda_memory_summary(device)}")
    micro_step = 0
    last_loss = None
    model.train()
    progress = tqdm(total=cfg.train.max_steps, initial=min(step, cfg.train.max_steps), desc="train")
    optimizer.zero_grad(set_to_none=True)
    last_fetch_end = time.perf_counter()
    window_start = time.perf_counter()
    window_tokens = 0
    window_data_wait_s = 0.0
    while step < cfg.train.max_steps:
        for x, y in loader:
            batch_ready = time.perf_counter()
            window_data_wait_s += batch_ready - last_fetch_end
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context(device, cfg.train.precision):
                _logits, loss, info = model(x, y)
            if loss is None:
                raise RuntimeError("Loss was not produced")
            scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            if micro_step % accumulation != 0:
                last_fetch_end = time.perf_counter()
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            last_loss = float(loss.detach())
            current_lr = set_optimizer_lr(optimizer, learning_rate_at_step(cfg, step))
            window_tokens += int(cfg.train.batch_size) * int(cfg.data.block_size) * accumulation
            if step % cfg.train.log_every == 0:
                now = time.perf_counter()
                elapsed = max(now - window_start, 1e-6)
                tokens_per_s = window_tokens / elapsed
                postfix = {
                    "loss": f"{last_loss:.4f}",
                    "moe_aux": f"{float(info['moe_aux_loss'].detach()):.4f}",
                    "tok/s": f"{tokens_per_s:.0f}",
                }
                log_row = {
                    "step": step,
                    "loss": last_loss,
                    "moe_aux_loss": float(info["moe_aux_loss"].detach()),
                    "tokens_per_s": tokens_per_s,
                    "tokens_seen": step * int(cfg.train.batch_size) * int(cfg.train.gradient_accumulation_steps) * int(cfg.data.block_size),
                    "lr": current_lr,
                    "data_wait_s": window_data_wait_s,
                    "window_s": elapsed,
                    "micro_batch": int(cfg.train.batch_size),
                    "accumulation": accumulation,
                }
                if device.type == "cuda":
                    alloc_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                    reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
                    postfix["vram"] = f"{alloc_gb:.2f}GB"
                    log_row["vram_allocated_gb"] = alloc_gb
                    log_row["vram_reserved_gb"] = reserved_gb
                progress.set_postfix(postfix)
                if val_loader is not None and (step == 1 or step % int(cfg.train.eval_every) == 0):
                    eval_row = {"step": step, **run_validation(model, val_loader, device, cfg), "created_at": time.time()}
                    append_jsonl(out_dir / "eval_log.jsonl", eval_row)
                    log_row.update(eval_row)
                    metric = float(eval_row["val_loss"])
                    if metric < best_metric:
                        best_metric = metric
                        best_loss = last_loss
                        save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, step, args.config, cfg)
                elif last_loss < best_metric:
                    best_metric = last_loss
                    best_loss = last_loss
                    save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, step, args.config, cfg)
                append_jsonl(train_log_path, log_row)
                window_start = now
                window_tokens = 0
                window_data_wait_s = 0.0
            if step % cfg.train.save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, step, args.config, cfg)
                save_training_state(state_path, optimizer, scaler, step, best_metric, best_loss)
            progress.update(1)
            if step >= cfg.train.max_steps:
                break
            last_fetch_end = time.perf_counter()
    progress.close()
    save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, step, args.config, cfg)
    save_training_state(state_path, optimizer, scaler, step, best_metric, best_loss)
    print(f"Saved checkpoint: {out_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
