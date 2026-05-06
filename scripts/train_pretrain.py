from __future__ import annotations

import argparse
from contextlib import nullcontext
import random
import sys
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
        "config_path": config_path,
        "checkpoint_dtype": cfg.train.checkpoint_dtype,
        "metadata": {
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
    torch.save(payload, path)


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
    raw_model = GAIModel(cfg.model).to(device)
    raw_model.set_gradient_checkpointing(bool(cfg.train.gradient_checkpointing))
    model: torch.nn.Module = raw_model
    if cfg.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scaler = make_grad_scaler(device, cfg.train.precision)
    accumulation = max(1, int(cfg.train.gradient_accumulation_steps))

    out_dir = ROOT / cfg.train.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
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
            "cuda_memory_at_start": cuda_memory_summary(device),
            "dataset": cfg.data.train_path,
            "dataset_streaming": bool(getattr(cfg.data, "streaming", False)),
            "tokenizer": cfg.tokenizer.path,
        },
    )

    step = 0
    resume_path = out_dir / "last.pt"
    if cfg.train.resume:
        step = maybe_resume(model, optimizer, scaler, resume_path, device, cfg.train.save_optimizer_state)

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
    model.train()
    progress = tqdm(total=cfg.train.max_steps, initial=min(step, cfg.train.max_steps), desc="train")
    optimizer.zero_grad(set_to_none=True)
    while step < cfg.train.max_steps:
        for x, y in loader:
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
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % cfg.train.log_every == 0:
                postfix = {
                    "loss": f"{float(loss.detach()):.4f}",
                    "moe_aux": f"{float(info['moe_aux_loss'].detach()):.4f}",
                }
                if device.type == "cuda":
                    postfix["vram"] = f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GB"
                progress.set_postfix(postfix)
            if step % cfg.train.save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, step, args.config, cfg)
            progress.update(1)
            if step >= cfg.train.max_steps:
                break
    progress.close()
    save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, step, args.config, cfg)
    print(f"Saved checkpoint: {out_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
