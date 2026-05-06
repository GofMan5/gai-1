from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


ARTIFACTS = [
    ("base checkpoint", Path("outputs/gai1_train_gpu/last.pt")),
    ("chat LoRA", Path("outputs/gai1_sft_lora/adapter.pt")),
    ("reasoning LoRA", Path("outputs/gai1_reasoning_lora/adapter.pt")),
    ("INT8 quantized", Path("outputs/quantized/last_int8.pt")),
    ("INT4 quantized", Path("outputs/quantized/last_int4.pt")),
]


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024**2


def checkpoint_summary(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"error": str(exc)}
    summary: dict[str, object] = {}
    for key in ("format", "step", "rank", "alpha", "dropout", "checkpoint_dtype"):
        if key in payload:
            summary[key] = payload[key]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("trained_context_length", "context_length", "final_loss", "data", "data_records"):
            if key in metadata:
                summary[key] = metadata[key]
    return summary


def main() -> int:
    rows = []
    for label, rel_path in ARTIFACTS:
        path = ROOT / rel_path
        exists = path.exists()
        rows.append(
            {
                "name": label,
                "path": str(rel_path),
                "exists": exists,
                "size_mb": round(size_mb(path), 2) if exists else None,
                "summary": checkpoint_summary(path) if exists else None,
            }
        )
    print(json.dumps({"root": str(ROOT), "artifacts": rows}, ensure_ascii=False, indent=2))
    print("\nNote: checkpoints/ is a small registry placeholder; training writes local weights to outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
