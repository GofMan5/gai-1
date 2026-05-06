from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split JSONL into deterministic deduped train/val files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--manifest", default="data/processed/pretrain_split_manifest.json")
    parser.add_argument("--field", default="text")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--min-records", type=int, default=2)
    return parser.parse_args()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_score(seed: int, digest: str) -> float:
    split_digest = hashlib.sha256(f"{seed}:{digest}".encode("utf-8")).digest()
    value = int.from_bytes(split_digest[:8], "big")
    return value / float(2**64 - 1)


def relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    configure_console()
    args = parse_args()
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    if args.min_records < 2:
        raise ValueError("--min-records must be at least 2")

    input_path = ROOT / args.input
    train_path = ROOT / args.train_out
    val_path = ROOT / args.val_out
    manifest_path = ROOT / args.manifest
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    stats = {
        "input_records": 0,
        "kept_records": 0,
        "duplicate_records": 0,
        "too_short_records": 0,
        "train_records": 0,
        "val_records": 0,
        "train_chars": 0,
        "val_chars": 0,
    }
    source_counts: dict[str, int] = {}

    with input_path.open("r", encoding="utf-8") as src, train_path.open("w", encoding="utf-8") as train, val_path.open(
        "w", encoding="utf-8"
    ) as val:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            stats["input_records"] += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {input_path} line {line_no}")
            text = str(row.get(args.field, "")).strip()
            if len(text) < args.min_chars:
                stats["too_short_records"] += 1
                continue
            digest = stable_hash(text)
            if digest in seen:
                stats["duplicate_records"] += 1
                continue
            seen.add(digest)
            row.setdefault("dedupe_sha256", digest)
            source = str(row.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            stats["kept_records"] += 1
            if split_score(args.seed, digest) < args.val_fraction:
                val.write(encoded)
                stats["val_records"] += 1
                stats["val_chars"] += len(text)
            else:
                train.write(encoded)
                stats["train_records"] += 1
                stats["train_chars"] += len(text)

    if stats["kept_records"] < args.min_records:
        raise ValueError(f"Split kept too few records: {stats['kept_records']} < {args.min_records}")
    if stats["train_records"] <= 0 or stats["val_records"] <= 0:
        raise ValueError(
            "Split produced an empty train or validation file. "
            f"train={stats['train_records']} val={stats['val_records']}; increase input size or val fraction."
        )

    write_manifest(
        manifest_path,
        {
            "format": "gai1_jsonl_split_manifest_v1",
            "algorithm_version": 1,
            "created_by": "scripts/split_jsonl_dataset.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language": "ru",
            "input": relative_or_absolute(input_path),
            "input_sha256": file_sha256(input_path),
            "train": relative_or_absolute(train_path),
            "train_sha256": file_sha256(train_path),
            "val": relative_or_absolute(val_path),
            "val_sha256": file_sha256(val_path),
            "field": args.field,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "min_chars": args.min_chars,
            "min_records": args.min_records,
            "stats": stats,
            "sources": source_counts,
        },
    )
    print(json.dumps({"train": stats["train_records"], "val": stats["val_records"], "manifest": args.manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
