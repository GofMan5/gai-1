from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_spec_from_config(cfg: Any, root: str | Path) -> dict[str, Any]:
    tokenizer_path = Path(root) / cfg.tokenizer.path
    return {
        "kind": cfg.tokenizer.kind,
        "path": cfg.tokenizer.path,
        "sha256": file_sha256(tokenizer_path),
        "vocab_size": cfg.tokenizer.vocab_size,
        "byte_fallback": getattr(cfg.tokenizer, "byte_fallback", None),
    }


def tokenizer_spec_from_tokenizer(cfg: Any, tokenizer: object, root: str | Path) -> dict[str, Any]:
    spec = tokenizer_spec_from_config(cfg, root)
    spec["vocab_size"] = int(getattr(tokenizer, "vocab_size", spec["vocab_size"]))
    return spec


def tokenizer_compatibility_issues(
    checkpoint_spec: object,
    runtime_spec: dict[str, Any],
    strict_path: bool = False,
) -> list[str]:
    if not isinstance(checkpoint_spec, dict) or not checkpoint_spec:
        return []
    issues: list[str] = []
    for key in ("kind", "vocab_size"):
        expected = checkpoint_spec.get(key)
        actual = runtime_spec.get(key)
        if expected is not None and actual is not None and expected != actual:
            issues.append(f"tokenizer {key} mismatch: checkpoint={expected!r} runtime={actual!r}")
    expected_hash = checkpoint_spec.get("sha256")
    actual_hash = runtime_spec.get("sha256")
    if expected_hash and actual_hash and expected_hash != actual_hash:
        issues.append("tokenizer sha256 mismatch")
    expected_path = checkpoint_spec.get("path")
    actual_path = runtime_spec.get("path")
    if strict_path and expected_path and actual_path and Path(str(expected_path)).as_posix() != Path(str(actual_path)).as_posix():
        issues.append(f"tokenizer path mismatch: checkpoint={expected_path!r} runtime={actual_path!r}")
    return issues


def assert_tokenizer_compatible(
    checkpoint_metadata: dict[str, Any],
    cfg: Any,
    root: str | Path,
    tokenizer: object | None = None,
    allow_mismatch: bool = False,
    strict_path: bool = False,
) -> dict[str, Any]:
    runtime_spec = tokenizer_spec_from_tokenizer(cfg, tokenizer, root) if tokenizer is not None else tokenizer_spec_from_config(cfg, root)
    checkpoint_spec = checkpoint_metadata.get("tokenizer")
    issues = tokenizer_compatibility_issues(checkpoint_spec, runtime_spec, strict_path=strict_path)
    status = "unknown" if not isinstance(checkpoint_spec, dict) or not checkpoint_spec else ("ok" if not issues else "mismatch")
    if issues and not allow_mismatch:
        joined = "; ".join(issues)
        raise ValueError(f"Tokenizer/checkpoint mismatch: {joined}. Pass an explicit allow-mismatch flag only for debugging.")
    return {"status": status, "runtime": runtime_spec, "checkpoint": checkpoint_spec, "issues": issues}
