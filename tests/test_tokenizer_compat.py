from __future__ import annotations

from types import SimpleNamespace

import pytest

from gai1.tokenizer.compat import (
    assert_tokenizer_compatible,
    tokenizer_compatibility_issues,
    tokenizer_spec_from_config,
    tokenizer_spec_from_tokenizer,
)


def make_cfg(path: str, kind: str = "bpe", vocab_size: int = 32000):
    return SimpleNamespace(
        tokenizer=SimpleNamespace(
            kind=kind,
            path=path,
            vocab_size=vocab_size,
            byte_fallback=True,
        )
    )


def test_tokenizer_spec_from_config_hashes_file(tmp_path) -> None:
    tokenizer_path = tmp_path / "tok.json"
    tokenizer_path.write_text('{"x":1}', encoding="utf-8")
    cfg = make_cfg("tok.json")

    spec = tokenizer_spec_from_config(cfg, tmp_path)

    assert spec["kind"] == "bpe"
    assert spec["vocab_size"] == 32000
    assert spec["sha256"]


def test_tokenizer_spec_from_tokenizer_uses_actual_vocab_size(tmp_path) -> None:
    tokenizer_path = tmp_path / "tok.json"
    tokenizer_path.write_text("runtime", encoding="utf-8")
    cfg = make_cfg("tok.json", vocab_size=32000)
    tokenizer = SimpleNamespace(vocab_size=123)

    spec = tokenizer_spec_from_tokenizer(cfg, tokenizer, tmp_path)

    assert spec["vocab_size"] == 123


def test_tokenizer_compatibility_detects_mismatch(tmp_path) -> None:
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text("runtime", encoding="utf-8")
    cfg = make_cfg("runtime.json", vocab_size=100)
    runtime = tokenizer_spec_from_config(cfg, tmp_path)
    checkpoint = {"kind": "byte", "vocab_size": 260, "sha256": "different"}

    issues = tokenizer_compatibility_issues(checkpoint, runtime)

    assert any("kind" in issue for issue in issues)
    assert any("vocab_size" in issue for issue in issues)
    assert any("sha256" in issue for issue in issues)


def test_assert_tokenizer_compatible_allows_legacy_missing_metadata(tmp_path) -> None:
    cfg = make_cfg("missing.json")

    result = assert_tokenizer_compatible({"tokenizer": None}, cfg, tmp_path)

    assert result["status"] == "unknown"
    assert result["issues"] == []


def test_assert_tokenizer_compatible_raises_unless_allowed(tmp_path) -> None:
    tokenizer_path = tmp_path / "tok.json"
    tokenizer_path.write_text("runtime", encoding="utf-8")
    cfg = make_cfg("tok.json", vocab_size=100)
    metadata = {"tokenizer": {"kind": "bpe", "vocab_size": 101, "sha256": "bad"}}

    with pytest.raises(ValueError, match="Tokenizer/checkpoint mismatch"):
        assert_tokenizer_compatible(metadata, cfg, tmp_path)

    result = assert_tokenizer_compatible(metadata, cfg, tmp_path, allow_mismatch=True)
    assert result["status"] == "mismatch"
    assert result["issues"]


def test_path_mismatch_is_only_strict_when_requested(tmp_path) -> None:
    tokenizer_path = tmp_path / "tok.json"
    tokenizer_path.write_text("same", encoding="utf-8")
    cfg = make_cfg("tok.json", vocab_size=100)
    runtime = tokenizer_spec_from_config(cfg, tmp_path)
    checkpoint = {**runtime, "path": "elsewhere/tok.json"}

    assert tokenizer_compatibility_issues(checkpoint, runtime) == []
    assert any("path" in issue for issue in tokenizer_compatibility_issues(checkpoint, runtime, strict_path=True))
