from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_split(src: Path, train: Path, val: Path, manifest: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "split_jsonl_dataset.py"),
            "--input",
            str(src),
            "--train-out",
            str(train),
            "--val-out",
            str(val),
            "--manifest",
            str(manifest),
            "--val-fraction",
            "0.49",
            "--seed",
            "7",
        ],
        cwd=ROOT,
        check=True,
    )


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_split_jsonl_dataset_dedupes_and_writes_manifest(tmp_path: Path) -> None:
    src = tmp_path / "input.jsonl"
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    manifest = tmp_path / "manifest.json"
    rows = [
        {"text": "\u041f\u0435\u0440\u0432\u044b\u0439 \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u043f\u0440\u043e \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u043c\u043e\u0434\u0435\u043b\u0438.", "source": "test"},
        {"text": "\u0412\u0442\u043e\u0440\u043e\u0439 \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u043f\u0440\u043e \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0443 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0430.", "source": "test"},
        {"text": "\u041f\u0435\u0440\u0432\u044b\u0439 \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u043f\u0440\u043e \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u043c\u043e\u0434\u0435\u043b\u0438.", "source": "test"},
        {"text": "\u0422\u0440\u0435\u0442\u0438\u0439 \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u043f\u0440\u043e tokenizer \u0438 \u0434\u0430\u043d\u043d\u044b\u0435.", "source": "test"},
    ]
    write_rows(src, rows)

    run_split(src, train, val, manifest)

    split_rows = read_jsonl(train) + read_jsonl(val)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert len(split_rows) == 3
    assert all("dedupe_sha256" in row for row in split_rows)
    assert payload["format"] == "gai1_jsonl_split_manifest_v1"
    assert payload["language"] == "ru"
    assert payload["input_sha256"]
    assert payload["train_sha256"]
    assert payload["val_sha256"]
    assert payload["stats"]["input_records"] == 4
    assert payload["stats"]["duplicate_records"] == 1
    assert payload["stats"]["kept_records"] == 3
    assert payload["stats"]["train_records"] > 0
    assert payload["stats"]["val_records"] > 0


def test_split_jsonl_dataset_is_deterministic(tmp_path: Path) -> None:
    rows = [
        {"text": f"\u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442 {idx} \u043f\u0440\u043e \u0440\u0443\u0441\u0441\u043a\u0443\u044e \u043c\u043e\u0434\u0435\u043b\u044c \u0438 eval.", "source": "test"}
        for idx in range(12)
    ]
    src = tmp_path / "input.jsonl"
    write_rows(src, rows)
    train_a, val_a, manifest_a = tmp_path / "train_a.jsonl", tmp_path / "val_a.jsonl", tmp_path / "manifest_a.json"
    train_b, val_b, manifest_b = tmp_path / "train_b.jsonl", tmp_path / "val_b.jsonl", tmp_path / "manifest_b.json"

    run_split(src, train_a, val_a, manifest_a)
    run_split(src, train_b, val_b, manifest_b)

    assert train_a.read_text(encoding="utf-8") == train_b.read_text(encoding="utf-8")
    assert val_a.read_text(encoding="utf-8") == val_b.read_text(encoding="utf-8")
