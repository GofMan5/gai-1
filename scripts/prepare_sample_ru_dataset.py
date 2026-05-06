from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "sample_ru_chat.jsonl"


SAMPLES = [
    "Пользователь: Привет. Кто ты?\nАссистент: Я GAI-1, русскоязычная модельная заготовка для обучения и reasoning.",
    "Пользователь: Объясни кратко, что такое reasoning.\nАссистент: Reasoning - это способность строить план, проверять шаги и выдавать ответ после внутренней проверки.",
    "Пользователь: Как дообучать модель безопасно?\nАссистент: Нужно сохранять data manifest, base checkpoint, training config, eval report и rollback target.",
    "Пользователь: Почему русский tokenizer важен?\nАссистент: Если tokenizer плохо сжимает кириллицу, модель тратит больше токенов и хуже учит русский контекст.",
    "Пользователь: Что такое MoE?\nАссистент: MoE - это mixture of experts: на токен активируется часть экспертов, поэтому total parameters могут быть большими.",
    "Пользователь: Как проверить новый checkpoint?\nАссистент: Его сравнивают с baseline на eval gates: русский чат, reasoning, код, tool-use, safety и latency.",
]


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    configure_console()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for sample in SAMPLES:
            fh.write(json.dumps({"text": sample}, ensure_ascii=False))
            fh.write("\n")
    print(f"Saved {len(SAMPLES)} samples: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
