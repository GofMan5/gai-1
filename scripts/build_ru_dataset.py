from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


TOPICS = [
    ("архитектура модели", "объясняет слои пайплайна: данные, tokenizer, pretrain, SFT, eval и serving"),
    ("reasoning", "строит план, проверяет слабые места и выдает короткий финальный ответ"),
    ("дообучение", "сохраняет base checkpoint, data manifest, config и rollback target"),
    ("квантование", "отделяет число параметров от размера хранения весов"),
    ("код", "предлагает минимальный патч и просит проверить тестами"),
    ("русский язык", "использует точные термины и не льет воду"),
    ("безопасность", "отделяет защитные задачи от вредных инструкций"),
    ("ошибки", "ищет воспроизводимый пример и проверяет гипотезы"),
]


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stronger local RU SFT/pretrain seed dataset.")
    parser.add_argument("--out", default="data/raw/gai1_ru_seed.jsonl")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def make_record(rng: random.Random) -> dict[str, str]:
    topic, behavior = rng.choice(TOPICS)
    styles = [
        "ответь кратко и точно",
        "сначала дай план, потом ответ",
        "объясни по-русски без воды",
        "проверь слабые места решения",
        "сделай production-style вывод",
    ]
    prompt = f"{rng.choice(styles)}: как GAI-1 должен работать с темой '{topic}'?"
    response = (
        f"GAI-1 должен {behavior}. "
        f"Рабочая схема: определить цель, собрать контекст, сделать черновик, проверить результат, "
        f"записать trace для будущего дообучения."
    )
    return {"prompt": prompt, "response": response, "topic": topic}


def main() -> int:
    configure_console()
    args = parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with out.open("w", encoding="utf-8") as fh:
        for _ in range(args.count):
            fh.write(json.dumps(make_record(rng), ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
    print(f"Saved {args.count} records: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
