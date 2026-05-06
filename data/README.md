# Data Layout

GAI-1 держит данные отдельно от кода.

- `raw/` - исходные JSONL/текстовые файлы.
- `processed/` - очищенные, дедуплицированные и токенизированные шарды.
- `evals/` - holdout-наборы, которые никогда не попадают в train.

Минимальный JSONL формат:

```jsonl
{"text":"Пользователь: Привет\nАссистент: Привет. Я GAI-1, русскоязычная модель."}
```

Позже этот слой должен стать полноценным data lake: source, license, language, quality score, dedupe hash, tokenizer version и dataset manifest для каждого shard.

