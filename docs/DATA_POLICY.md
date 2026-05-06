# GAI-1 Data Policy

GAI-1 нельзя масштабировать без строгой политики данных. Каждый shard должен быть воспроизводимым и объяснимым.

## Обязательные поля manifest

- `source` - откуда пришел текст;
- `license` - право на обучение;
- `language` - ru/en/mixed/code;
- `domain` - chat, code, math, docs, books, web;
- `quality_score` - результат фильтрации;
- `dedupe_hash` - exact/near dedupe ключ;
- `tokenizer_version` - чем токенизировали;
- `created_at` - когда создан shard;
- `split` - train/eval/holdout.

## Правила

- Eval/holdout заносится в blacklist до токенизации train data.
- PII, secrets, private keys и случайные дампы не идут в train.
- Синтетика маркируется отдельно и не доминирует в смеси.
- Русский корпус проверяется на качество, а не просто на наличие кириллицы.
- Любое дообучение должно ссылаться на data manifest.

