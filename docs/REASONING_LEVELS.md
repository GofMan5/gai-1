# GAI-1 Reasoning Levels

GAI-1 имеет четыре runtime-режима мышления: `low`, `medium`, `high`, `max`.

| Level | Для чего | Что меняется |
| --- | --- | --- |
| `low` | быстрый простой ответ | короткий план, 1 черновик, без критика/verifier |
| `medium` | обычный чат и небольшие задачи | план глубже, 1 critic pass, 1 verifier pass, 1 rollback |
| `high` | код, архитектура, математика | 2 черновика, 2 critic/verifier pass, self-consistency |
| `max` | сложные и рискованные задачи | 3 черновика, 3 critic/verifier pass, больше rollback/tool budget |

## Реальные параметры

Профили живут в:

- `src/gai1/reasoning.py` - runtime implementation;
- `configs/reasoning_modes.json` - конфиг-описание для будущего UI/API;
- `scripts/reason.py` - CLI smoke/demo.

Пример:

```powershell
python .\scripts\reason.py --level high --task "Спроектируй pipeline дообучения GAI-1" --show-private-trace
```

## Важное

Эти уровни не заставляют модель магически умнеть. Они управляют runtime-циклом: сколько планировать, сколько черновиков делать, сколько раз критиковать, проверять и откатывать. Для настоящего качества эти traces потом надо использовать в SFT/reasoning-tuning и eval gates.

## Как добавить уровень в будущем

1. Открой `configs/reasoning_modes.json`.
2. В объект `levels` добавь новый ключ, например `extreme`.
3. Скопируй структуру с `max` и измени параметры.
4. Проверь:

```powershell
python .\scripts\reason.py --level extreme --task "Проверь новый режим" --show-private-trace
```

5. Добавь тест или eval gate, если уровень будет использоваться не только для экспериментов.

Минимальный набор полей:

- `planning_depth`;
- `draft_count`;
- `critic_passes`;
- `verifier_passes`;
- `rollback_limit`;
- `tool_budget`;
- `private_token_budget`;
- `self_consistency`;
- `temperature`.
