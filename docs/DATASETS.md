# GAI-1 Datasets

## Pretraining

- Source: `HuggingFaceFW/fineweb-2`
- Subset: `rus_Cyrl`
- Purpose: Russian continued pretraining / language modeling
- Local sample: `data/raw/fineweb2_ru_sample.jsonl`
- Script: `scripts/download_fineweb2_ru.py`

Command:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_fineweb2_ru.py --max-docs 1200 --min-chars 500 --max-chars 6000
```

This validates the pipeline, but it is far too small for strong chat quality.

## Chat SFT

- Source: `IlyaGusev/ru_turbo_alpaca`
- Purpose: Russian instruction tuning
- Local sample: `data/raw/ru_turbo_alpaca_sample.jsonl`
- Script: `scripts/download_ru_turbo_alpaca.py`

Command:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_ru_turbo_alpaca.py --max-records 5000
```

Important licensing note: `ru_turbo_alpaca` is shown as CC-BY-4.0 on Hugging Face, but its dataset card says it was generated with gpt-3.5-turbo and includes restrictions around commercial competing products. Use it for local prototyping, not as the final production legal dataset.

## Production Data Direction

For a serious model, replace or mix the prototype SFT set with a larger legally clean dataset:

- Russian dialogs.
- Instruction following.
- Safety refusals.
- Code tasks.
- Math/reasoning traces.
- Regression eval prompts held out from training.
