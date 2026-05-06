# Contributing

Thanks for improving GAI-1. Keep contributions focused, tested, and easy to review.

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

## Checks

Run tests before opening a pull request:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

For CUDA changes, also run:

```powershell
.\.venv\Scripts\python.exe scripts\check_accelerator.py
.\.venv\Scripts\python.exe scripts\train_pretrain.py --max-steps 1
.\.venv\Scripts\python.exe scripts\train_sft.py --checkpoint outputs\gai1_train_gpu\last.pt --lora --max-steps 1
```

## Pull Request Rules

- Keep changes scoped to one concern.
- Include tests for behavior changes.
- Do not commit checkpoints, adapters, raw datasets, caches, or local virtual environments.
- Document new training flags and data assumptions.
- Mention any dataset license or safety implications.

## Data Contributions

Do not submit data unless you have the right to redistribute it. Prefer scripts
that download public datasets from their original source and record their
licenses in `docs/DATASETS.md`.

## License

By contributing, you agree that your contribution is licensed under the Apache
License 2.0.
