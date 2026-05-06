# Release Checklist

Use this checklist before tagging or publishing a GitHub release.

## Code

- `python -m pytest -q` passes.
- Package build passes with `python -m build`.
- CI is green on `main`.
- `CHANGELOG.md` has the release notes.
- `pyproject.toml` version matches the tag.

## Training Artifacts

- No `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, `.bin`, raw datasets, caches, or virtual environments are committed.
- If a checkpoint is published separately, include:
  - model card;
  - base commit hash;
  - training config;
  - dataset summary and licenses;
  - eval results;
  - SHA256 hashes.

## Data And Safety

- Dataset licenses are reviewed in `docs/DATASETS.md`.
- Safety limitations are documented in `DISCLAIMER.md` and `MODEL_CARD.md`.
- Eval gates are run on release checkpoints.
- Known failure modes are listed in release notes.
- Long-context claims include explicit trained/tested context lengths and eval
  results. Do not call a checkpoint 128k-ready based only on RoPE scaling.

## GitHub

- Repository visibility is public.
- Issues are enabled.
- Security advisories are enabled if possible.
- Release artifacts are attached intentionally, never by accident.
