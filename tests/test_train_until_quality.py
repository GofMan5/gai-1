from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "gai1_train_until_quality_for_tests",
        ROOT / "scripts" / "train_until_quality.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_eval_failure_is_fatal_by_default() -> None:
    pipeline = load_pipeline_module()

    assert pipeline.eval_exit_code(3, allow_failed_eval=False) == 3


def test_eval_failure_can_be_allowed_explicitly() -> None:
    pipeline = load_pipeline_module()

    assert pipeline.eval_exit_code(3, allow_failed_eval=True) is None
    assert pipeline.eval_exit_code(0, allow_failed_eval=False) is None
