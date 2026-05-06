from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_serve_module():
    spec = importlib.util.spec_from_file_location("gai1_serve_for_tests", ROOT / "scripts" / "serve.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_stop_accepts_openai_shapes() -> None:
    serve = load_serve_module()

    assert serve.parse_stop(None) == ()
    assert serve.parse_stop("STOP") == ("STOP",)
    assert serve.parse_stop(["A", "B"]) == ("A", "B")
    with pytest.raises(ValueError):
        serve.parse_stop([1])


def test_chat_prompt_from_messages_uses_ru_labels() -> None:
    serve = load_serve_module()

    prompt = serve.chat_prompt_from_messages(
        [
            {"role": "system", "content": "Отвечай кратко"},
            {"role": "user", "content": "Привет"},
        ]
    )

    assert "Пользователь: Привет" in prompt
    assert prompt.endswith("Ассистент:")


def test_service_response_schema() -> None:
    serve = load_serve_module()
    service = object.__new__(serve.GAIService)
    service.model = object()
    service.tokenizer = object()

    class Result:
        text = "ответ"
        finish_reason = "stop"
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    def fake_generate_text(_model, _tokenizer, prompt_text, config):
        assert "Ассистент:" in prompt_text
        assert config.max_new_tokens == 7
        return Result()

    old_generate_text = serve.generate_text
    serve.generate_text = fake_generate_text
    try:
        response = service.complete(
            {
                "model": "gai-1-test",
                "messages": [{"role": "user", "content": "Привет"}],
                "max_tokens": 7,
                "temperature": 0.1,
            }
        )
    finally:
        serve.generate_text = old_generate_text

    assert response["object"] == "chat.completion"
    assert response["model"] == "gai-1-test"
    assert response["choices"][0]["message"]["content"] == "ответ"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 5


def test_service_rejects_streaming() -> None:
    serve = load_serve_module()
    service = SimpleNamespace()
    complete = serve.GAIService.complete

    with pytest.raises(ValueError, match="stream"):
        complete(service, {"stream": True, "messages": [{"role": "user", "content": "x"}]})
