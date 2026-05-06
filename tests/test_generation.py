from __future__ import annotations

from types import SimpleNamespace

import torch

from gai1.inference import GenerationConfig, generate_text
from gai1.inference.generation import trim_stop_strings
from gai1.tokenizer import ByteTokenizer


class FixedModel:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens
        self.cfg = SimpleNamespace(block_size=128)
        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.param

    def generate(self, idx, max_new_tokens, **_kwargs):
        suffix = torch.tensor([self.tokens[:max_new_tokens]], dtype=torch.long, device=idx.device)
        return torch.cat((idx, suffix), dim=1)


def test_trim_stop_strings() -> None:
    text, stopped = trim_stop_strings("ответ\nПользователь: новый вопрос", ("\nПользователь:",))

    assert text == "ответ"
    assert stopped is True


def test_generate_text_returns_suffix_only_and_usage() -> None:
    tokenizer = ByteTokenizer()
    generated = tokenizer.encode(" ответ", add_eos=True)
    result = generate_text(FixedModel(generated), tokenizer, "Пользователь: привет\nАссистент:", GenerationConfig())

    assert "Пользователь" not in result.text
    assert "ответ" in result.text
    assert result.finish_reason == "stop"
    assert result.usage["completion_tokens"] == len(generated)
