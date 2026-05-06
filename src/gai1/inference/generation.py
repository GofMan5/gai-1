from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = None
    repetition_penalty: float = 1.0
    stop_token_ids: set[int] = field(default_factory=set)
    stop_strings: tuple[str, ...] = ()
    return_full_text: bool = False
    use_cache: bool = True


@dataclass(frozen=True)
class GenerationResult:
    text: str
    full_text: str
    generated_ids: list[int]
    finish_reason: str
    usage: dict[str, int]


def default_stop_token_ids(tokenizer: object) -> set[int]:
    ids: set[int] = set()
    for name in ("eos_id", "eot_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            ids.add(int(value))
    return ids


def trim_stop_strings(text: str, stop_strings: tuple[str, ...]) -> tuple[str, bool]:
    if not stop_strings:
        return text, False
    first_index: int | None = None
    for stop in stop_strings:
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0 and (first_index is None or index < first_index):
            first_index = index
    if first_index is None:
        return text, False
    return text[:first_index], True


@torch.no_grad()
def generate_text(model, tokenizer, prompt_text: str, config: GenerationConfig) -> GenerationResult:
    prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    stop_ids = set(config.stop_token_ids) or default_stop_token_ids(tokenizer)
    out = model.generate(
        idx,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        stop_token_ids=stop_ids,
        use_cache=config.use_cache,
    )
    output_ids = out[0].tolist()
    generated_ids = output_ids[len(prompt_ids) :]
    finish_reason = "length"
    if generated_ids and generated_ids[-1] in stop_ids:
        finish_reason = "stop"
    generated_text = tokenizer.decode(generated_ids).strip()
    generated_text, stopped_by_string = trim_stop_strings(generated_text, config.stop_strings)
    if stopped_by_string:
        finish_reason = "stop"
    full_text = tokenizer.decode(output_ids)
    text = full_text if config.return_full_text else generated_text
    return GenerationResult(
        text=text,
        full_text=full_text,
        generated_ids=generated_ids,
        finish_reason=finish_reason,
        usage={
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(prompt_ids) + len(generated_ids),
        },
    )
