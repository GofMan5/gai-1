from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

USER_LABEL = "\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c"
ASSISTANT_LABEL = "\u0410\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442"


@dataclass(frozen=True)
class SupervisedTokens:
    tokens: list[int]
    train_mask: list[bool]


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    data_path = Path(path)
    rows: list[dict[str, object]] = []
    with data_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {data_path} line {line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {data_path}")
    return rows


def read_jsonl_texts(path: str | Path, field: str = "text") -> list[str]:
    data_path = Path(path)
    texts: list[str] = []
    for line_no, row in enumerate(_read_jsonl(data_path), start=1):
        if field not in row:
            raise ValueError(f"Missing field '{field}' in {data_path} line {line_no}")
        texts.append(str(row[field]))
    if not texts:
        raise ValueError(f"No training texts found in {data_path}")
    return texts


def format_sft_record(row: dict[str, object]) -> str:
    if "text" in row:
        return str(row["text"])
    if "prompt" in row and "response" in row:
        return f"{USER_LABEL}: {row['prompt']}\n{ASSISTANT_LABEL}: {row['response']}"
    if "instruction" in row and "output" in row:
        prompt = str(row["instruction"]).strip()
        extra = str(row.get("input", "")).strip()
        if extra:
            prompt = f"{prompt}\n{extra}"
        return f"{USER_LABEL}: {prompt}\n{ASSISTANT_LABEL}: {row['output']}"
    if "messages" in row and isinstance(row["messages"], list):
        parts: list[str] = []
        for message in row["messages"]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            label = ASSISTANT_LABEL if role == "assistant" else USER_LABEL
            parts.append(f"{label}: {content}")
        return "\n".join(parts)
    raise ValueError("SFT record must contain text, prompt/response, instruction/output, or messages")


def read_jsonl_formatted(path: str | Path) -> list[str]:
    return [format_sft_record(row) for row in _read_jsonl(path)]


def _prompt_response_from_record(row: dict[str, object]) -> tuple[str, str]:
    if "prompt" in row and "response" in row:
        return str(row["prompt"]).strip(), str(row["response"]).strip()
    if "instruction" in row and "output" in row:
        prompt = str(row["instruction"]).strip()
        extra = str(row.get("input", "")).strip()
        if extra:
            prompt = f"{prompt}\n{extra}"
        return prompt, str(row["output"]).strip()
    if "messages" in row and isinstance(row["messages"], list):
        messages = [message for message in row["messages"] if isinstance(message, dict)]
        for idx in range(len(messages) - 1, -1, -1):
            if str(messages[idx].get("role", "")) == "assistant":
                prompt_parts: list[str] = []
                for message in messages[:idx]:
                    role = str(message.get("role", "user"))
                    label = ASSISTANT_LABEL if role == "assistant" else USER_LABEL
                    prompt_parts.append(f"{label}: {message.get('content', '')}")
                return "\n".join(prompt_parts), str(messages[idx].get("content", "")).strip()
    if "text" in row:
        return "", str(row["text"]).strip()
    raise ValueError("SFT record must contain prompt/response, instruction/output, messages, or text")


def format_chat_prompt(prompt: str) -> str:
    return f"{USER_LABEL}: {prompt.strip()}\n{ASSISTANT_LABEL}:"


def _append_segment(
    tokens: list[int],
    train_mask: list[bool],
    tokenizer: object,
    text: str,
    train: bool,
    add_bos: bool = False,
    add_eos: bool = False,
) -> None:
    ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    tokens.extend(ids)
    train_mask.extend([train] * len(ids))


def _prompt_response_tokens(row: dict[str, object], tokenizer: object) -> SupervisedTokens | None:
    pair = _prompt_response_from_record(row)
    if pair is None:
        return None
    prompt, response = pair
    if not response:
        return None
    prompt_text = format_chat_prompt(prompt) if prompt else ""
    prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
    response_prefix = " " if prompt_text and not response.startswith(("\n", " ")) else ""
    response_ids = tokenizer.encode(response_prefix + response, add_eos=True)
    return SupervisedTokens(prompt_ids + response_ids, ([False] * len(prompt_ids)) + ([True] * len(response_ids)))


def _messages_tokens(row: dict[str, object], tokenizer: object) -> SupervisedTokens | None:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        return None
    messages = [message for message in raw_messages if isinstance(message, dict)]
    if not messages:
        return None
    tokens: list[int] = []
    train_mask: list[bool] = []
    _append_segment(tokens, train_mask, tokenizer, "", train=False, add_bos=True)
    last_role = ""
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        label = ASSISTANT_LABEL if role == "assistant" else USER_LABEL
        separator = "\n" if len(tokens) > 1 else ""
        if role == "assistant":
            _append_segment(tokens, train_mask, tokenizer, f"{separator}{label}:", train=False)
            content_prefix = " " if not content.startswith(("\n", " ")) else ""
            _append_segment(tokens, train_mask, tokenizer, content_prefix + content, train=True)
        else:
            _append_segment(tokens, train_mask, tokenizer, f"{separator}{label}: {content}", train=False)
        last_role = role
    if len(tokens) < 2 or not any(train_mask):
        return None
    eos_id = int(getattr(tokenizer, "eos_id", 2))
    tokens.append(eos_id)
    train_mask.append(last_role == "assistant")
    return SupervisedTokens(tokens, train_mask)


def _chunk_supervised_tokens(
    tokens: list[int],
    train_mask: list[bool],
    block_size: int,
    pad_id: int,
    stride: int | None = None,
    min_supervised_tokens: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if len(tokens) != len(train_mask):
        raise ValueError("tokens and train_mask length mismatch")
    if len(tokens) < 2:
        return []
    if block_size < 2:
        raise ValueError("block_size must be at least 2 for SFT")
    step = block_size if stride is None else int(stride)
    if step < 1 or step > block_size:
        raise ValueError("stride must be between 1 and block_size")
    items: list[tuple[torch.Tensor, torch.Tensor]] = []
    max_start = max(0, len(tokens) - 2)
    starts = list(range(0, max_start + 1, step))
    tail_start = max(0, len(tokens) - block_size - 1)
    if tail_start not in starts:
        starts.append(tail_start)
    starts = sorted(set(starts))
    for start in starts:
        chunk_tokens = tokens[start : start + block_size + 1]
        chunk_mask = train_mask[start : start + block_size + 1]
        if len(chunk_tokens) < 2:
            continue
        x_ids = chunk_tokens[:-1]
        y_ids = chunk_tokens[1:]
        y_mask = chunk_mask[1:]
        labels = [target if should_train else -100 for target, should_train in zip(y_ids, y_mask)]
        supervised = sum(1 for label in labels if label != -100)
        if supervised < min_supervised_tokens:
            continue
        if len(x_ids) < block_size:
            pad_count = block_size - len(x_ids)
            x_ids.extend([pad_id] * pad_count)
            labels.extend([-100] * pad_count)
        items.append((torch.tensor(x_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))
    return items


class SFTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: object,
        block_size: int,
        stride: int | None = None,
        min_supervised_tokens: int = 1,
    ) -> None:
        self.block_size = block_size
        self.stride = stride
        self.min_supervised_tokens = min_supervised_tokens
        self.pad_id = int(getattr(tokenizer, "pad_id", 0))
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.records = 0
        self.supervised_tokens = 0
        self.ignored_tokens = 0
        for row in _read_jsonl(path):
            self.records += 1
            encoded = _messages_tokens(row, tokenizer) if "messages" in row else _prompt_response_tokens(row, tokenizer)
            if encoded is None:
                continue
            chunks = _chunk_supervised_tokens(
                encoded.tokens,
                encoded.train_mask,
                block_size,
                self.pad_id,
                stride=stride,
                min_supervised_tokens=min_supervised_tokens,
            )
            for _x, labels in chunks:
                self.supervised_tokens += int((labels != -100).sum().item())
                self.ignored_tokens += int((labels == -100).sum().item())
            self.items.extend(chunks)
        if not self.items:
            raise ValueError(f"No usable SFT records found in {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[index]


class PackedTextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: object,
        block_size: int,
        field: str = "text",
    ) -> None:
        try:
            texts = read_jsonl_texts(path, field=field)
        except ValueError as exc:
            if "Missing field" not in str(exc):
                raise
            texts = read_jsonl_formatted(path)
        stream: list[int] = []
        for text in texts:
            stream.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
            stream.append(tokenizer.eot_id)
        if len(stream) < block_size + 1:
            raise ValueError(f"Dataset is too small: need at least {block_size + 1} tokens, got {len(stream)}")
        self.tokens = torch.tensor(stream, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(1, self.tokens.numel() - self.block_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class StreamingPackedTextDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: object,
        block_size: int,
        field: str = "text",
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.field = field
        self.eot_id = int(getattr(tokenizer, "eot_id", 3))

    def _row_text(self, row: dict[str, object]) -> str:
        if self.field in row:
            return str(row[self.field])
        return format_sft_record(row)

    def _iter_texts(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                if worker_count > 1 and line_no % worker_count != worker_id:
                    continue
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                text = self._row_text(row).strip()
                if text:
                    yield text

    def __iter__(self):
        buffer: list[int] = []
        for text in self._iter_texts():
            buffer.extend(self.tokenizer.encode(text, add_bos=True, add_eos=True))
            buffer.append(self.eot_id)
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[: self.block_size + 1]
                del buffer[: self.block_size]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
