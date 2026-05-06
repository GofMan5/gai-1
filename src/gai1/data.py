from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

USER_LABEL = "\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c"
ASSISTANT_LABEL = "\u0410\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442"


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


class SFTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path, tokenizer: object, block_size: int) -> None:
        self.block_size = block_size
        self.pad_id = int(getattr(tokenizer, "pad_id", 0))
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        for row in _read_jsonl(path):
            prompt, response = _prompt_response_from_record(row)
            if not response:
                continue
            prompt_text = format_chat_prompt(prompt) if prompt else ""
            prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
            response_prefix = " " if prompt_text and not response.startswith(("\n", " ")) else ""
            full_ids = prompt_ids + tokenizer.encode(response_prefix + response, add_eos=True)
            if len(full_ids) < 2:
                continue
            full_ids = full_ids[: block_size + 1]
            x_ids = full_ids[:-1]
            y_ids = full_ids[1:]
            ignore_count = min(max(0, len(prompt_ids) - 1), len(y_ids))
            labels = [-100] * ignore_count + y_ids[ignore_count:]
            if len(x_ids) < block_size:
                pad_count = block_size - len(x_ids)
                x_ids.extend([self.pad_id] * pad_count)
                labels.extend([-100] * pad_count)
            self.items.append((torch.tensor(x_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))
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
