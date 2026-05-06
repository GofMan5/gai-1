from __future__ import annotations

import json

from gai1.data import ASSISTANT_LABEL, SFTDataset, USER_LABEL, format_chat_prompt
from gai1.tokenizer import ByteTokenizer


def test_chat_prompt_labels_are_valid_unicode() -> None:
    assert format_chat_prompt("привет") == f"{USER_LABEL}: привет\n{ASSISTANT_LABEL}:"


def test_sft_dataset_masks_user_prompt(tmp_path) -> None:
    data_path = tmp_path / "sft.jsonl"
    data_path.write_text(json.dumps({"prompt": "Привет", "response": "Здравствуйте"}, ensure_ascii=False) + "\n", encoding="utf-8")
    tokenizer = ByteTokenizer()
    dataset = SFTDataset(data_path, tokenizer=tokenizer, block_size=64)
    x, y = dataset[0]
    assert x.shape == y.shape
    assert (y == -100).any()
    assert (y != -100).any()
