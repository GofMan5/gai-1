from __future__ import annotations

import json

from gai1.data import ASSISTANT_LABEL, SFTDataset, StreamingPackedTextDataset, USER_LABEL, format_chat_prompt
from gai1.tokenizer import ByteTokenizer


def label_text(tokenizer: ByteTokenizer, labels) -> str:
    return tokenizer.decode([int(token) for token in labels.tolist() if int(token) >= 0])


def test_chat_prompt_labels_are_valid_unicode() -> None:
    prompt = "\u043f\u0440\u0438\u0432\u0435\u0442"
    assert format_chat_prompt(prompt) == f"{USER_LABEL}: {prompt}\n{ASSISTANT_LABEL}:"


def test_sft_dataset_masks_user_prompt(tmp_path) -> None:
    data_path = tmp_path / "sft.jsonl"
    data_path.write_text(
        json.dumps(
            {"prompt": "\u041f\u0440\u0438\u0432\u0435\u0442", "response": "\u0417\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = ByteTokenizer()
    dataset = SFTDataset(data_path, tokenizer=tokenizer, block_size=128)
    x, y = dataset[0]
    trained_text = label_text(tokenizer, y)

    assert x.shape == y.shape
    assert (y == -100).any()
    assert (y != -100).any()
    assert "\u0417\u0434\u0440\u0430\u0432" in trained_text
    assert "\u041f\u0440\u0438\u0432\u0435\u0442" not in trained_text


def test_sft_dataset_trains_all_assistant_spans_in_messages(tmp_path) -> None:
    data_path = tmp_path / "messages.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "\u0432\u043e\u043f\u0440\u043e\u0441 1"},
            {"role": "assistant", "content": "\u043e\u0442\u0432\u0435\u0442 1"},
            {"role": "user", "content": "\u0432\u043e\u043f\u0440\u043e\u0441 2"},
            {"role": "assistant", "content": "\u043e\u0442\u0432\u0435\u0442 2"},
        ]
    }
    data_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    tokenizer = ByteTokenizer()
    dataset = SFTDataset(data_path, tokenizer=tokenizer, block_size=160)
    trained_text = "".join(label_text(tokenizer, labels) for _x, labels in dataset)

    assert "\u043e\u0442\u0432\u0435\u0442 1" in trained_text
    assert "\u043e\u0442\u0432\u0435\u0442 2" in trained_text
    assert "\u0432\u043e\u043f\u0440\u043e\u0441 1" not in trained_text
    assert "\u0432\u043e\u043f\u0440\u043e\u0441 2" not in trained_text


def test_sft_dataset_splits_long_records_into_windows(tmp_path) -> None:
    data_path = tmp_path / "long.jsonl"
    response = " ".join(["\u0441\u043b\u043e\u0432\u043e"] * 80)
    data_path.write_text(
        json.dumps({"prompt": "\u0441\u043a\u0430\u0436\u0438 \u0434\u043b\u0438\u043d\u043d\u043e", "response": response}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    dataset = SFTDataset(data_path, tokenizer=ByteTokenizer(), block_size=64)

    assert len(dataset) > 1
    assert all((labels != -100).any() for _x, labels in dataset)
    assert dataset.supervised_tokens > 0
    assert dataset.ignored_tokens > 0


def test_sft_dataset_stride_keeps_tail_supervision(tmp_path) -> None:
    data_path = tmp_path / "tail.jsonl"
    response = " ".join(["\u0434\u043e\u043b\u0433\u043e"] * 40) + " \u0424\u0418\u041d\u0410\u041b"
    data_path.write_text(
        json.dumps({"prompt": "\u043e\u0442\u0432\u0435\u0442\u044c", "response": response}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tokenizer = ByteTokenizer()
    dataset = SFTDataset(data_path, tokenizer=tokenizer, block_size=64, stride=32)
    trained_text = "".join(label_text(tokenizer, labels) for _x, labels in dataset)

    assert len(dataset) > 1
    assert "\u0424\u0418\u041d\u0410\u041b" in trained_text


def test_streaming_packed_text_dataset_reads_without_materializing(tmp_path) -> None:
    data_path = tmp_path / "pretrain.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"text": "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440. \u042d\u0442\u043e \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442 \u0434\u043b\u044f \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f."},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"text": "\u0412\u0442\u043e\u0440\u0430\u044f \u0441\u0442\u0440\u043e\u043a\u0430 \u0442\u043e\u0436\u0435 \u0434\u043e\u043b\u0436\u043d\u0430 \u043f\u043e\u043f\u0430\u0441\u0442\u044c \u0432 \u043f\u043e\u0442\u043e\u043a."},
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = StreamingPackedTextDataset(data_path, tokenizer=ByteTokenizer(), block_size=16)
    x, y = next(iter(dataset))
    assert x.shape == y.shape
    assert x.numel() == 16
