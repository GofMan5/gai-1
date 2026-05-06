from __future__ import annotations

from pathlib import Path


class BPETokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    eot_id = 3

    def __init__(self, path: str | Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install tokenizers first: pip install tokenizers") from exc
        self.path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.tokenizer.encode(text).ids)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, tokens: list[int]) -> str:
        payload = [token for token in tokens if token >= 4]
        return self.tokenizer.decode(payload, skip_special_tokens=True)


def train_bpe_tokenizer(
    texts: list[str],
    output_path: str | Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> None:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError("Install tokenizers first: pip install tokenizers") from exc

    special_tokens = ["<pad>", "<bos>", "<eos>", "<eot>", "<unk>"]
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[("<bos>", 1), ("<eos>", 2), ("<eot>", 3)],
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
