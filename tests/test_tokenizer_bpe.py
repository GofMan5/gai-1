from __future__ import annotations

from gai1.tokenizer import BPETokenizer, train_bpe_tokenizer


def test_bpe_tokenizer_roundtrip_ru(tmp_path) -> None:
    path = tmp_path / "tok.json"
    train_bpe_tokenizer(["Привет мир", "GAI-1 reasoning модель"], path, vocab_size=64, min_frequency=1)
    tokenizer = BPETokenizer(path)
    ids = tokenizer.encode("Привет мир", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert "Привет" in tokenizer.decode(ids)

