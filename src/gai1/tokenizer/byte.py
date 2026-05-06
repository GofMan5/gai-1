from __future__ import annotations


class ByteTokenizer:
    """UTF-8 byte tokenizer. It is crude, but it never loses Russian text."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    eot_id = 3
    byte_offset = 4
    vocab_size = 260

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend(byte + self.byte_offset for byte in text.encode("utf-8"))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        payload = bytearray()
        for token in tokens:
            if token >= self.byte_offset:
                payload.append(token - self.byte_offset)
        return payload.decode("utf-8", errors="replace")

