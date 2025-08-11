from __future__ import annotations

from typing import Dict, Iterable, List


class FeatureTokenizer:
    """Converts raw hex strings/fields to integer tokens.

    The debunk_v1 preset performs:
    - hex string -> per-byte int tokens (0..255)
    - bucketized ports (0..3)
    - bucketized packet length (0..7)
    This is a simplified placeholder; adapt as needed.
    """

    def __init__(self, preset: str = "debunk_v1") -> None:
        self.preset = preset

    def tokenize_hex(self, hex_str: str) -> List[int]:
        hex_str = hex_str.replace(" ", "").replace(":", "").replace("0x", "")
        if len(hex_str) % 2 == 1:
            hex_str = "0" + hex_str
        return [int(hex_str[i : i + 2], 16) for i in range(0, len(hex_str), 2)]

    def bucketize_port(self, port: int) -> int:
        if port < 1024:
            return 0
        if port < 49152:
            return 1
        if port < 65536:
            return 2
        return 3

    def bucketize_length(self, length: int) -> int:
        bins = [64, 128, 256, 512, 1024, 1500, 2000]
        for i, b in enumerate(bins):
            if length <= b:
                return i
        return len(bins)

    def encode(self, row: Dict) -> List[int]:
        payload_hex = str(row.get("payload_hex", ""))
        byte_tokens = self.tokenize_hex(payload_hex)
        src_port = int(row.get("src_port", 0))
        dst_port = int(row.get("dst_port", 0))
        length = int(row.get("length", len(byte_tokens)))
        # Offset meta tokens to avoid collision with 0..255 bytes.
        meta_base = 256
        src_tok = meta_base + self.bucketize_port(src_port)  # 256..259
        dst_tok = meta_base + 8 + self.bucketize_port(dst_port)  # 264..267
        len_tok = meta_base + 16 + self.bucketize_length(length)  # 272..
        tokens = [src_tok, dst_tok, len_tok] + byte_tokens
        return tokens


