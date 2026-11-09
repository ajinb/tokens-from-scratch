"""Byte-pair-encoding tokenizer training and encoding, from scratch. Stdlib only."""

from __future__ import annotations

from collections import Counter

Pair = tuple[int, int]


def _pairs(ids: list[int]) -> Counter[Pair]:
    return Counter(zip(ids, ids[1:], strict=False))


def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text: str, vocab_size: int) -> dict[Pair, int]:
    """Repeatedly merge the most common adjacent pair. Returns merges in the order learned."""
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256, one id for each byte")
    ids = list(text.encode("utf-8"))
    merges: dict[Pair, int] = {}
    for new_id in range(256, vocab_size):
        counts = _pairs(ids)
        if not counts:
            break
        pair, count = counts.most_common(1)[0]
        if count < 2:
            break
        merges[pair] = new_id
        ids = _merge(ids, pair, new_id)
    return merges


def encode(text: str, merges: dict[Pair, int]) -> list[int]:
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        # Apply the earliest-learned merge available, the order training created them in.
        pair = min(set(zip(ids, ids[1:], strict=False)), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = _merge(ids, pair, merges[pair])
    return ids


def decode(ids: list[int], merges: dict[Pair, int]) -> str:
    vocab = {i: bytes([i]) for i in range(256)}
    for (left, right), new_id in merges.items():
        vocab[new_id] = vocab[left] + vocab[right]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
