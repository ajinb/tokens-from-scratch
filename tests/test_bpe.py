import random

import pytest

import bpe

TEXT = "the cat sat on the mat. the cat ate the rat. " * 40


def test_round_trip_on_training_text():
    merges = bpe.train(TEXT, 300)
    assert bpe.decode(bpe.encode(TEXT, merges), merges) == TEXT


@pytest.mark.parametrize("seed", range(20))
def test_round_trip_on_arbitrary_unicode(seed):
    merges = bpe.train(TEXT, 300)
    rng = random.Random(seed)
    text = "".join(
        chr(rng.choice([rng.randint(32, 126), rng.randint(0x400, 0x4FF), 0x1F600]))
        for _ in range(200)
    )
    assert bpe.decode(bpe.encode(text, merges), merges) == text


def test_merging_compresses_what_it_was_trained_on():
    merges = bpe.train(TEXT, 320)
    assert len(bpe.encode(TEXT, merges)) < len(TEXT.encode()) / 4


def test_unseen_content_barely_compresses():
    merges = bpe.train(TEXT, 320)
    unseen = "7f3a9c1e4b82d05f" * 3
    assert len(bpe.encode(unseen, merges)) == len(unseen)


def test_first_merge_is_the_most_common_pair():
    merges = bpe.train("ababababab xy", 257)
    assert list(merges) == [(ord("a"), ord("b"))]


def test_training_stops_when_nothing_repeats():
    assert bpe.train("abc", 1000) == {}


def test_rejects_vocab_smaller_than_bytes():
    with pytest.raises(ValueError):
        bpe.train("abc", 10)
