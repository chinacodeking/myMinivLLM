import pytest

from myvllm.engine.sequence import Sequence, SequenceStatus


def make_sequence(token_ids, block_size=4):
    return Sequence(token_ids=token_ids, block_size=block_size)


def test_sequence_copies_input_tokens():
    token_ids = [1, 2, 3]

    seq = make_sequence(token_ids)
    token_ids.append(4)

    assert seq.token_ids == [1, 2, 3]


def test_sequence_starts_with_prompt_state():
    seq = make_sequence([1, 2, 3])

    assert seq.status is SequenceStatus.WAITING
    assert seq.num_tokens == 3
    assert seq.num_prompt_tokens == 3
    assert seq.num_completion_tokens == 0
    assert seq.num_cached_tokens == 0
    assert seq.block_table == []
    assert seq.last_token == 3


def test_block_accounting_with_partial_last_block():
    seq = make_sequence([1, 2, 3, 4, 5])

    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens == 1
    assert seq.block(0) == [1, 2, 3, 4]
    assert seq.block(1) == [5]


def test_append_token_updates_completion_state():
    seq = make_sequence([1, 2])

    seq.append_token(9)

    assert seq.num_tokens == 3
    assert seq.num_prompt_tokens == 2
    assert seq.num_completion_tokens == 1
    assert seq.prompt_token_ids == [1, 2]
    assert seq.completion_token_ids == [9]
    assert seq.last_token == 9


def test_block_rejects_out_of_range_indices():
    seq = make_sequence([1, 2, 3])

    with pytest.raises(AssertionError):
        seq.block(-1)

    with pytest.raises(AssertionError):
        seq.block(1)


@pytest.mark.parametrize(
    ("num_cached_tokens", "expected_blocks"),
    [
        (0, 0),
        (1, 1),
        (4, 1),
        (5, 2),
    ],
)
def test_cached_block_accounting(num_cached_tokens, expected_blocks):
    seq = make_sequence([1, 2, 3, 4, 5])
    seq.num_cached_tokens = num_cached_tokens

    assert seq.num_cached_blocks == expected_blocks
