import pickle

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


def test_computed_token_progress_starts_at_zero():
    seq = make_sequence([1, 2, 3])

    assert seq.num_computed_tokens == 0
    assert seq.num_uncomputed_tokens == 3


def test_advance_computed_tokens_tracks_partial_progress():
    seq = make_sequence([1, 2, 3, 4, 5])

    seq.advance_computed_tokens(2)

    assert seq.num_computed_tokens == 2
    assert seq.num_uncomputed_tokens == 3

    seq.advance_computed_tokens(3)

    assert seq.num_computed_tokens == 5
    assert seq.num_uncomputed_tokens == 0


def test_append_token_leaves_new_token_uncomputed():
    seq = make_sequence([1, 2])

    seq.advance_computed_tokens(2)
    seq.append_token(9)

    assert seq.num_tokens == 3
    assert seq.num_computed_tokens == 2
    assert seq.num_uncomputed_tokens == 1


@pytest.mark.parametrize("amount", [-1, 4])
def test_advance_computed_tokens_rejects_invalid_progress(amount):
    seq = make_sequence([1, 2, 3])

    with pytest.raises(ValueError):
        seq.advance_computed_tokens(amount)


def test_computed_token_progress_survives_pickle_round_trip():
    seq = make_sequence([1, 2, 3, 4, 5])
    seq.advance_computed_tokens(2)

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.block_size == 4
    assert restored.num_computed_tokens == 2
    assert restored.num_uncomputed_tokens == 3
