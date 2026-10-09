from myvllm.engine.model_input import build_prefill_metadata
from myvllm.engine.scheduler import ScheduledSequence
from myvllm.engine.sequence import Sequence


def make_sequence(token_ids, block_size=4):
    return Sequence(
        token_ids=token_ids,
        block_size=block_size,
    )


def test_prefill_metadata_uses_computed_offset_and_chunk_size():
    seq = make_sequence(
        list(range(1, 11)),
        block_size=4,
    )
    seq.block_table = [7, 2, 9]
    seq.advance_computed_tokens(4)

    scheduled = [
        ScheduledSequence(
            sequence=seq,
            num_scheduled_tokens=4,
            is_prefill=True,
        )
    ]

    metadata = build_prefill_metadata(
        scheduled,
        block_size=4,
    )

    assert metadata.input_ids == [5, 6, 7, 8]
    assert metadata.positions == [4, 5, 6, 7]
    assert metadata.slot_mapping == [8, 9, 10, 11]

    assert metadata.seqlens_q == [4]
    assert metadata.seqlens_k == [8]
    assert metadata.cu_seqlens_q == [0, 4]
    assert metadata.cu_seqlens_k == [0, 8]

    assert metadata.block_tables == [[7, 2]]
def test_prefill_metadata_maps_chunk_across_block_boundary():
    seq = make_sequence(
        list(range(1, 11)),
        block_size=4,
    )
    seq.block_table = [7, 2, 9]
    seq.advance_computed_tokens(3)

    scheduled = [
        ScheduledSequence(
            sequence=seq,
            num_scheduled_tokens=3,
            is_prefill=True,
        )
    ]

    metadata = build_prefill_metadata(
        scheduled,
        block_size=4,
    )

    assert metadata.input_ids == [4, 5, 6]
    assert metadata.positions == [3, 4, 5]
    assert metadata.slot_mapping == [31, 8, 9]

    assert metadata.seqlens_q == [3]
    assert metadata.seqlens_k == [6]
    assert metadata.cu_seqlens_q == [0, 3]
    assert metadata.cu_seqlens_k == [0, 6]

    assert metadata.block_tables == [[7, 2]]
def test_prefill_metadata_flattens_multiple_sequences():
    first = make_sequence(
        [1, 2, 3, 4, 5, 6],
        block_size=4,
    )
    first.block_table = [3, 8]
    first.advance_computed_tokens(2)

    second = make_sequence(
        [11, 12, 13, 14],
        block_size=4,
    )
    second.block_table = [5]

    scheduled = [
        ScheduledSequence(
            sequence=first,
            num_scheduled_tokens=3,
            is_prefill=True,
        ),
        ScheduledSequence(
            sequence=second,
            num_scheduled_tokens=2,
            is_prefill=True,
        ),
    ]

    metadata = build_prefill_metadata(
        scheduled,
        block_size=4,
    )

    assert metadata.input_ids == [
        3,
        4,
        5,
        11,
        12,
    ]
    assert metadata.positions == [
        2,
        3,
        4,
        0,
        1,
    ]
    assert metadata.slot_mapping == [
        14,
        15,
        32,
        20,
        21,
    ]

    assert metadata.seqlens_q == [3, 2]
    assert metadata.seqlens_k == [5, 2]
    assert metadata.cu_seqlens_q == [0, 3, 5]
    assert metadata.cu_seqlens_k == [0, 5, 7]

    assert metadata.block_tables == [
        [3, 8],
        [5, -1],
    ]
def test_prefill_metadata_allows_warmup_without_block_table():
    seq = make_sequence(
        [1, 2, 3],
        block_size=4,
    )

    scheduled = [
        ScheduledSequence(
            sequence=seq,
            num_scheduled_tokens=3,
            is_prefill=True,
        )
    ]

    metadata = build_prefill_metadata(
        scheduled,
        block_size=4,
    )

    assert metadata.input_ids == [1, 2, 3]
    assert metadata.positions == [0, 1, 2]
    assert metadata.slot_mapping == [-1, -1, -1]

    assert metadata.seqlens_q == [3]
    assert metadata.seqlens_k == [3]
    assert metadata.cu_seqlens_q == [0, 3]
    assert metadata.cu_seqlens_k == [0, 3]

    assert metadata.block_tables == []