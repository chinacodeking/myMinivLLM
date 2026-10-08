from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence


def make_sequence(token_ids, block_size=4):
    return Sequence(token_ids=token_ids, block_size=block_size)


def test_can_allocate_checks_available_capacity():
    manager = BlockManager(num_blocks=2, block_size=4)

    sequence_that_fits = make_sequence([1, 2, 3, 4, 5])
    sequence_that_is_too_large = make_sequence(
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    assert sequence_that_fits.num_blocks == 2
    assert sequence_that_is_too_large.num_blocks == 3
    assert manager.can_allocate(sequence_that_fits)
    assert not manager.can_allocate(sequence_that_is_too_large)


def test_allocate_assigns_blocks_and_updates_pool():
    manager = BlockManager(num_blocks=3, block_size=4)
    seq = make_sequence([1, 2, 3, 4, 5])

    manager.allocate(seq)

    assert len(seq.block_table) == 2
    assert len(set(seq.block_table)) == 2
    assert set(seq.block_table) == manager.used_block_ids
    assert len(manager.free_block_ids) == 1

    for block_id in seq.block_table:
        assert manager.blocks[block_id].ref_count == 1


def test_deallocate_returns_blocks_to_pool():
    manager = BlockManager(num_blocks=3, block_size=4)
    seq = make_sequence([1, 2, 3, 4, 5])
    manager.allocate(seq)

    allocated_block_ids = set(seq.block_table)
    manager.deallocate(seq)

    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert manager.used_block_ids == set()
    assert set(manager.free_block_ids) == {0, 1, 2}

    for block_id in allocated_block_ids:
        assert manager.blocks[block_id].ref_count == 0


def test_identical_active_prefix_reuses_the_same_block():
    manager = BlockManager(num_blocks=2, block_size=4)
    first = make_sequence([1, 2, 3, 4])
    second = make_sequence([1, 2, 3, 4])

    manager.allocate(first)
    manager.allocate(second)

    shared_block_id = first.block_table[0]

    assert second.block_table == [shared_block_id]
    assert second.num_cached_tokens == 4
    assert manager.blocks[shared_block_id].ref_count == 2
    assert manager.used_block_ids == {shared_block_id}
    assert len(manager.free_block_ids) == 1


def test_shared_block_is_freed_after_last_reference():
    manager = BlockManager(num_blocks=2, block_size=4)
    first = make_sequence([1, 2, 3, 4])
    second = make_sequence([1, 2, 3, 4])

    manager.allocate(first)
    manager.allocate(second)
    shared_block_id = first.block_table[0]

    manager.deallocate(first)

    assert manager.blocks[shared_block_id].ref_count == 1
    assert shared_block_id in manager.used_block_ids
    assert second.block_table == [shared_block_id]

    manager.deallocate(second)

    assert manager.blocks[shared_block_id].ref_count == 0
    assert shared_block_id not in manager.used_block_ids
    assert shared_block_id in manager.free_block_ids


def test_append_allocates_only_when_crossing_block_boundary():
    manager = BlockManager(num_blocks=3, block_size=4)
    seq = make_sequence([1, 2, 3, 4])
    manager.allocate(seq)

    original_block_id = seq.block_table[0]

    seq.append_token(5)
    assert manager.can_append(seq)
    manager.append(seq)

    assert len(seq.block_table) == 2
    assert seq.block_table[0] == original_block_id

    block_table_after_boundary = list(seq.block_table)

    seq.append_token(6)
    assert manager.can_append(seq)
    manager.append(seq)

    assert seq.block_table == block_table_after_boundary


def test_can_append_detects_exhaustion_at_block_boundary():
    manager = BlockManager(num_blocks=1, block_size=4)
    seq = make_sequence([1, 2, 3, 4])
    manager.allocate(seq)

    seq.append_token(5)

    assert not manager.can_append(seq)
