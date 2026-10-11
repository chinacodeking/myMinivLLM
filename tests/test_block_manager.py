from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence


def make_sequence(token_ids, block_size=4):
    return Sequence(token_ids=token_ids, block_size=block_size)


def release_ready_block(manager, token_ids):
    seq = make_sequence(
        token_ids,
        block_size=manager.block_size,
    )
    manager.allocate(seq)
    seq.advance_computed_tokens(len(token_ids))
    manager.mark_computed(seq)
    block_id = seq.block_table[0]
    manager.deallocate(seq)
    return block_id


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


def test_uncomputed_block_is_not_exposed_as_cache_hit():
    manager = BlockManager(num_blocks=2, block_size=4)
    first = make_sequence([1, 2, 3, 4])
    second = make_sequence([1, 2, 3, 4])

    manager.allocate(first)
    manager.allocate(second)

    assert first.block_table != second.block_table
    assert second.num_cached_tokens == 0


def test_mark_computed_publishes_only_complete_blocks():
    manager = BlockManager(num_blocks=2, block_size=4)
    seq = make_sequence([1, 2, 3, 4, 5, 6])
    manager.allocate(seq)

    seq.advance_computed_tokens(3)
    manager.mark_computed(seq)

    assert manager.blocks[seq.block_table[0]].hash == -1
    assert manager.blocks[seq.block_table[1]].hash == -1

    seq.advance_computed_tokens(1)
    manager.mark_computed(seq)

    assert manager.blocks[seq.block_table[0]].hash != -1
    assert manager.blocks[seq.block_table[1]].hash == -1


def test_identical_active_prefix_reuses_the_same_block():
    manager = BlockManager(num_blocks=1, block_size=4)
    first = make_sequence([1, 2, 3, 4])
    second = make_sequence([1, 2, 3, 4])

    manager.allocate(first)
    first.advance_computed_tokens(4)
    manager.mark_computed(first)

    assert len(manager.free_block_ids) == 0
    assert manager.can_allocate(second)

    manager.allocate(second)

    shared_block_id = first.block_table[0]

    assert second.block_table == [shared_block_id]
    assert second.num_cached_tokens == 4
    assert manager.blocks[shared_block_id].ref_count == 2
    assert manager.used_block_ids == {shared_block_id}
    assert len(manager.free_block_ids) == 0


def test_shared_block_is_cached_after_last_reference():
    manager = BlockManager(num_blocks=2, block_size=4)
    first = make_sequence([1, 2, 3, 4])
    second = make_sequence([1, 2, 3, 4])

    manager.allocate(first)
    first.advance_computed_tokens(4)
    manager.mark_computed(first)
    manager.allocate(second)
    shared_block_id = first.block_table[0]

    manager.deallocate(first)

    assert manager.blocks[shared_block_id].ref_count == 1
    assert shared_block_id in manager.used_block_ids
    assert second.block_table == [shared_block_id]

    manager.deallocate(second)

    assert manager.blocks[shared_block_id].ref_count == 0
    assert shared_block_id not in manager.used_block_ids
    assert shared_block_id not in manager.free_block_ids
    assert shared_block_id in manager.cached_block_ids


def test_released_computed_block_can_be_reused():
    manager = BlockManager(num_blocks=1, block_size=4)
    cached_block_id = release_ready_block(
        manager,
        [1, 2, 3, 4],
    )

    assert list(manager.cached_block_ids) == [
        cached_block_id
    ]
    assert len(manager.free_block_ids) == 0

    second = make_sequence([1, 2, 3, 4])

    assert manager.can_allocate(second)
    manager.allocate(second)

    assert second.block_table == [cached_block_id]
    assert second.num_cached_tokens == 4
    assert manager.blocks[cached_block_id].ref_count == 1
    assert cached_block_id in manager.used_block_ids
    assert cached_block_id not in manager.cached_block_ids


def test_lru_evicts_oldest_unreferenced_block():
    manager = BlockManager(num_blocks=2, block_size=4)
    oldest_block_id = release_ready_block(
        manager,
        [1, 2, 3, 4],
    )
    newest_block_id = release_ready_block(
        manager,
        [5, 6, 7, 8],
    )

    assert list(manager.cached_block_ids) == [
        oldest_block_id,
        newest_block_id,
    ]
    assert len(manager.free_block_ids) == 0

    replacement = make_sequence([9, 10, 11, 12])
    assert manager.can_allocate(replacement)
    manager.allocate(replacement)

    assert replacement.block_table == [
        oldest_block_id
    ]
    assert list(manager.cached_block_ids) == [
        newest_block_id
    ]
    assert manager.blocks[oldest_block_id].hash == -1


def test_cache_hit_refreshes_lru_order():
    manager = BlockManager(num_blocks=2, block_size=4)
    first_block_id = release_ready_block(
        manager,
        [1, 2, 3, 4],
    )
    second_block_id = release_ready_block(
        manager,
        [5, 6, 7, 8],
    )

    reused = make_sequence([1, 2, 3, 4])
    manager.allocate(reused)
    manager.deallocate(reused)

    assert list(manager.cached_block_ids) == [
        second_block_id,
        first_block_id,
    ]

    replacement = make_sequence([9, 10, 11, 12])
    manager.allocate(replacement)

    assert replacement.block_table == [
        second_block_id
    ]
    assert list(manager.cached_block_ids) == [
        first_block_id
    ]


def test_append_can_evict_cached_block_at_boundary():
    manager = BlockManager(num_blocks=2, block_size=4)
    cached_block_id = release_ready_block(
        manager,
        [1, 2, 3, 4],
    )

    active = make_sequence([5, 6, 7, 8])
    manager.allocate(active)
    active.advance_computed_tokens(4)
    manager.mark_computed(active)

    active.append_token(9)

    assert manager.can_append(active)
    manager.append(active)

    assert active.block_table[-1] == cached_block_id
    assert cached_block_id not in manager.cached_block_ids
    assert cached_block_id in manager.used_block_ids


def test_append_allocates_only_when_crossing_block_boundary():
    manager = BlockManager(num_blocks=3, block_size=4)
    seq = make_sequence([1, 2, 3, 4])
    manager.allocate(seq)

    seq.advance_computed_tokens(4)
    manager.mark_computed(seq)
    original_block_id = seq.block_table[0]

    seq.append_token(5)
    assert manager.can_append(seq)
    manager.append(seq)

    assert len(seq.block_table) == 2
    assert seq.block_table[0] == original_block_id

    block_table_after_boundary = list(seq.block_table)

    seq.advance_computed_tokens(1)
    seq.append_token(6)
    assert manager.can_append(seq)
    manager.append(seq)

    assert seq.block_table == block_table_after_boundary


def test_can_append_detects_exhaustion_at_block_boundary():
    manager = BlockManager(num_blocks=1, block_size=4)
    seq = make_sequence([1, 2, 3, 4])
    manager.allocate(seq)

    seq.advance_computed_tokens(4)
    manager.mark_computed(seq)
    seq.append_token(5)

    assert not manager.can_append(seq)
