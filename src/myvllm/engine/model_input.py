from dataclasses import dataclass

from myvllm.engine.scheduler import ScheduledSequence


@dataclass(frozen=True)
class PrefillMetadata:
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    seqlens_q: list[int]
    seqlens_k: list[int]
    cu_seqlens_q: list[int]
    cu_seqlens_k: list[int]
    block_tables: list[list[int]]


def build_prefill_metadata(
    scheduled: list[ScheduledSequence],
    block_size: int,
) -> PrefillMetadata:
    if not scheduled:
        raise ValueError("scheduled must not be empty")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    seqlens_q: list[int] = []
    seqlens_k: list[int] = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    visible_block_tables: list[list[int]] = []

    for item in scheduled:
        if not item.is_prefill:
            raise ValueError(
                "build_prefill_metadata only accepts prefill work"
            )

        seq = item.sequence
        if seq.block_size != block_size:
            raise ValueError(
                "Sequence block size does not match ModelRunner block size"
            )

        start = seq.num_computed_tokens
        end = start + item.num_scheduled_tokens

        if end > seq.num_tokens:
            raise ValueError(
                "Scheduled prefill tokens exceed known sequence tokens"
            )

        chunk_token_ids = seq.token_ids[start:end]
        input_ids.extend(chunk_token_ids)
        positions.extend(range(start, end))

        query_length = item.num_scheduled_tokens
        context_length = end

        seqlens_q.append(query_length)
        seqlens_k.append(context_length)
        cu_seqlens_q.append(
            cu_seqlens_q[-1] + query_length
        )
        cu_seqlens_k.append(
            cu_seqlens_k[-1] + context_length
        )

        num_context_blocks = (
            context_length + block_size - 1
        ) // block_size

        if not seq.block_table:
            slot_mapping.extend([-1] * query_length)
            visible_block_tables.append([])
            continue

        if len(seq.block_table) < num_context_blocks:
            raise ValueError(
                "Sequence block table cannot cover scheduled context"
            )

        visible_block_tables.append(
            seq.block_table[:num_context_blocks]
        )

        for position in range(start, end):
            logical_block = position // block_size
            block_offset = position % block_size
            physical_block = seq.block_table[logical_block]

            slot_mapping.append(
                physical_block * block_size + block_offset
            )

    if any(visible_block_tables):
        max_num_blocks = max(
            len(block_table)
            for block_table in visible_block_tables
        )
        block_tables = [
            block_table
            + [-1] * (max_num_blocks - len(block_table))
            for block_table in visible_block_tables
        ]
    else:
        block_tables = []

    return PrefillMetadata(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        seqlens_q=seqlens_q,
        seqlens_k=seqlens_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        block_tables=block_tables,
    )