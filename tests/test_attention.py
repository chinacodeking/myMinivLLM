import torch

from myvllm.layers import attention as attention_module


def test_gather_paged_kv_follows_logical_block_order():
    cache = torch.tensor(
        [
            [0.0, 1.0],
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
        ],
        dtype=torch.float32,
    ).reshape(
        4,
        2,
        1,
        1,
    )

    block_table = torch.tensor(
        [2, 0, 3],
        dtype=torch.int32,
    )

    gathered = attention_module.gather_paged_kv(
        cache=cache,
        block_table=block_table,
        context_len=5,
        block_size=2,
    )

    assert gathered.shape == (
        5,
        1,
        1,
    )
    assert gathered[:, 0, 0].tolist() == [
        20.0,
        21.0,
        0.0,
        1.0,
        30.0,
    ]
def test_paged_prefill_attends_to_prefix_with_absolute_causal_mask():
    query = torch.zeros(
        (2, 1, 1),
        dtype=torch.float32,
    )

    k_cache = torch.zeros(
        (2, 2, 1, 1),
        dtype=torch.float32,
    )

    v_cache = torch.tensor(
        [
            [5.0, 7.0],
            [1.0, 3.0],
        ],
        dtype=torch.float32,
    ).reshape(
        2,
        2,
        1,
        1,
    )

    block_tables = torch.tensor(
        [
            [1, 0],
        ],
        dtype=torch.int32,
    )

    cu_seqlens_q = torch.tensor(
        [0, 2],
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0, 4],
        dtype=torch.int32,
    )
    positions = torch.tensor(
        [2, 3],
        dtype=torch.long,
    )

    output = (
        attention_module.paged_attention_prefill_reference(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            positions=positions,
            scale=1.0,
            block_size=2,
        )
    )

    assert output.shape == (
        2,
        1,
        1,
    )
    torch.testing.assert_close(
        output[:, 0, 0],
        torch.tensor(
            [3.0, 4.0],
            dtype=torch.float32,
        ),
    )
def test_paged_prefill_handles_packed_sequences_and_gqa():
    query = torch.zeros(
        (3, 2, 1),
        dtype=torch.float32,
    )

    k_cache = torch.zeros(
        (3, 2, 1, 1),
        dtype=torch.float32,
    )

    v_cache = torch.tensor(
        [
            [5.0, 99.0],
            [10.0, 14.0],
            [1.0, 3.0],
        ],
        dtype=torch.float32,
    ).reshape(
        3,
        2,
        1,
        1,
    )

    block_tables = torch.tensor(
        [
            [2, 0],
            [1, -1],
        ],
        dtype=torch.int32,
    )

    cu_seqlens_q = torch.tensor(
        [0, 1, 3],
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0, 3, 5],
        dtype=torch.int32,
    )
    positions = torch.tensor(
        [2, 0, 1],
        dtype=torch.long,
    )

    output = (
        attention_module.paged_attention_prefill_reference(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            positions=positions,
            scale=1.0,
            block_size=2,
        )
    )

    assert output.shape == (
        3,
        2,
        1,
    )
    torch.testing.assert_close(
        output[:, :, 0],
        torch.tensor(
            [
                [3.0, 3.0],
                [10.0, 10.0],
                [12.0, 12.0],
            ],
            dtype=torch.float32,
        ),
    )