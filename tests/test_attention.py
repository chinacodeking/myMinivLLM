from unittest.mock import MagicMock

import torch

from myvllm.layers import attention as attention_module
from myvllm.utils.context import reset_context, set_context
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


def test_attention_uses_paged_prefill_when_kv_context_is_longer(
    monkeypatch,
):
    query = torch.zeros(
        (2, 1, 1),
        dtype=torch.float32,
    )
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)

    layer = attention_module.Attention(
        num_heads=1,
        head_dim=1,
        scale=1.0,
        num_kv_heads=1,
        block_size=2,
    )
    layer.k_cache = torch.zeros(
        (2, 2, 1, 1),
        dtype=torch.float32,
    )
    layer.v_cache = torch.zeros_like(layer.k_cache)

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
    block_tables = torch.tensor(
        [
            [1, 0],
        ],
        dtype=torch.int32,
    )

    flash_prefill = MagicMock(return_value=torch.zeros_like(query))
    paged_prefill = MagicMock(return_value=torch.zeros_like(query))

    monkeypatch.setattr(
        attention_module,
        "flash_attention_prefill",
        flash_prefill,
    )
    monkeypatch.setattr(
        attention_module,
        "paged_attention_prefill_reference",
        paged_prefill,
    )

    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        block_tables=block_tables,
        positions=positions,
    )

    try:
        output = layer(
            query,
            key,
            value,
        )
    finally:
        reset_context()

    paged_prefill.assert_called_once()
    flash_prefill.assert_not_called()

    arguments = paged_prefill.call_args.kwargs

    assert arguments["query"] is query
    assert arguments["k_cache"] is layer.k_cache
    assert arguments["v_cache"] is layer.v_cache
    assert arguments["block_tables"] is block_tables
    assert arguments["cu_seqlens_q"] is cu_seqlens_q
    assert arguments["cu_seqlens_k"] is cu_seqlens_k
    assert arguments["positions"] is positions
    assert arguments["scale"] == 1.0
    assert arguments["block_size"] == 2

    assert output.shape == (
        2,
        1,
    )


def test_attention_keeps_flash_prefill_for_full_prompt(
    monkeypatch,
):
    query = torch.zeros(
        (2, 1, 1),
        dtype=torch.float32,
    )
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)

    layer = attention_module.Attention(
        num_heads=1,
        head_dim=1,
        scale=1.0,
        num_kv_heads=1,
        block_size=2,
    )
    layer.k_cache = torch.zeros(
        (1, 2, 1, 1),
        dtype=torch.float32,
    )
    layer.v_cache = torch.zeros_like(layer.k_cache)

    cu_seqlens_q = torch.tensor(
        [0, 2],
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0, 2],
        dtype=torch.int32,
    )

    flash_prefill = MagicMock(return_value=torch.zeros_like(query))
    paged_prefill = MagicMock(return_value=torch.zeros_like(query))

    monkeypatch.setattr(
        attention_module,
        "flash_attention_prefill",
        flash_prefill,
    )
    monkeypatch.setattr(
        attention_module,
        "paged_attention_prefill_reference",
        paged_prefill,
    )

    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        positions=torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
    )

    try:
        output = layer(
            query,
            key,
            value,
        )
    finally:
        reset_context()

    flash_prefill.assert_called_once()
    paged_prefill.assert_not_called()

    assert output.shape == (
        2,
        1,
    )