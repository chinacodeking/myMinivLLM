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