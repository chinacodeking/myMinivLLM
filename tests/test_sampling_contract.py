import torch

from myvllm.engine.model_runner import (
    select_sample_logits,
)
from myvllm.engine.scheduler import (
    ScheduledSequence,
)
from myvllm.engine.sequence import Sequence
from myvllm.layers import (
    embedding_head as embedding_head_module,
)
from myvllm.utils.context import (
    reset_context,
    set_context,
)


def test_real_lm_head_output_matches_sampling_selection(
    monkeypatch,
):
    monkeypatch.setattr(
        embedding_head_module.dist,
        "get_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        embedding_head_module.dist,
        "get_rank",
        lambda: 0,
    )

    head = embedding_head_module.ParallelLMHead(
        num_embeddings=3,
        embedding_dim=2,
    ).to(
        device="cpu",
        dtype=torch.float32,
    )

    with torch.no_grad():
        head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=torch.float32,
                device="cpu",
            )
        )

    intermediate = Sequence(
        token_ids=[1, 2, 3, 4],
        block_size=4,
    )
    complete = Sequence(
        token_ids=[5, 6, 7],
        block_size=4,
    )
    scheduled = [
        ScheduledSequence(
            sequence=intermediate,
            num_scheduled_tokens=2,
            is_prefill=True,
        ),
        ScheduledSequence(
            sequence=complete,
            num_scheduled_tokens=3,
            is_prefill=True,
        ),
    ]

    hidden_states = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [8.0, 9.0],
        ],
        dtype=torch.float32,
        device="cpu",
    )

    set_context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor(
            [0, 2, 5],
            dtype=torch.int32,
            device="cpu",
        ),
    )
    try:
        with torch.no_grad():
            logits = head(
                hidden_states
            )
    finally:
        reset_context()

    torch.testing.assert_close(
        logits,
        torch.tensor(
            [
                [2.0, 3.0, 5.0],
                [8.0, 9.0, 17.0],
            ],
            dtype=torch.float32,
            device="cpu",
        ),
    )

    selected = select_sample_logits(
        logits,
        scheduled,
    )

    torch.testing.assert_close(
        selected,
        torch.tensor(
            [
                [8.0, 9.0, 17.0],
            ],
            dtype=torch.float32,
            device="cpu",
        ),
    )