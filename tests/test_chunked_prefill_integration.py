from unittest.mock import MagicMock

import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence


def make_sequence(
    token_ids,
    block_size=4,
):
    return Sequence(
        token_ids=token_ids,
        block_size=block_size,
    )


def test_long_prompt_moves_through_three_prefill_steps():
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=4,
        max_cached_blocks=10,
        block_size=4,
        eos=0,
    )
    seq = make_sequence(
        list(range(1, 11)),
        block_size=4,
    )
    scheduler.add_sequence(seq)

    engine = LLMEngine.__new__(
        LLMEngine
    )
    engine.scheduler = scheduler
    engine.model_runner = MagicMock()

    seen_methods = []
    seen_chunk_sizes = []
    seen_token_slices = []
    seen_sampling_decisions = []

    def run_scheduled_work(
        method_name,
        scheduled,
    ):
        seen_methods.append(
            method_name
        )
        seen_chunk_sizes.append(
            [
                item.num_scheduled_tokens
                for item in scheduled
            ]
        )
        seen_token_slices.append(
            [
                item.sequence.token_ids[
                    item.sequence.num_computed_tokens:
                    (
                        item.sequence.num_computed_tokens
                        + item.num_scheduled_tokens
                    )
                ]
                for item in scheduled
            ]
        )

        sampling_decisions = [
            item.should_sample
            for item in scheduled
        ]
        seen_sampling_decisions.append(
            sampling_decisions
        )

        sampled_token_ids = [
            99
            for should_sample
            in sampling_decisions
            if should_sample
        ]
        return torch.tensor(
            sampled_token_ids,
            dtype=torch.long,
        )

    engine.model_runner.call.side_effect = (
        run_scheduled_work
    )

    processed_per_step = []
    computed_after_step = []
    phases = []

    for _ in range(3):
        _, num_processed, is_prefill = (
            engine.step()
        )
        processed_per_step.append(
            num_processed
        )
        computed_after_step.append(
            seq.num_computed_tokens
        )
        phases.append(
            is_prefill
        )

    assert seen_methods == [
        "run",
        "run",
        "run",
    ]
    assert seen_chunk_sizes == [
        [4],
        [4],
        [2],
    ]
    assert seen_token_slices == [
        [[1, 2, 3, 4]],
        [[5, 6, 7, 8]],
        [[9, 10]],
    ]
    assert seen_sampling_decisions == [
        [False],
        [False],
        [True],
    ]

    assert processed_per_step == [
        4,
        4,
        2,
    ]
    assert computed_after_step == [
        4,
        8,
        10,
    ]
    assert phases == [
        True,
        True,
        True,
    ]

    assert seq.num_prompt_tokens == 10
    assert seq.num_computed_tokens == 10
    assert seq.completion_token_ids == [
        99
    ]
    assert engine.model_runner.call.call_count == 3