from unittest.mock import MagicMock

import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


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
def test_decode_and_new_prefill_share_one_engine_step():
    scheduler = Scheduler(
        max_num_sequences=2,
        max_num_batched_tokens=4,
        max_cached_blocks=10,
        block_size=4,
        eos=0,
    )

    decoding = make_sequence(
        [1, 2, 3, 4],
        block_size=4,
    )
    scheduler.add_sequence(decoding)

    engine = LLMEngine.__new__(
        LLMEngine
    )
    engine.scheduler = scheduler
    engine.model_runner = MagicMock()

    engine.model_runner.call.return_value = (
        torch.tensor(
            [99],
            dtype=torch.long,
        )
    )

    (
        first_finished,
        first_processed,
        first_is_prefill,
    ) = engine.step()

    assert first_finished == []
    assert first_processed == 4
    assert first_is_prefill is True

    assert decoding.num_computed_tokens == 4
    assert decoding.completion_token_ids == [99]

    incoming = make_sequence(
        [10, 11, 12, 13, 14, 15],
        block_size=4,
    )
    scheduler.add_sequence(incoming)

    seen_work = []

    def run_scheduled_work(
        method_name,
        scheduled,
    ):
        assert method_name == "run"

        assert all(
            item.is_prefill
            == scheduled[0].is_prefill
            for item in scheduled
        )

        seen_work.append(
            [
                (
                    item.sequence,
                    item.is_prefill,
                    item.num_scheduled_tokens,
                )
                for item in scheduled
            ]
        )

        if scheduled[0].is_prefill:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        return torch.tensor(
            [77],
            dtype=torch.long,
        )

    engine.model_runner.call.reset_mock()
    engine.model_runner.call.side_effect = (
        run_scheduled_work
    )

    (
        mixed_finished,
        mixed_processed,
        mixed_is_prefill,
    ) = engine.step()

    assert seen_work == [
        [
            (
                decoding,
                False,
                1,
            )
        ],
        [
            (
                incoming,
                True,
                3,
            )
        ],
    ]

    assert mixed_finished == []
    assert mixed_processed == 4
    assert mixed_is_prefill is None

    assert decoding.num_computed_tokens == 5
    assert decoding.completion_token_ids == [
        99,
        77,
    ]

    assert incoming.num_computed_tokens == 3
    assert incoming.completion_token_ids == []

    assert decoding in scheduler.running
    assert incoming in scheduler.running
    assert not scheduler.waiting


def test_active_prefix_cache_skips_ready_full_blocks():
    scheduler = Scheduler(
        max_num_sequences=2,
        max_num_batched_tokens=6,
        max_cached_blocks=3,
        block_size=4,
        eos=0,
    )

    first = make_sequence(
        [1, 2, 3, 4, 5, 6],
        block_size=4,
    )
    scheduler.add_sequence(first)

    engine = LLMEngine.__new__(
        LLMEngine
    )
    engine.scheduler = scheduler
    engine.model_runner = MagicMock()
    engine.model_runner.call.return_value = (
        torch.tensor(
            [99],
            dtype=torch.long,
        )
    )

    (
        first_finished,
        first_processed,
        first_is_prefill,
    ) = engine.step()

    assert first_finished == []
    assert first_processed == 6
    assert first_is_prefill is True
    assert first.num_computed_tokens == 6
    assert first.completion_token_ids == [99]

    second = make_sequence(
        [1, 2, 3, 4, 5, 6],
        block_size=4,
    )
    scheduler.add_sequence(second)

    seen_work = []

    def run_scheduled_work(
        method_name,
        scheduled,
    ):
        assert method_name == "run"

        seen_work.append(
            [
                (
                    item.sequence,
                    item.is_prefill,
                    item.num_scheduled_tokens,
                    item.sequence.token_ids[
                        item.sequence.num_computed_tokens:
                        (
                            item.sequence.num_computed_tokens
                            + item.num_scheduled_tokens
                        )
                    ],
                )
                for item in scheduled
            ]
        )

        if scheduled[0].is_prefill:
            return torch.tensor(
                [88],
                dtype=torch.long,
            )

        return torch.tensor(
            [77],
            dtype=torch.long,
        )

    engine.model_runner.call.reset_mock()
    engine.model_runner.call.side_effect = (
        run_scheduled_work
    )

    (
        mixed_finished,
        mixed_processed,
        mixed_is_prefill,
    ) = engine.step()

    assert seen_work == [
        [
            (
                first,
                False,
                1,
                [99],
            )
        ],
        [
            (
                second,
                True,
                2,
                [5, 6],
            )
        ],
    ]

    assert mixed_finished == []
    assert mixed_processed == 3
    assert mixed_is_prefill is None

    assert first.completion_token_ids == [
        99,
        77,
    ]
    assert second.completion_token_ids == [88]

    assert second.num_cached_tokens == 4
    assert second.num_computed_tokens == 6
    assert (
        second.block_table[0]
        == first.block_table[0]
    )


def test_finished_request_prefix_cache_is_reused():
    scheduler = Scheduler(
        max_num_sequences=1,
        max_num_batched_tokens=6,
        max_cached_blocks=2,
        block_size=4,
        eos=0,
    )

    first = Sequence(
        token_ids=[1, 2, 3, 4, 5, 6],
        block_size=4,
        sampling_params=SamplingParams(
            max_tokens=1,
        ),
    )
    scheduler.add_sequence(first)

    engine = LLMEngine.__new__(
        LLMEngine
    )
    engine.scheduler = scheduler
    engine.model_runner = MagicMock()
    engine.model_runner.call.return_value = (
        torch.tensor(
            [99],
            dtype=torch.long,
        )
    )

    (
        first_finished,
        first_processed,
        first_is_prefill,
    ) = engine.step()

    assert first_finished == [
        (
            first.seq_id,
            [99],
        )
    ]
    assert first_processed == 6
    assert first_is_prefill is True
    assert scheduler.is_finished()

    assert len(
        scheduler.block_manager.cached_block_ids
    ) == 1
    cached_block_id = next(
        iter(
            scheduler.block_manager.cached_block_ids
        )
    )

    second = make_sequence(
        [1, 2, 3, 4, 5, 6],
        block_size=4,
    )
    scheduler.add_sequence(second)

    seen_work = []

    def run_scheduled_work(
        method_name,
        scheduled,
    ):
        assert method_name == "run"
        seen_work.append(
            [
                (
                    item.sequence,
                    item.num_scheduled_tokens,
                    item.sequence.token_ids[
                        item.sequence.num_computed_tokens:
                        (
                            item.sequence.num_computed_tokens
                            + item.num_scheduled_tokens
                        )
                    ],
                )
                for item in scheduled
            ]
        )
        return torch.tensor(
            [88],
            dtype=torch.long,
        )

    engine.model_runner.call.reset_mock()
    engine.model_runner.call.side_effect = (
        run_scheduled_work
    )

    (
        second_finished,
        second_processed,
        second_is_prefill,
    ) = engine.step()

    assert seen_work == [
        [
            (
                second,
                2,
                [5, 6],
            )
        ]
    ]
    assert second_finished == []
    assert second_processed == 2
    assert second_is_prefill is True

    assert second.num_cached_tokens == 4
    assert second.num_computed_tokens == 6
    assert second.completion_token_ids == [88]
    assert second.block_table[0] == cached_block_id
