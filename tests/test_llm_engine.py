from unittest.mock import MagicMock, call
import torch
import pytest

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.scheduler import ScheduledSequence
from myvllm.engine.sequence import Sequence


def make_sequence(token_ids, block_size=4):
    return Sequence(
        token_ids=token_ids,
        block_size=block_size,
    )


def make_engine(scheduled):
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = MagicMock()
    engine.model_runner = MagicMock()
    engine.scheduler.schedule.return_value = scheduled
    return engine


def test_step_postprocesses_only_sequences_that_should_sample():
    intermediate = make_sequence(
        [1, 2, 3, 4]
    )
    complete = make_sequence(
        [5, 6, 7]
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

    engine = LLMEngine.__new__(
        LLMEngine
    )
    engine.scheduler = MagicMock()
    engine.scheduler.schedule.return_value = (
        scheduled
    )
    engine.model_runner = MagicMock()
    engine.model_runner.call.return_value = (
        torch.tensor(
            [99],
            dtype=torch.long,
        )
    )

    finished, num_processed, is_prefill = (
        engine.step()
    )

    engine.model_runner.call.assert_called_once_with(
        "run",
        scheduled,
    )
    engine.scheduler.postprocess.assert_called_once_with(
        [complete],
        [99],
    )

    assert intermediate.num_computed_tokens == 2
    assert complete.num_computed_tokens == 3
    assert finished == []
    assert num_processed == 5
    assert is_prefill


def test_step_passes_work_and_commits_progress_after_model_execution():
    seq = make_sequence([1, 2, 3])
    scheduled = [
        ScheduledSequence(
            sequence=seq,
            num_scheduled_tokens=3,
            is_prefill=True,
        )
    ]
    engine = make_engine(scheduled)

    model_outputs = MagicMock()
    model_outputs.cpu.return_value.tolist.return_value = [9]

    def run_model(method_name, scheduled_work):
        assert method_name == "run"
        assert scheduled_work == scheduled
        assert scheduled_work[0].sequence is seq
        assert scheduled_work[0].num_scheduled_tokens == 3
        assert scheduled_work[0].is_prefill
        assert seq.num_computed_tokens == 0
        return model_outputs

    def check_postprocess(sequences, token_ids):
        assert sequences == [seq]
        assert token_ids == [9]
        assert seq.num_computed_tokens == 3

    engine.model_runner.call.side_effect = run_model
    engine.scheduler.postprocess.side_effect = check_postprocess

    outputs, num_processed_tokens, is_prefill = engine.step()

    assert outputs == []
    assert num_processed_tokens == 3
    assert is_prefill
def test_step_advances_intermediate_prefill_without_sampling():
    seq = make_sequence(
        [1, 2, 3, 4]
    )
    scheduled = [
        ScheduledSequence(
            sequence=seq,
            num_scheduled_tokens=2,
            is_prefill=True,
        )
    ]

    engine = LLMEngine.__new__(
        LLMEngine
    )
    engine.scheduler = MagicMock()
    engine.scheduler.schedule.return_value = (
        scheduled
    )
    engine.model_runner = MagicMock()
    engine.model_runner.call.return_value = (
        torch.empty(
            0,
            dtype=torch.long,
        )
    )

    finished, num_processed, is_prefill = (
        engine.step()
    )

    engine.model_runner.call.assert_called_once_with(
        "run",
        scheduled,
    )
    engine.scheduler.postprocess.assert_called_once_with(
        [],
        [],
    )

    assert seq.num_computed_tokens == 2
    assert seq.completion_token_ids == []
    assert finished == []
    assert num_processed == 2
    assert is_prefill
def test_step_rejects_sampled_token_count_mismatch():
    seq = make_sequence(
        [1, 2, 3]
    )
    scheduled = [
        ScheduledSequence(
            sequence=seq,
            num_scheduled_tokens=3,
            is_prefill=True,
        )
    ]

    engine = LLMEngine.__new__(
        LLMEngine
    )
    engine.scheduler = MagicMock()
    engine.scheduler.schedule.return_value = (
        scheduled
    )
    engine.model_runner = MagicMock()
    engine.model_runner.call.return_value = (
        torch.empty(
            0,
            dtype=torch.long,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="sampled token count",
    ):
        engine.step()

    assert seq.num_computed_tokens == 0
    engine.scheduler.postprocess.assert_not_called()
def test_step_dispatches_mixed_work_as_homogeneous_model_calls():
    decoding = make_sequence(
        [1, 2, 3, 4],
        block_size=4,
    )
    decoding.advance_computed_tokens(
        decoding.num_prompt_tokens
    )
    decoding.append_token(99)

    incoming = make_sequence(
        [10, 11, 12, 13, 14, 15],
        block_size=4,
    )

    scheduled = [
        ScheduledSequence(
            sequence=decoding,
            num_scheduled_tokens=1,
            is_prefill=False,
        ),
        ScheduledSequence(
            sequence=incoming,
            num_scheduled_tokens=3,
            is_prefill=True,
        ),
    ]

    engine = make_engine(scheduled)

    decode_outputs = torch.tensor(
        [77],
        dtype=torch.long,
    )
    intermediate_prefill_outputs = torch.empty(
        0,
        dtype=torch.long,
    )

    engine.model_runner.call.side_effect = [
        decode_outputs,
        intermediate_prefill_outputs,
    ]

    (
        finished,
        num_processed,
        is_prefill,
    ) = engine.step()

    assert engine.model_runner.call.call_args_list == [
        call(
            "run",
            [scheduled[0]],
        ),
        call(
            "run",
            [scheduled[1]],
        ),
    ]

    engine.scheduler.postprocess.assert_called_once_with(
        [decoding],
        [77],
    )

    assert decoding.num_computed_tokens == 5
    assert incoming.num_computed_tokens == 3

    assert finished == []
    assert num_processed == 4
    assert is_prefill is None