from unittest.mock import MagicMock

import pytest

from myvllm.engine import model_runner as model_runner_module
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.scheduler import ScheduledSequence
from myvllm.engine.sequence import Sequence
from myvllm.engine.model_input import PrefillMetadata

def make_sequence(token_ids, block_size=4):
    return Sequence(
        token_ids=token_ids,
        block_size=block_size,
    )


def test_run_reads_phase_and_sequences_from_scheduled_work():
    seq = make_sequence([1, 2, 3])
    scheduled = [
        ScheduledSequence(
            sequence=seq,
            num_scheduled_tokens=3,
            is_prefill=True,
        )
    ]

    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0

    input_ids = object()
    logits = object()
    temperatures = object()
    sampled_token_ids = object()

    runner.prepare_prefill = MagicMock(return_value=input_ids)
    runner.prepare_decode = MagicMock()
    runner.run_model = MagicMock(return_value=logits)
    runner.prepare_sample = MagicMock(return_value=temperatures)
    runner.sampler = MagicMock(return_value=sampled_token_ids)

    outputs = runner.run(scheduled)

    runner.prepare_prefill.assert_called_once_with(scheduled)
    runner.prepare_decode.assert_not_called()
    runner.run_model.assert_called_once_with(input_ids, True)
    runner.prepare_sample.assert_called_once_with([seq])
    runner.sampler.assert_called_once_with(
        logits,
        temperatures,
    )
    assert outputs is sampled_token_ids
def test_prepare_prefill_builds_metadata_from_scheduled_work(
    monkeypatch,
):
    seq = make_sequence([1, 2, 3])
    scheduled = [
        ScheduledSequence(
            sequence=seq,
            num_scheduled_tokens=3,
            is_prefill=True,
        )
    ]

    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 4

    class MetadataWasBuilt(Exception):
        pass

    def fake_build_prefill_metadata(
        actual_scheduled,
        block_size,
    ):
        assert actual_scheduled is scheduled
        assert block_size == 4
        raise MetadataWasBuilt

    monkeypatch.setattr(
        model_runner_module,
        "build_prefill_metadata",
        fake_build_prefill_metadata,
        raising=False,
    )

    with pytest.raises(MetadataWasBuilt):
        runner.prepare_prefill(scheduled)

def test_prepare_prefill_puts_absolute_positions_in_context(
    monkeypatch,
):
    seq = make_sequence([1, 2, 3, 4, 5, 6])
    scheduled = [
        ScheduledSequence(
            sequence=seq,
            num_scheduled_tokens=2,
            is_prefill=True,
        )
    ]

    metadata = PrefillMetadata(
        input_ids=[5, 6],
        positions=[4, 5],
        slot_mapping=[8, 9],
        seqlens_q=[2],
        seqlens_k=[6],
        cu_seqlens_q=[0, 2],
        cu_seqlens_k=[0, 6],
        block_tables=[[7, 2]],
    )

    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 4

    class FakeTensor:
        def __init__(
            self,
            values,
            **kwargs,
        ):
            self.values = list(values)
            self.options = kwargs

        def cuda(self, non_blocking):
            assert non_blocking
            return self

    def fake_build_prefill_metadata(
        actual_scheduled,
        block_size,
    ):
        assert actual_scheduled is scheduled
        assert block_size == 4
        return metadata

    fake_set_context = MagicMock()

    monkeypatch.setattr(
        model_runner_module,
        "build_prefill_metadata",
        fake_build_prefill_metadata,
    )
    monkeypatch.setattr(
        model_runner_module.torch,
        "tensor",
        FakeTensor,
    )
    monkeypatch.setattr(
        model_runner_module,
        "set_context",
        fake_set_context,
    )

    input_ids = runner.prepare_prefill(scheduled)

    assert input_ids.values == [5, 6]

    context_arguments = fake_set_context.call_args.kwargs

    assert context_arguments["positions"].values == [4, 5]