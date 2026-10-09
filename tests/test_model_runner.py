from unittest.mock import MagicMock

from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.scheduler import ScheduledSequence
from myvllm.engine.sequence import Sequence


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

    runner.prepare_prefill.assert_called_once_with([seq])
    runner.prepare_decode.assert_not_called()
    runner.run_model.assert_called_once_with(input_ids, True)
    runner.prepare_sample.assert_called_once_with([seq])
    runner.sampler.assert_called_once_with(
        logits,
        temperatures,
    )
    assert outputs is sampled_token_ids