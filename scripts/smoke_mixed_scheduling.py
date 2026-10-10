import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams


class CountingGreedySampler:
    def __init__(self):
        self.num_calls = 0

    def __call__(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        self.num_calls += 1
        return logits.argmax(dim=-1)


def make_chat_prompt(
    tokenizer,
    user_text: str,
) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": user_text,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def snapshot_work(scheduled):
    return [
        {
            "seq_id": item.sequence.seq_id,
            "phase": (
                "prefill"
                if item.is_prefill
                else "decode"
            ),
            "tokens": item.num_scheduled_tokens,
        }
        for item in scheduled
    ]


def main():
    model_path = os.environ.get(
        "MINIVLLM_MODEL_PATH"
    )
    if not model_path:
        raise RuntimeError(
            "MINIVLLM_MODEL_PATH is not set"
        )

    if not Path(model_path).is_dir():
        raise RuntimeError(
            "MINIVLLM_MODEL_PATH does not point "
            f"to a directory: {model_path}"
        )

    if Path(model_path).name != "Qwen3-0.6B":
        raise RuntimeError(
            "ModelRunner selects the architecture "
            "from the directory name; expected "
            f"'Qwen3-0.6B', got "
            f"'{Path(model_path).name}'"
        )

    token_budget = int(
        os.environ.get(
            "MINIVLLM_TOKEN_BUDGET",
            "16",
        )
    )
    if token_budget <= 1:
        raise RuntimeError(
            "MINIVLLM_TOKEN_BUDGET must be greater "
            "than 1 so Decode can use one token and "
            "Prefill can use the remaining budget"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_default_dtype(
        torch.float32
    )

    config = {
        "max_num_sequences": 2,
        "max_num_batched_tokens": token_budget,
        "max_cached_blocks": 1,
        "block_size": 16,
        "world_size": 1,
        "model_name_or_path": model_path,
        "enforce_eager": True,
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_heads": 16,
        "head_dim": 128,
        "num_kv_heads": 8,
        "intermediate_size": 3072,
        "num_layers": 28,
        "tie_word_embeddings": True,
        "base": 1000000,
        "rms_norm_epsilon": 1e-6,
        "qkv_bias": False,
        "scale": 1,
        "max_position": 32768,
        "ffn_bias": False,
        "max_num_batch_tokens": 128,
        "max_model_length": 128,
        "gpu_memory_utilization": 0.35,
        "eos": -1,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    first_prompt = make_chat_prompt(
        tokenizer,
        "Hi",
    )
    incoming_prompt = make_chat_prompt(
        tokenizer,
        (
            "Explain why mixed scheduling helps "
            "online language model inference. "
        )
        * 4,
    )

    first_prompt_token_ids = tokenizer.encode(
        first_prompt
    )
    incoming_prompt_token_ids = tokenizer.encode(
        incoming_prompt
    )

    first_prompt_tokens = len(
        first_prompt_token_ids
    )
    incoming_prompt_tokens = len(
        incoming_prompt_token_ids
    )

    if first_prompt_tokens > token_budget:
        raise RuntimeError(
            "The first prompt must fit in one Prefill "
            "step so it can enter Decode before the "
            "second prompt arrives: "
            f"prompt={first_prompt_tokens}, "
            f"budget={token_budget}"
        )

    expected_incoming_chunk = (
        token_budget - 1
    )

    if (
        incoming_prompt_tokens
        <= expected_incoming_chunk
    ):
        raise RuntimeError(
            "The incoming prompt must be longer than "
            "the budget remaining after one Decode "
            "token: "
            f"prompt={incoming_prompt_tokens}, "
            f"remaining={expected_incoming_chunk}"
        )

    if first_prompt_tokens + 4 > 128:
        raise RuntimeError(
            "First prompt exceeds max_model_length"
        )

    if incoming_prompt_tokens + 2 > 128:
        raise RuntimeError(
            "Incoming prompt exceeds max_model_length"
        )

    print(
        f"model_path={model_path}"
    )
    print(
        f"token_budget={token_budget}"
    )
    print(
        f"first_prompt_tokens={first_prompt_tokens}"
    )
    print(
        "incoming_prompt_tokens="
        f"{incoming_prompt_tokens}"
    )
    print(
        "expected_mixed_work="
        f"decode:1,prefill:{expected_incoming_chunk}"
    )

    engine = LLMEngine(
        config=config
    )

    sampler = CountingGreedySampler()
    engine.model_runner.sampler = sampler

    schedule_records = []
    original_schedule = (
        engine.scheduler.schedule
    )

    def recording_schedule():
        scheduled = original_schedule()
        record = snapshot_work(
            scheduled
        )
        schedule_records.append(record)
        print(
            f"scheduled={record}"
        )
        return scheduled

    engine.scheduler.schedule = (
        recording_schedule
    )

    model_phase_records = []
    original_model_call = (
        engine.model_runner.call
    )

    def recording_model_call(
        method_name,
        *args,
    ):
        if method_name == "run":
            scheduled = args[0]

            phases = {
                item.is_prefill
                for item in scheduled
            }
            if len(phases) != 1:
                raise RuntimeError(
                    "ModelRunner received a mixed "
                    "sub-batch"
                )

            phase = (
                "prefill"
                if scheduled[0].is_prefill
                else "decode"
            )
            model_phase_records.append(
                phase
            )
            print(
                "model_call="
                f"{phase},"
                f"sequences={len(scheduled)},"
                "tokens="
                f"{sum(item.num_scheduled_tokens for item in scheduled)}"
            )

        return original_model_call(
            method_name,
            *args,
        )

    engine.model_runner.call = (
        recording_model_call
    )

    first_sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=4,
        ignore_eos=True,
        max_model_length=128,
    )
    incoming_sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=2,
        ignore_eos=True,
        max_model_length=128,
    )

    # Step 1: run the first prompt by itself so that it
    # reaches Decode before the second prompt arrives.
    engine.add_prompt(
        first_prompt,
        first_sampling_params,
    )
    first_seq = engine.scheduler.waiting[-1]

    schedule_records.clear()
    model_phase_records.clear()

    (
        first_finished,
        first_processed,
        first_is_prefill,
    ) = engine.step()

    expected_first_schedule = [
        {
            "seq_id": first_seq.seq_id,
            "phase": "prefill",
            "tokens": first_prompt_tokens,
        }
    ]

    if schedule_records != [
        expected_first_schedule
    ]:
        raise RuntimeError(
            "Unexpected first schedule: "
            f"{schedule_records}"
        )

    if model_phase_records != [
        "prefill"
    ]:
        raise RuntimeError(
            "Unexpected first model calls: "
            f"{model_phase_records}"
        )

    if first_finished:
        raise RuntimeError(
            "The first request finished before the "
            "mixed step"
        )

    if first_processed != first_prompt_tokens:
        raise RuntimeError(
            "Unexpected number of tokens in first "
            f"step: {first_processed}"
        )

    if first_is_prefill is not True:
        raise RuntimeError(
            "The first step was not pure Prefill"
        )

    if (
        first_seq.num_computed_tokens
        != first_prompt_tokens
    ):
        raise RuntimeError(
            "First prompt progress was not committed"
        )

    if first_seq.num_completion_tokens != 1:
        raise RuntimeError(
            "First Prefill did not sample exactly "
            "one token"
        )

    print(
        "after_first_step="
        f"computed:{first_seq.num_computed_tokens},"
        f"completion:{first_seq.num_completion_tokens}"
    )

    # Step 2: add a long prompt while the first request
    # is ready for Decode.
    engine.add_prompt(
        incoming_prompt,
        incoming_sampling_params,
    )
    incoming_seq = engine.scheduler.waiting[-1]

    first_computed_before = (
        first_seq.num_computed_tokens
    )
    first_completion_before = (
        first_seq.num_completion_tokens
    )

    schedule_records.clear()
    model_phase_records.clear()

    (
        mixed_finished,
        mixed_processed,
        mixed_is_prefill,
    ) = engine.step()

    expected_mixed_schedule = [
        {
            "seq_id": first_seq.seq_id,
            "phase": "decode",
            "tokens": 1,
        },
        {
            "seq_id": incoming_seq.seq_id,
            "phase": "prefill",
            "tokens": expected_incoming_chunk,
        },
    ]

    if schedule_records != [
        expected_mixed_schedule
    ]:
        raise RuntimeError(
            "Unexpected mixed schedule: "
            f"{schedule_records}"
        )

    if model_phase_records != [
        "decode",
        "prefill",
    ]:
        raise RuntimeError(
            "Mixed work was not executed as Decode "
            "then Prefill: "
            f"{model_phase_records}"
        )

    if mixed_finished:
        raise RuntimeError(
            "A request finished unexpectedly during "
            "the mixed step"
        )

    if mixed_processed != token_budget:
        raise RuntimeError(
            "Mixed step did not consume the complete "
            f"token budget: {mixed_processed}"
        )

    if mixed_is_prefill is not None:
        raise RuntimeError(
            "Mixed step did not report mixed phase"
        )

    if (
        first_seq.num_computed_tokens
        != first_computed_before + 1
    ):
        raise RuntimeError(
            "Decode progress did not advance by one"
        )

    if (
        first_seq.num_completion_tokens
        != first_completion_before + 1
    ):
        raise RuntimeError(
            "Decode did not sample exactly one token"
        )

    if (
        incoming_seq.num_computed_tokens
        != expected_incoming_chunk
    ):
        raise RuntimeError(
            "Incoming Prefill progress is incorrect: "
            f"{incoming_seq.num_computed_tokens}"
        )

    if incoming_seq.num_completion_tokens != 0:
        raise RuntimeError(
            "Intermediate Prefill sampled too early"
        )

    if sampler.num_calls != 2:
        raise RuntimeError(
            "Expected exactly two sampling calls: "
            "one for the first Prefill and one for "
            f"Decode, got {sampler.num_calls}"
        )

    print(
        "mixed_step="
        f"processed:{mixed_processed},"
        "phase:mixed"
    )
    print(
        "first_request="
        f"computed:{first_seq.num_computed_tokens},"
        "completion_tokens:"
        f"{first_seq.completion_token_ids}"
    )
    print(
        "incoming_request="
        f"computed:{incoming_seq.num_computed_tokens},"
        f"prompt_tokens:{incoming_prompt_tokens},"
        "completion_tokens:"
        f"{incoming_seq.completion_token_ids}"
    )
    print(
        "SMOKE PASS: real GPU Decode and Prefill "
        "shared one engine step"
    )


if __name__ == "__main__":
    main()