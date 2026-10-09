import math
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams


class RecordingGreedySampler:
    def __init__(self):
        self.logits = []

    def __call__(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        self.logits.append(
            logits.detach().float().cpu().clone()
        )
        return logits.argmax(dim=-1)


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

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_default_dtype(
        torch.float32
    )

    config = {
        "max_num_sequences": 1,
        "max_num_batched_tokens": int(
            os.environ.get(
                "MINIVLLM_TOKEN_BUDGET",
                "16",
            )
        ),
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

    user_prompt = (
        "Explain why chunked prefill helps online "
        "language model serving. "
        * 4
    )
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=2,
        max_model_length=(
            config["max_model_length"]
        ),
    )

    prompt_token_ids = tokenizer.encode(
        prompt
    )
    prompt_token_count = len(
        prompt_token_ids
    )
    token_budget = config[
        "max_num_batched_tokens"
    ]

    if token_budget <= 0:
        raise RuntimeError(
            "Scheduler token budget must be positive"
        )

    if (
        prompt_token_count
        + sampling_params.max_tokens
        > config["max_model_length"]
    ):
        raise RuntimeError(
            "Prompt plus generated tokens exceeds "
            "max_model_length: "
            f"{prompt_token_count} + "
            f"{sampling_params.max_tokens} > "
            f"{config['max_model_length']}"
        )

    expected_prefill_steps = math.ceil(
        prompt_token_count
        / token_budget
    )

    print(
        f"model_path={model_path}"
    )
    print(
        f"prompt_tokens={prompt_token_count}"
    )
    print(
        f"scheduler_token_budget={token_budget}"
    )
    print(
        "expected_prefill_steps="
        f"{expected_prefill_steps}"
    )

    engine = LLMEngine(
        config=config
    )
    recording_sampler = RecordingGreedySampler()
    engine.model_runner.sampler = recording_sampler

    engine.add_prompt(
        prompt,
        sampling_params,
    )

    seq = engine.scheduler.waiting[-1]
    records = []
    max_steps = (
        expected_prefill_steps
        + sampling_params.max_tokens
        + 4
    )

    for step_number in range(
        1,
        max_steps + 1,
    ):
        if engine.scheduler.is_finished():
            break

        computed_before = (
            seq.num_computed_tokens
        )
        completion_before = (
            seq.num_completion_tokens
        )

        (
            finished,
            num_processed,
            is_prefill,
        ) = engine.step()

        computed_after = (
            seq.num_computed_tokens
        )
        completion_after = (
            seq.num_completion_tokens
        )
        phase = (
            "prefill"
            if is_prefill
            else "decode"
        )
        sampled_this_step = (
            completion_after
            - completion_before
        )

        records.append(
            {
                "phase": phase,
                "processed": num_processed,
                "computed_before": (
                    computed_before
                ),
                "computed_after": (
                    computed_after
                ),
                "sampled": (
                    sampled_this_step
                ),
            }
        )

        print(
            f"step={step_number} "
            f"phase={phase} "
            f"processed={num_processed} "
            f"computed="
            f"{computed_before}->"
            f"{computed_after} "
            f"sampled={sampled_this_step} "
            f"finished={len(finished)}"
        )

    if not engine.scheduler.is_finished():
        raise RuntimeError(
            "Engine did not finish within "
            f"{max_steps} steps"
        )

    prefill_sizes = [
        record["processed"]
        for record in records
        if record["phase"] == "prefill"
    ]
    decode_sizes = [
        record["processed"]
        for record in records
        if record["phase"] == "decode"
    ]

    expected_prefill_sizes = [
        min(
            token_budget,
            prompt_token_count - start,
        )
        for start in range(
            0,
            prompt_token_count,
            token_budget,
        )
    ]

    if prefill_sizes != expected_prefill_sizes:
        raise RuntimeError(
            "Unexpected prefill sizes: "
            f"expected {expected_prefill_sizes}, "
            f"got {prefill_sizes}"
        )

    if any(
        size > token_budget
        for size in prefill_sizes
    ):
        raise RuntimeError(
            "A prefill chunk exceeded the "
            "scheduler token budget: "
            f"{prefill_sizes}"
        )

    if sum(prefill_sizes) != prompt_token_count:
        raise RuntimeError(
            "Prefill work did not cover the full "
            "prompt exactly once: "
            f"sizes={prefill_sizes}, "
            f"prompt_tokens={prompt_token_count}"
        )

    if not decode_sizes:
        raise RuntimeError(
            "Smoke test did not execute a decode step"
        )

    if seq.num_completion_tokens != 2:
        raise RuntimeError(
            "Expected exactly 2 completion tokens, "
            f"got {seq.num_completion_tokens}"
        )

    if len(recording_sampler.logits) != sampling_params.max_tokens:
        raise RuntimeError(
            "Expected one logits record per sampled token, "
            f"got {len(recording_sampler.logits)}"
        )

    print(
        "recorded_sampling_steps="
        f"{len(recording_sampler.logits)}"
    )

    result_path = os.environ.get(
        "MINIVLLM_RESULT_PATH"
    )
    if result_path:
        torch.save(
            {
                "model_path": model_path,
                "prompt_token_ids": prompt_token_ids,
                "token_budget": token_budget,
                "prefill_sizes": prefill_sizes,
                "completion_token_ids": list(
                    seq.completion_token_ids
                ),
                "sampling_logits": recording_sampler.logits,
            },
            result_path,
        )
        print(
            f"result_path={result_path}"
        )

    generated_text = tokenizer.decode(
        seq.completion_token_ids,
        skip_special_tokens=False,
    )

    print(
        f"prefill_sizes={prefill_sizes}"
    )
    print(
        f"decode_sizes={decode_sizes}"
    )
    print(
        "completion_token_ids="
        f"{seq.completion_token_ids}"
    )
    print(
        f"generated_text={generated_text!r}"
    )
    print(
        "SMOKE PASS: real GPU prefill "
        "and decode completed"
    )


if __name__ == "__main__":
    main()