import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        self.config = config
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256)
        )

        atexit.register(self.exit)


    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled = self.scheduler.schedule()
        num_processed_tokens = 0

        if not scheduled:
            return [], num_processed_tokens, False

        is_prefill = scheduled[0].is_prefill
        if any(item.is_prefill != is_prefill for item in scheduled):
            raise RuntimeError(
                "Mixed prefill/decode batches are not supported by ModelRunner yet"
            )

        has_partial_prefill = any(
            item.is_prefill
            and item.num_scheduled_tokens
            != (
                item.sequence.num_prompt_tokens
                - item.sequence.num_computed_tokens
            )
            for item in scheduled
        )
        if has_partial_prefill:
            raise RuntimeError(
                "Chunked prefill execution is not supported by ModelRunner yet"
            )

        scheduled_sequences = [item.sequence for item in scheduled]

        outputs = self.model_runner.call(
            "run",
            scheduled_sequences,
            is_prefill,
        )

        if outputs is not None:
            outputs = outputs.cpu().tolist()

        # Commit progress only after model execution succeeds.
        for item in scheduled:
            item.sequence.advance_computed_tokens(
                item.num_scheduled_tokens
            )

        self.scheduler.postprocess(scheduled_sequences, outputs)

        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in scheduled_sequences
            if seq.is_finished
        ]
        num_processed_tokens = sum(
            item.num_scheduled_tokens for item in scheduled
        )

        return outputs, num_processed_tokens, is_prefill
    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params))

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
