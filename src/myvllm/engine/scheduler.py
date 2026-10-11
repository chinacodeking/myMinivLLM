from dataclasses import dataclass
from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager

@dataclass(frozen=True)
class ScheduledSequence:
    sequence: Sequence
    num_scheduled_tokens: int
    is_prefill: bool

    def __post_init__(self):
        if self.num_scheduled_tokens <= 0:
            raise ValueError(
                "num_scheduled_tokens must be positive"
            )

    @property
    def should_sample(self) -> bool:
        computed_after_step = (
            self.sequence.num_computed_tokens
            + self.num_scheduled_tokens
        )
        return (
            computed_after_step
            == self.sequence.num_tokens
        )


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        # Reject up front what the block manager could never satisfy, otherwise the
        # sequence sits in `waiting` forever and only surfaces as a stalled engine.
        capacity = len(self.block_manager.blocks)
        if sequence.num_blocks > capacity:
            raise ValueError(
                f"Sequence {sequence.seq_id} needs {sequence.num_blocks} blocks "
                f"({len(sequence)} tokens at block_size={self.block_manager.block_size}) "
                f"but the KV cache only holds {capacity}. "
                f"Raise max_cached_blocks or block_size, or shorten the prompt."
            )
        self.waiting.append(sequence)

    @staticmethod
    def _needs_prefill(seq: Sequence) -> bool:
        return (
            seq.needs_recompute
            or seq.num_computed_tokens
            < seq.num_prompt_tokens
        )


    def schedule(self) -> list[ScheduledSequence]:
        scheduled: list[ScheduledSequence] = []
        scheduled_decode_sequences: list[Sequence] = []

        current_scheduled_tokens = 0
        preempted = False

        # Decode has priority so that an arriving long prompt does not
        # delay requests that are already generating tokens.
        num_running_sequences = len(self.running)

        for _ in range(num_running_sequences):
            if (
                current_scheduled_tokens
                >= self.max_num_batched_tokens
                or len(scheduled)
                >= self.max_num_sequences
            ):
                break

            seq = self.running.popleft()

            # Incomplete prompts and preempted histories both need prefill
            # work. Put them back for the prefill pass below.
            if self._needs_prefill(seq):
                self.running.append(seq)
                continue

            # Decode processes one token. If that token crosses a block
            # boundary, the block manager may need another physical block.
            if not self.block_manager.can_append(seq):
                preempted = True

                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break

                continue

            self.block_manager.append(seq)

            scheduled_decode_sequences.append(seq)
            scheduled.append(
                ScheduledSequence(
                    sequence=seq,
                    num_scheduled_tokens=1,
                    is_prefill=False,
                )
            )
            current_scheduled_tokens += 1

        # Decode sequences remain active after this step, so restore them
        # to the running queue in their original scheduled order.
        if scheduled_decode_sequences:
            self.running.extendleft(
                reversed(scheduled_decode_sequences)
            )

        # Use the remaining token budget to resume prompts that were
        # partially computed in earlier steps.
        for seq in self.running:
            if (
                current_scheduled_tokens
                >= self.max_num_batched_tokens
                or len(scheduled)
                >= self.max_num_sequences
            ):
                break

            if not self._needs_prefill(seq):
                continue

            num_uncomputed_tokens = (
                seq.num_uncomputed_tokens
            )

            if num_uncomputed_tokens <= 0:
                continue

            remaining_budget = (
                self.max_num_batched_tokens
                - current_scheduled_tokens
            )
            num_scheduled_tokens = min(
                num_uncomputed_tokens,
                remaining_budget,
            )

            scheduled.append(
                ScheduledSequence(
                    sequence=seq,
                    num_scheduled_tokens=(
                        num_scheduled_tokens
                    ),
                    is_prefill=True,
                )
            )
            current_scheduled_tokens += (
                num_scheduled_tokens
            )

        # If capacity is still available, admit new prompts from the
        # waiting queue and give them the remaining token budget. After
        # preemption, wait until the next schedule call before admitting
        # waiting work so the released capacity remains available for the
        # scheduling decision that caused the preemption.
        while (
            self.waiting
            and not preempted
            and current_scheduled_tokens
            < self.max_num_batched_tokens
            and len(scheduled)
            < self.max_num_sequences
        ):
            seq = self.waiting[0]

            if not self.block_manager.can_allocate(seq):
                break

            seq = self.waiting.popleft()
            self.block_manager.allocate(seq)

            # A fully cached prompt still needs its last token executed to
            # produce the logits used for sampling. KV cache stores history,
            # not the output logits of a previous request.
            cached_computed_tokens = min(
                seq.num_cached_tokens,
                max(seq.num_tokens - 1, 0),
            )
            if (
                cached_computed_tokens
                > seq.num_computed_tokens
            ):
                seq.advance_computed_tokens(
                    cached_computed_tokens
                    - seq.num_computed_tokens
                )

            remaining_budget = (
                self.max_num_batched_tokens
                - current_scheduled_tokens
            )
            num_uncomputed_tokens = (
                seq.num_uncomputed_tokens
            )
            num_scheduled_tokens = min(
                num_uncomputed_tokens,
                remaining_budget,
            )

            if num_scheduled_tokens <= 0:
                self.block_manager.deallocate(seq)
                self.waiting.appendleft(seq)
                break

            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

            scheduled.append(
                ScheduledSequence(
                    sequence=seq,
                    num_scheduled_tokens=(
                        num_scheduled_tokens
                    ),
                    is_prefill=True,
                )
            )
            current_scheduled_tokens += (
                num_scheduled_tokens
            )

        if (
            not scheduled
            and not preempted
            and (self.waiting or self.running)
        ):
            raise RuntimeError(
                "Scheduler made no progress: "
                f"{len(self.waiting)} waiting and "
                f"{len(self.running)} running sequences, "
                f"{len(self.block_manager.free_block_ids)} free and "
                f"{len(self.block_manager.cached_block_ids)} "
                "evictable cached blocks out of "
                f"{len(self.block_manager.blocks)} total blocks. "
                "This means either a sequence that cannot fit "
                "in the KV cache, or blocks leaked because their "
                "ref_count never returned to 0."
            )

        return scheduled

    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.reset_computed_tokens()
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
