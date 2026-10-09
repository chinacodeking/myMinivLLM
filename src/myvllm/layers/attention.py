import triton 
import triton.language as tl
from myvllm.utils import get_context
import torch
import torch.nn as nn

def gather_paged_kv(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
    if cache.dim() != 4:
        raise ValueError(
            "cache must have shape "
            "(num_blocks, block_size, num_kv_heads, head_dim)"
        )

    if block_table.dim() != 1:
        raise ValueError(
            "block_table must be one-dimensional"
        )

    if context_len < 0:
        raise ValueError(
            "context_len must not be negative"
        )

    if block_size <= 0:
        raise ValueError(
            "block_size must be positive"
        )

    if cache.shape[1] != block_size:
        raise ValueError(
            "cache block size does not match block_size"
        )

    if cache.device != block_table.device:
        raise ValueError(
            "cache and block_table must be on the same device"
        )

    max_context_len = block_table.numel() * block_size
    if context_len > max_context_len:
        raise ValueError(
            "block_table cannot cover context_len"
        )

    if context_len == 0:
        return cache.new_empty(
            (
                0,
                cache.shape[2],
                cache.shape[3],
            )
        )

    logical_positions = torch.arange(
        context_len,
        dtype=torch.long,
        device=block_table.device,
    )

    logical_blocks = torch.div(
        logical_positions,
        block_size,
        rounding_mode="floor",
    )
    block_offsets = torch.remainder(
        logical_positions,
        block_size,
    )

    physical_blocks = block_table[
        logical_blocks
    ].to(
        dtype=torch.long,
    )

    if physical_blocks.lt(0).any():
        raise ValueError(
            "block_table contains an invalid physical block"
        )

    if physical_blocks.ge(cache.shape[0]).any():
        raise ValueError(
            "block_table references a block outside the cache"
        )

    return cache[
        physical_blocks,
        block_offsets,
    ]

def paged_attention_prefill_reference(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    if query.dim() != 3:
        raise ValueError(
            "query must have shape "
            "(num_query_tokens, num_heads, head_dim)"
        )

    if k_cache.shape != v_cache.shape:
        raise ValueError(
            "K and V cache shapes must match"
        )

    if k_cache.dim() != 4:
        raise ValueError(
            "K and V caches must be four-dimensional"
        )

    if block_tables.dim() != 2:
        raise ValueError(
            "block_tables must be two-dimensional"
        )

    if cu_seqlens_q.dim() != 1:
        raise ValueError(
            "cu_seqlens_q must be one-dimensional"
        )

    if cu_seqlens_k.dim() != 1:
        raise ValueError(
            "cu_seqlens_k must be one-dimensional"
        )

    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "Query and KV sequence counts must match"
        )

    if positions.numel() != query.shape[0]:
        raise ValueError(
            "positions must contain one entry per query token"
        )

    tensors = (
        k_cache,
        v_cache,
        block_tables,
        cu_seqlens_q,
        cu_seqlens_k,
        positions,
    )
    if any(
        tensor.device != query.device
        for tensor in tensors
    ):
        raise ValueError(
            "All attention tensors must be on the same device"
        )

    num_sequences = cu_seqlens_q.numel() - 1
    if block_tables.shape[0] != num_sequences:
        raise ValueError(
            "block_tables must contain one row per sequence"
        )

    num_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = k_cache.shape[2]

    if k_cache.shape[3] != head_dim:
        raise ValueError(
            "Query and KV head dimensions must match"
        )

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "Query heads must be divisible by KV heads"
        )

    q_offsets = (
        cu_seqlens_q
        .detach()
        .cpu()
        .tolist()
    )
    k_offsets = (
        cu_seqlens_k
        .detach()
        .cpu()
        .tolist()
    )

    if q_offsets[-1] != query.shape[0]:
        raise ValueError(
            "cu_seqlens_q does not cover all query tokens"
        )

    outputs = []
    heads_per_kv_head = num_heads // num_kv_heads

    for sequence_index in range(num_sequences):
        q_start = q_offsets[sequence_index]
        q_end = q_offsets[sequence_index + 1]

        context_start = k_offsets[sequence_index]
        context_end = k_offsets[sequence_index + 1]
        context_len = context_end - context_start

        sequence_query = query[
            q_start:q_end
        ]
        sequence_positions = positions[
            q_start:q_end
        ]

        logical_k = gather_paged_kv(
            cache=k_cache,
            block_table=block_tables[sequence_index],
            context_len=context_len,
            block_size=block_size,
        )
        logical_v = gather_paged_kv(
            cache=v_cache,
            block_table=block_tables[sequence_index],
            context_len=context_len,
            block_size=block_size,
        )

        if heads_per_kv_head > 1:
            logical_k = logical_k.repeat_interleave(
                heads_per_kv_head,
                dim=1,
            )
            logical_v = logical_v.repeat_interleave(
                heads_per_kv_head,
                dim=1,
            )

        query_by_head = (
            sequence_query
            .transpose(0, 1)
            .float()
        )
        key_by_head = (
            logical_k
            .transpose(0, 1)
            .float()
        )
        value_by_head = (
            logical_v
            .transpose(0, 1)
            .float()
        )

        scores = torch.matmul(
            query_by_head,
            key_by_head.transpose(-1, -2),
        )
        scores = scores * scale

        key_positions = torch.arange(
            context_len,
            dtype=torch.long,
            device=query.device,
        )
        causal_mask = (
            key_positions.unsqueeze(0)
            <= sequence_positions.unsqueeze(1)
        )

        scores = scores.masked_fill(
            ~causal_mask.unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )

        probabilities = torch.softmax(
            scores,
            dim=-1,
        )

        sequence_output = torch.matmul(
            probabilities,
            value_by_head,
        )
        sequence_output = (
            sequence_output
            .transpose(0, 1)
            .to(dtype=query.dtype)
        )
        outputs.append(sequence_output)

    if not outputs:
        return torch.empty_like(query)

    return torch.cat(
        outputs,
        dim=0,
    )


@triton.jit
def store_kvcache_kernel(
    key_ptr, # pointer to what we want to store
    value_ptr,
    k_cache_ptr, # pointer to where we want to store
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr
):
    """
    Store keys and values into paged KV cache.
    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)
    Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    # thread ID, in dimension 0
    token_idx = tl.program_id(0) # each GPU thread processes one token
    # slot ID, where in cache to store this token
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    if slot_idx == -1:
        return
    
    # Calculate which block and position within block
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    # Process each head
    # program_id(0) = which token
    # program_id(1) = which head
    head_idx = tl.program_id(1)
    
    # it creates a vector [0, 1, ..., head_dim-1]
    # Load key and value for this token and head
    head_offsets = tl.arange(0, head_dim)
    # Input: (num_tokens, num_kv_heads, head_dim)
    # example: input_offset = 5 * (8 * 128) + 3 * 128 + [0, 1, 2, ..., 127]
    #         = 5120 + 384 + [0, 1, 2, ..., 127]
    #         = [5504, 5505, 5506, ..., 5631]
    input_offset = (token_idx * num_kv_heads * head_dim + # skip previous tokens
                    head_idx * head_dim + # skip previous heads
                    head_offsets)

    # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim + # skip previous blocks
                   block_offset * num_kv_heads * head_dim + # skip previous positions in block
                   head_idx * head_dim + # skip previous heads
                   head_offsets) 
    
    # load key and value value floats from the pointers's memory
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)
    
    # store into cache
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor,
    block_size: int
):
    """
    Store key-value pairs into paged cache.
    
    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        block_size: number of tokens per block
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    
    # Make contiguous if needed
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"
    
    grid = (num_tokens, num_kv_heads)
    # launch num_tokens x num_kv_heads threads
    store_kvcache_kernel[grid](
        key, # tensors are automatically converted to pointers by triton
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size
    )


@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences.
    Each program processes one block of queries for one head in one sequence.
    """
    # Program IDs
    start_m = tl.program_id(0) # block index
    off_h = tl.program_id(1) # head index
    seq_idx = tl.program_id(2) # sequence index

    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    # Early exit if this block is beyond sequence length
    if start_m * BLOCK_M >= seq_len:
        return
    
    # Offset for this block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize output accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    # Number of blocks to process
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    # Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Mask for valid positions
        mask_n = offs_n < seq_len
        
        # K pointers: K has shape (total_tokens, num_kv_heads, head_dim)
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        
        # Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # Apply causal mask: only attend to positions <= current position
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Rescale previous accumulator
        acc = acc * alpha[:, None]
        
        # Load V block - shape (BLOCK_N, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Optimized Flash Attention for prefill phase with variable-length sequences.
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: cumulative sequence lengths
        scale: attention scale factor
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Conservative block sizes to avoid OOM on shared memory
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (for float32 attention scores)
    # + BLOCK_M * head_dim * 4 (for Q)
    # + BLOCK_N * head_dim * 4 (for K, V)
    # Want to keep total < 48KB for most GPUs
    
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16
    
    # Number of sequences
    num_seqs = cu_seqlens.shape[0] - 1
    
    # Find max sequence length to determine grid size
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # Calculate grid dimensions - launch all kernels at once
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    
    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized paged attention kernel for decode phase.
    Processes KV cache in chunks.

    The block table is gathered per token rather than per chunk, so a chunk may
    straddle any number of blocks and no relation between BLOCK_N and block_size
    is assumed. Each lane resolves its own token independently:

        token t  ->  block_tables[batch, t // block_size], slot t % block_size
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Determine which KV head this query head uses (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load context length
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    # Load query: (batch_size, num_heads, head_dim)
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
    # Initialize accumulators
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    # Calculate total number of chunks to process
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    
    # Process all tokens in chunks
    for chunk_idx in range(max_chunks):
        # Global token index for this chunk
        token_start = chunk_idx * BLOCK_N
        
        # Only process if within valid range
        if token_start < context_len:
            # Determine which tokens in this chunk are valid
            offs_n = token_start + tl.arange(0, BLOCK_N)
            logical_block = offs_n // block_size
            # physical_block * block_size already moves the base pointer to the
            # start of the block, so what is added is the offset *within* the
            # block, not the global token index.
            offs_in_block = offs_n % block_size

            in_range = (offs_n < context_len) & (logical_block < max_num_blocks)

            # One block-table entry per token: a chunk is free to span several
            # blocks, and those blocks need not be adjacent in the cache.
            physical_block = tl.load(
                block_tables_ptr + batch_idx * max_num_blocks + logical_block,
                mask=in_range, other=-1)
            valid = in_range & (physical_block != -1)
            # Masked-out lanes still take part in the address arithmetic, so give
            # them block 0 to keep every computed offset inside the cache.
            physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

            # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
            kv_offset = (physical_block[None, :] * (block_size * num_kv_heads * head_dim)
                         + offs_in_block[None, :] * (num_kv_heads * head_dim)
                         + kv_head_idx * head_dim
                         + offs_d[:, None])

            # Compute attention scores for this chunk
            k = tl.load(k_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            k = tl.cast(k, tl.float32)
            score = tl.sum(q[:, None] * k, axis=0) * scale
            qk = tl.where(valid, score, -1e10)

            # Online softmax
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)

            # Rescale accumulator
            acc = acc * alpha
            l_i = l_i * alpha

            # Accumulate weighted values
            v = tl.load(v_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            v = tl.cast(v, tl.float32)
            weight = tl.where(valid, p, 0.0)
            acc = acc + tl.sum(weight[None, :] * v, axis=1)
            l_i = l_i + tl.sum(weight)

            m_i = m_i_new
    
    # Normalize
    output = acc / l_i
    
    # Store output
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.
    
    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
        scale: attention scale factor
    
    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    # Make contiguous
    query = query.contiguous()
    
    output = torch.empty_like(query)
    
    # Chunk size for processing KV tokens
    BLOCK_N = 64 if head_dim <= 128 else 32
    
    grid = (batch_size, num_heads)
    
    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    
    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Store current k, v into cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # Ensure k, v are in the right shape: (num_tokens, num_kv_heads, head_dim)
            if k.dim() == 4:
                # Batched: (B, N, num_kv_heads, head_dim) -> reshape to (B*N, num_kv_heads, head_dim)
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # Already in correct shape (num_tokens, num_kv_heads, head_dim)
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()
            
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # Prefill: use flash attention
            # Varlen mode: (total_tokens, num_heads, head_dim)
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale, 
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            o = paged_attention_decode(
                q, 
                k_cache, 
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # Example usage
    layer = Attention(num_heads=8, head_dim=64).cuda()
    B, N, D = 4, 1024, 512
    q = torch.randn(B, N, D).cuda()
    k = torch.randn(B, N, D).cuda()
    v = torch.randn(B, N, D).cuda()
    layer.k_cache = torch.zeros(B, N, D).cuda()
    layer.v_cache = torch.zeros(B, N, D).cuda()
    slot_mapping = torch.arange(N).cuda()

    for _ in range(10):  # Warm-up iterations
        _ = layer(q, k, v)

    import time
    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")