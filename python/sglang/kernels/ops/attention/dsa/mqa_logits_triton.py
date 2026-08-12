# SPDX-License-Identifier: Apache-2.0
"""Triton fallback for DeepGEMM's fp8_mqa_logits / fp8_paged_mqa_logits.

Ported from vLLM PR #38476 (TRITON_MLA_SPARSE backend for SM8x) to run the DSA
lightning-indexer logits on Ampere (SM80/A100), where DeepGEMM's fp8 paged MQA
logits kernels raise "Unsupported architecture".

SGLang's index-K cache layout matches what these kernels expect: per block,
`block_size * head_dim` FP8 K bytes followed by `block_size * 4` fp32 scale
bytes, exposed as `[num_blocks, block_size, 1, head_dim + 4]` uint8 (head_dim
128 -> 132). SM80 Triton cannot `tl.load` fp8e4nv directly, so the paged decode
kernel decodes FP8 -> bf16 through a 256-entry LUT; the prefill kernel
pre-decodes q/k to bf16 in the wrapper (compute-bound, LUT contends with the
matmul there).
"""

import torch
import triton
import triton.language as tl

# Paged decode: num_warps=4 dominated on A100/SM80 across {2,4,8}; the others
# were 1.5-1.7x slower at (num_heads=32, head_dim=128, block_size=64), so
# narrow the sweep to keep autotune from latching onto a bad pick under noise.
_PAGED_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

# Grid axis 1 of the paged decode kernel. Each of the NUM_SPLITS programs per
# query token walks the token's context pages with stride NUM_SPLITS, so the
# grid stays static (CUDA-graph safe) without scaling with the block-table
# width. The former per-(token, page) grid dispatched one program per possible
# page (block_tables width ~16k for a 1M-token pool) and early-exited ~99% of
# them past context_len; at B*next_n=24, ctx 2816, that empty dispatch made the
# kernel 311us vs 58us for the strided walk (A100 SM80), and it scaled with
# batch (B*next_n=96: 1229us vs 65us).
_PAGED_NUM_SPLITS = 64

# Prefill kernel adds BLOCK_N as a free tile axis. num_warps=8 was 1.5-3x
# worse than {2,4} across the sweep; keep BLOCK_N in {32, 64, 128} so autotune
# can pick per shape (BN=128 wins for GLM-5.1 long chunks).
_PREFILL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bn in (32, 64, 128)
    for nw in (2, 4)
    for ns in (2, 4)
]

# Query tokens handled per prefill program. One program per query token (the
# original shape) re-loaded the whole [BLOCK_N, head_dim] K tile for every
# token: arithmetic intensity is 1/(1/BLOCK_N + 1/(BLOCK_M*num_heads)), so at
# BLOCK_M=1 / num_heads=32 / BLOCK_N=128 it was 25.6 flop/byte against the ~156
# an A100 needs to be compute-bound, and the kernel ran at 29% of bf16 peak with
# an effective K bandwidth of 2.85 TB/s -- above HBM peak, i.e. L2 was absorbing
# the re-reads and L2 bandwidth was the limit.
#
# Sweeping BLOCK_M on A100 at the real chunked-prefill shapes (M=4096, H=32,
# D=128, N up to 32768) put the optimum at 16, worth ~1.6x: 8/32/64/128 all lose
# (128 needs num_warps=8 and still only reaches 1.36x, because the fully
# unrolled loop pushes registers 156 -> 191 and the grid gets too small to fill
# the SMs). Capped by M at dispatch so short prefills don't run masked-off
# iterations.
_PREFILL_BLOCK_M = 16

# ...but that sweep ran at num_heads=32, and the intensity term
# 1/(BLOCK_M*num_heads) means the optimum moves with the head count. Re-swept on
# A100 at num_heads=64 (DeepSeek-V4-Flash) over M in {256..8192} x N in
# {8192..262144}: BLOCK_M=8 wins by 1.27-1.44x at every M >= 1536, ties at
# M=1024, and loses slightly below that. num_heads=32 re-confirmed BLOCK_M=16
# across the same grid, so this has to key on both.
#
# The mechanism is L2 residency of q, not occupancy: at M=8192 / num_heads=64 the
# q tile is 134 MB against a 40 MB L2, so the kernel pays for re-reads and the
# narrower M tile cuts them; at M <= 1024 q fits and BLOCK_M stops mattering.
# Occupancy is a red herring -- num_warps=8 lifts it 12% -> 25% and is 9% SLOWER,
# because this kernel is short of registers, not of warps.
_PREFILL_BLOCK_M_WIDE_HEADS = 8
_PREFILL_WIDE_HEADS = 64
_PREFILL_WIDE_HEADS_MIN_M = 1024


def _prefill_block_m(num_heads: int, m: int) -> int:
    if num_heads >= _PREFILL_WIDE_HEADS and m >= _PREFILL_WIDE_HEADS_MIN_M:
        block_m = _PREFILL_BLOCK_M_WIDE_HEADS
    else:
        block_m = _PREFILL_BLOCK_M
    return min(block_m, triton.next_power_of_2(m))

# Warmup shape mirrors the chunked-prefill regime so autotune picks a tile
# sized for real serving. M matters now that the kernel tiles over it -- a
# dummy M of 8 would leave a single program on the M axis and mis-rank BLOCK_N.
_PREFILL_WARMUP_M = 4096
_PREFILL_WARMUP_N = 8192


_E4M3FN_BF16_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_e4m3fn_bf16_lut(device: torch.device) -> torch.Tensor:
    lut = _E4M3FN_BF16_LUT_CACHE.get(device)
    if lut is not None:
        return lut
    lut = (
        torch.arange(256, dtype=torch.uint8, device=device)
        .view(torch.float8_e4m3fn)
        .to(torch.bfloat16)
    )
    lut[0x7F] = 480.0
    lut[0xFF] = -480.0
    _E4M3FN_BF16_LUT_CACHE[device] = lut
    return lut


@triton.jit
def _decode_e4m3fn_bf16_lut(u, lut_ptr):
    return tl.load(lut_ptr + u.to(tl.uint32))


@triton.autotune(
    configs=_PAGED_AUTOTUNE_CONFIGS,
    key=["num_heads", "head_dim", "block_size"],
)
@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_fp8_ptr,
    kv_scale_ptr,
    weights_ptr,
    fp8_lut_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_kvf_block,
    stride_kvf_s,
    stride_kvf_d,
    stride_kvs_block,
    stride_kvs_s,
    stride_w_t,
    stride_w_h,
    stride_bt_b,
    stride_bt_k,
    stride_l_t,
    stride_l_n,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_id = tl.program_id(0)
    split_id = tl.program_id(1)

    batch_id = token_id // next_n
    next_n_id = token_id % next_n

    context_len = tl.load(context_lens_ptr + batch_id)
    num_ctx_blocks = tl.cdiv(context_len, block_size)
    q_offset = context_len - next_n + next_n_id

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim
    mask_n = offs_n < block_size

    q_base = q_ptr + batch_id * stride_q_b + next_n_id * stride_q_n
    q_byte = tl.load(
        q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0,
    )
    q = _decode_e4m3fn_bf16_lut(q_byte, fp8_lut_ptr)
    w = tl.load(
        weights_ptr + token_id * stride_w_t + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )

    for block_rk in tl.range(split_id, num_ctx_blocks, NUM_SPLITS):
        block_idx = tl.load(
            block_tables_ptr + batch_id * stride_bt_b + block_rk * stride_bt_k
        )
        kvf_base = kv_fp8_ptr + block_idx * stride_kvf_block
        k_byte = tl.load(
            kvf_base
            + offs_n[:, None] * stride_kvf_s
            + offs_d[None, :] * stride_kvf_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0,
        )
        kvs_base = kv_scale_ptr + block_idx * stride_kvs_block
        k_scale = tl.load(
            kvs_base + offs_n * stride_kvs_s,
            mask=mask_n,
            other=0.0,
        )
        k = _decode_e4m3fn_bf16_lut(k_byte, fp8_lut_ptr)
        # Scale in fp32 after the dot to avoid an extra bf16 round-trip on K.
        s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

        s = tl.where(s > 0, s, 0.0) * w[:, None]
        out = tl.sum(s, axis=0)

        k_offset = block_rk * block_size + offs_n
        valid = mask_n & (k_offset < context_len) & (k_offset <= q_offset)
        out = tl.where(valid, out, float("-inf"))

        # See `_fp8_mqa_logits_kernel` for why the row base is widened to int64
        # here and nowhere else.
        tl.store(
            logits_ptr + token_id.to(tl.int64) * stride_l_t + k_offset * stride_l_n,
            out,
            mask=mask_n,
        )


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_paged_mqa_logits.

    Args:
        q:             [B, next_n, H, D] fp8_e4m3fn
        kv_cache:      [num_blocks, block_size, 1, D+4] uint8 (FP8 + fp32 scale)
        weights:       [B*next_n, H] float32
        context_lens:  [B] int32
        block_tables:  [B, max_blocks] int32
        max_model_len: output width. Caller passes the active batch max so
            the logits buffer stays tight.
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[:context_len]` per row).
    Returns:
        logits:        [B*next_n, max_model_len] float32
    """
    B, next_n, num_heads, head_dim = q.shape
    _, block_size, one, d_plus_4 = kv_cache.shape
    assert one == 1
    assert d_plus_4 == head_dim + 4

    # Cache layout from the index-K store: per block, FP8 K bytes
    # (block_size * head_dim) followed by fp32 scales (block_size * 4). The
    # `[NB, block_size, 1, head_dim+4]` shape is a stride trick; re-slice flat.
    # Kernel decodes FP8 from uint8 via LUT (SM80 Triton can't load fp8e4nv).
    num_blocks = kv_cache.shape[0]
    kv_flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * head_dim
    kv_byte = kv_flat[:, :k_end].as_strided(
        (num_blocks, block_size, head_dim),
        (kv_flat.stride(0), head_dim, 1),
    )
    kv_scale = kv_flat[:, k_end:].view(torch.float32)
    q_byte = q.view(torch.uint8)

    if clean_logits:
        logits = torch.full(
            (B * next_n, max_model_len),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        logits = torch.empty(
            (B * next_n, max_model_len), dtype=torch.float32, device=q.device
        )

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_N = triton.next_power_of_2(block_size)

    fp8_lut = _get_e4m3fn_bf16_lut(q.device)
    grid = (B * next_n, _PAGED_NUM_SPLITS)
    _fp8_paged_mqa_logits_kernel[grid](
        q_byte,
        kv_byte,
        kv_scale,
        weights,
        fp8_lut,
        context_lens,
        block_tables,
        logits,
        q_byte.stride(0),
        q_byte.stride(1),
        q_byte.stride(2),
        q_byte.stride(3),
        kv_byte.stride(0),
        kv_byte.stride(1),
        kv_byte.stride(2),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        NUM_SPLITS=_PAGED_NUM_SPLITS,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
    )
    return logits


@triton.autotune(
    configs=_PREFILL_AUTOTUNE_CONFIGS,
    # Per-program work is N-independent; key on (heads, dim) only so chunked
    # prefill with varying N doesn't re-tune on every new chunk size.
    key=["num_heads", "head_dim"],
)
@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    logits_ptr,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_w_m,
    stride_w_h,
    stride_l_m,
    stride_l_n,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    N,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # bf16 q/k inputs: the wrapper pre-decodes FP8 -> bf16. At compute-bound
    # prefill this is ~2x the in-kernel LUT (LUT lookups contend with the
    # matmul for ALU/regs). Paged-decode keeps the LUT path.
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)

    n_start = n_block * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    # Early-exit when no row in this block has its `[ks, ke)` range overlapping
    # the tile. Chunked prefill produces many such all-masked tiles. Per-row
    # masking still happens below; these bounds only need to be conservative,
    # which the identity padding (+inf for ks, 0 for ke) keeps them.
    ks_blk = tl.load(ks_ptr + offs_m, mask=mask_m, other=2147483647)
    ke_blk = tl.load(ke_ptr + offs_m, mask=mask_m, other=0)
    if (n_start >= tl.max(ke_blk)) | (n_start + BLOCK_N <= tl.min(ks_blk)):
        # When `clean_logits=False` the caller skipped the -inf pre-fill, so
        # write -inf here for the early-exit tile.
        # `logits` is [M, N] fp32, so the row base is `m * N` in ELEMENTS and
        # Triton does not promote int multiplies. M is the chunked-prefill size
        # and N is the c4 indexer width (context / 4), so M * N reaches 2**31 at
        # chunked_prefill_size 8192 and a 1M context -- exactly int32 max, zero
        # headroom. A larger chunk (max_prefill_tokens already allows 16384)
        # wraps the offset negative into an illegal access. Widen the two store
        # addresses only: the q / weights / ks / ke loads are indexed by the
        # same rows but with strides small enough that they cannot overflow, and
        # widening `m_block` itself instead costs ~8% on this kernel.
        tl.store(
            logits_ptr
            + offs_m[:, None].to(tl.int64) * stride_l_m
            + offs_n[None, :] * stride_l_n,
            tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32),
            mask=mask_m[:, None] & mask_n[None, :],
        )
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim

    # Loaded once and reused by every query token in this block -- that reuse is
    # the whole point of tiling over M (see `_PREFILL_BLOCK_M`). The per-token
    # intermediate deliberately stays [BLOCK_H, BLOCK_N]: a single
    # [BLOCK_M * BLOCK_H, BLOCK_N] dot would be 128 KB of fp32 accumulator at
    # BLOCK_M=16 and would spill.
    k = tl.load(
        k_ptr + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    )
    kt = tl.trans(k)
    k_scale = tl.load(k_scale_ptr + offs_n, mask=mask_n, other=0.0)

    for i in tl.static_range(BLOCK_M):
        m = m_block * BLOCK_M + i
        alive = m < M
        q = tl.load(
            q_ptr
            + m * stride_q_m
            + offs_h[:, None] * stride_q_h
            + offs_d[None, :] * stride_q_d,
            mask=(mask_h[:, None] & mask_d[None, :]) & alive,
            other=0.0,
        )
        s = tl.dot(q, kt) * k_scale[None, :]

        w = tl.load(
            weights_ptr + m * stride_w_m + offs_h * stride_w_h,
            mask=mask_h & alive,
            other=0.0,
        )
        s = tl.where(s > 0, s, 0.0) * w[:, None]
        out = tl.sum(s, axis=0)

        ks = tl.load(ks_ptr + m, mask=alive, other=0)
        ke = tl.load(ke_ptr + m, mask=alive, other=0)
        valid = mask_n & (offs_n >= ks) & (offs_n < ke)
        out = tl.where(valid, out, float("-inf"))

        tl.store(
            logits_ptr + m.to(tl.int64) * stride_l_m + offs_n * stride_l_n,
            out,
            mask=mask_n & alive,
        )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_mqa_logits.

    Args:
        q:            [M, H, D] fp8_e4m3fn
        kv:           (k_fp8 [N, D], k_scales [N]) - fp8_e4m3fn, float32
        weights:      [M, H] float32
        cu_seqlen_ks: [M] int32
        cu_seqlen_ke: [M] int32
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[ks, ke)` per row). Matches DeepGEMM.
    Returns:
        logits:       [M, N] float32
    """
    k_fp8, k_scales = kv
    k_scales = k_scales.reshape(-1)

    M, num_heads, head_dim = q.shape
    N = k_fp8.shape[0]

    if clean_logits:
        logits = torch.full((M, N), float("-inf"), dtype=torch.float32, device=q.device)
    else:
        logits = torch.empty((M, N), dtype=torch.float32, device=q.device)

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)
    # The M loop is a `static_range`, so an oversized BLOCK_M costs masked-off
    # iterations on short prefills (draft extend, tiny chunks). Cap it by M.
    BLOCK_M = _prefill_block_m(num_heads, M)

    # Pre-decode FP8 -> bf16; the kernel runs a straight `tl.dot`.
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k_fp8.to(torch.bfloat16)

    # Grid depends on the autotuned BLOCK_N.
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _fp8_mqa_logits_kernel[grid](
        q_bf16,
        k_bf16,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q_bf16.stride(0),
        q_bf16.stride(1),
        q_bf16.stride(2),
        k_bf16.stride(0),
        k_bf16.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        N=N,
        M=M,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
    return logits


def warmup_fp8_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> None:
    """Prime the prefill `@triton.autotune` cache so first-call doesn't pay
    the inline sweep (~5-8 s on A100 SM80). N is a runtime scalar, so one
    small-M / long-N shape covers all chunk lengths."""
    max_block_n = max(c.kwargs["BLOCK_N"] for c in _PREFILL_AUTOTUNE_CONFIGS)
    m = _PREFILL_WARMUP_M
    n = max(_PREFILL_WARMUP_N, max_block_n)
    q = torch.empty(m, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device)
    k = torch.empty(n, head_dim, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.zeros(n, dtype=torch.float32, device=device)
    weights = torch.zeros(m, num_heads, dtype=torch.float32, device=device)
    ks = torch.zeros(m, dtype=torch.int32, device=device)
    ke = torch.full((m,), n, dtype=torch.int32, device=device)
    fp8_mqa_logits_triton(q, (k, scales), weights, ks, ke)


def warmup_fp8_paged_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> None:
    """Prime the paged-decode `@triton.autotune` cache for the indexer's
    logits kernel (see `warmup_fp8_mqa_logits_triton` for rationale).
    """
    num_blocks = 2
    q = torch.empty(1, 1, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device)
    kv_cache = torch.zeros(
        num_blocks, block_size, 1, head_dim + 4, dtype=torch.uint8, device=device
    )
    weights = torch.zeros(1, num_heads, dtype=torch.float32, device=device)
    context_lens = torch.tensor([block_size], dtype=torch.int32, device=device)
    block_tables = torch.zeros(1, 1, dtype=torch.int32, device=device)
    fp8_paged_mqa_logits_triton(
        q, kv_cache, weights, context_lens, block_tables, max_model_len=block_size
    )
