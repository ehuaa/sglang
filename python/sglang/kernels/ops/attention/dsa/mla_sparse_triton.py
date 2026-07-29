# SPDX-License-Identifier: Apache-2.0
"""Triton sparse MLA attention with split-KV for low-batch decode.

Ported from vLLM PR #38476 (TRITON_MLA_SPARSE for SM8x) to replace the
Hopper-only FlashMLA sparse kernel (sgl_kernel sparse_prefill_fwd, "only
supported on SM90a and SM100f") on Ampere (A100/SM80). Sparse MLA over topk
indices in the absorbed form: q/kv dim_qk=576 (kv_lora_rank 512 + rope 64),
output dim_v=512, all bf16.
"""

import functools

import torch
import triton
import triton.language as tl

# log2(e) and ln(2): the kernels do exp2-based softmax, so sm_scale is folded
# with LOG2E and the LSE is converted back to natural log with LOGE2.
LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453


# DeepSeek-V3.2 / GLM-5 sparse MLA shape constants.
_BLOCK_DMODEL = 512
_BLOCK_DPE = 64
_BLOCK_DV = 512
_DIM_QK = _BLOCK_DMODEL + _BLOCK_DPE  # 576

_BLOCK_H = 16
# Smallest BLOCK_N the dispatch table offers; only used for the topk-divisibility
# check at dispatch time.
_MIN_BLOCK_N = 16

# Merge kernel grid is spread across heads and DV tiles to avoid a (1,1)
# launch starving the SMs (pattern from FlashMLA's combine kernel).
_MERGE_BLOCK_H = 1
_MERGE_BLOCK_DV_TILE = 128
assert _BLOCK_DV % _MERGE_BLOCK_DV_TILE == 0
_NUM_MERGE_DV_TILES = _BLOCK_DV // _MERGE_BLOCK_DV_TILE

# Launch configs come from a measured A100 sweep over the decode, spec-verify
# and prefill shapes at GLM-5.2 sizes (h_q=64, dim_qk=576, topk=2048).
# `_choose_config` holds the winners; num_warps varies with BLOCK_H, so
# `_NUM_WARPS` is only the default for the manual-override path. Explicit
# configs (no @triton.autotune) keep dispatch deterministic and avoid inline
# sweeps during CUDA-graph warmup.
_NUM_WARPS = 4
_NUM_STAGES = 2

# A100. Only used to score wave quantization in `_choose_config`, which was
# measured on this SM count.
_SM_COUNT = 108


@triton.jit
def _sparse_mla_compute_tile(
    q_buffer,
    k_buffer,  # V is the first BLOCK_DV lanes of each row of k_buffer.
    indices_ptr,
    cur_q,
    cur_head,
    cur_kv_head_id,
    mask_h,
    split_start,
    split_end,
    seq_kv,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
):
    """Shared stage-1 body: load Q, run the sparse online-softmax loop over
    `[split_start, split_end)` of the topk axis, return accumulators."""
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)

    q = tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )
    qpe = tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + offs_dpe[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )

    # Finite sentinel (not -inf) — when an entire BLOCK_N tile is masked,
    # `-inf - -inf = NaN` poisons the softmax; `sentinel - sentinel = 0`
    # gives `exp2(0) = 1` and the matching V rows are already 0.
    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_indice in range(split_start, split_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < split_end
        indices = tl.load(
            indices_ptr
            + cur_q * stride_indices_token
            + cur_kv_head_id * stride_indices_head
            + offs_indice,
            mask=mask_indice,
            other=-1,
        )
        mask_kv = (indices >= 0) & (indices < seq_kv)

        # ONE gather per tile. In the absorbed form V is exactly the first
        # BLOCK_DV lanes of K, so the tile is loaded token-major and transposed
        # in-register for the QK dot rather than gathered a second time
        # D-major. The gather is scattered (by topk index), so issuing it twice
        # doubled the HBM transactions, and holding both layouts doubled the
        # shared-memory footprint -- at BLOCK_H=32/BLOCK_N=32 that was 108,672 B
        # per CTA, over half of the A100's 164 KB, pinning the kernel to one
        # CTA (4 warps) per SM. Measured 1.77x at the unchanged tile size, and
        # it is what makes BLOCK_N=64 fit at all.
        kv_base = (
            k_buffer
            + indices[:, None] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
        )
        kv = tl.load(kv_base + offs_d[None, :], mask=mask_kv[:, None], other=0.0)
        kpe = tl.load(kv_base + offs_dpe[None, :], mask=mask_kv[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(kv).to(q.dtype))
        qk += tl.dot(qpe, tl.trans(kpe).to(q.dtype))

        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(kv.dtype), kv)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    return acc, e_max, e_sum


@triton.jit
def _sparse_mla_kernel_final(
    q_buffer,
    k_buffer,
    indices_ptr,
    out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_out_token,
    stride_out_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
):
    """Single-pass fast path: full topk, write final bf16 output directly."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        0,
        index_topk,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
    )

    # Guard against queries with zero valid KV (e_sum == 0 → NaN from 0/0).
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.jit
def _sparse_mla_kernel_split(
    q_buffer,
    k_buffer,
    indices_ptr,
    mid_out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    LOGE2: tl.constexpr,
):
    """Stage 1 of split-KV: process one slice of the topk axis and write
    its `(out_partial, lse_partial)` into the mid buffer."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    split_topk: tl.constexpr = tl.cdiv(index_topk, NUM_KV_SPLITS)
    split_start = split_kv_id * split_topk
    split_end = tl.minimum(split_start + split_topk, index_topk)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        split_start,
        split_end,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
    )

    # Partial output and natural-log LSE for stage-2 merge.
    # When a split has no valid KV (`e_sum == 0`), guard the divide so the
    # mid buffer holds 0 instead of NaN; otherwise the `0 * NaN = NaN` term
    # in stage 2 would poison every other split.
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    mid_base_2d = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head[:, None] * stride_mid_head
        + split_kv_id * stride_mid_split
    )
    tl.store(
        mid_base_2d + offs_dv[None, :],
        acc / e_sum_safe[:, None],
        mask=mask_h[:, None],
    )
    mid_lse_ptr = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head * stride_mid_head
        + split_kv_id * stride_mid_split
        + BLOCK_DV
    )
    tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2, mask=mask_h)


@triton.jit
def _sparse_mla_merge_kernel(
    mid_out_ptr,
    out_ptr,
    h_q,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_out_token,
    stride_out_head,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DV_TILE: tl.constexpr,
):
    """Stage 2: N-way online-softmax merge of per-split `(out, lse)` tiles.

    Grid is `(num_tokens, num_head_groups, num_dv_tiles)`. Each program handles
    `BLOCK_H` heads × `BLOCK_DV_TILE` output-dim lanes. The LSE reduction is
    identical across DV tiles for the same (token, head) — each program
    recomputes it locally, which is cheap (O(NUM_KV_SPLITS) scalars) and
    avoids inter-CTA synchronization.
    """
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_dv_tile = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_dv = cur_dv_tile * BLOCK_DV_TILE + tl.arange(0, BLOCK_DV_TILE)
    mask_dv = offs_dv < BLOCK_DV
    # Finite sentinel — same NaN guard as the split kernel for empty splits.
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV_TILE], dtype=tl.float32)

    mid_base_2d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head[:, None] * stride_mid_head
    )
    mid_lse_1d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head * stride_mid_head + BLOCK_DV
    )

    for split_kv_id in range(NUM_KV_SPLITS):
        tv = tl.load(
            mid_base_2d + split_kv_id * stride_mid_split + offs_dv[None, :],
            mask=mask_h[:, None] & mask_dv[None, :],
            other=0.0,
        )
        tlogic = tl.load(
            mid_lse_1d + split_kv_id * stride_mid_split,
            mask=mask_h,
            other=-float("inf"),
        )
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        exp_logic = tl.exp(tlogic - n_e_max)
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None] & mask_dv[None, :],
    )


def _wave_efficiency(grid: int, sm_count: int) -> float:
    """Fraction of the launched waves' SM slots that carry work."""
    return grid / (triton.cdiv(grid, sm_count) * sm_count)


@functools.lru_cache(maxsize=256)
def _choose_config(
    num_tokens: int, h_q: int, index_topk: int
) -> tuple[int, int, int, int]:
    """Pick `(num_kv_splits, block_h, block_n, num_warps)`; splits == 1 selects
    the single-pass final kernel.

    Measured on A100 (108 SMs) at GLM-5.2 shapes (h_q=64, dim_qk=576,
    topk=2048), in `units = num_tokens * cdiv(h_q, 16)`. Every tile here needs
    more than half of the 164 KB shared-memory budget, so exactly one CTA lands
    per SM and the grid size *is* the wave count.

    Above the small-batch table the choice is BLOCK_H=32 vs BLOCK_H=64. The
    grid is `num_tokens * cdiv(h_q, BLOCK_H)`, so BLOCK_H=64 halves it: that
    halves the per-head-group re-read of the topk KV, but can leave a much
    emptier tail wave. Wave efficiency decides, and reproduces all 11 measured
    points -- BLOCK_H=64 wins at nt=64/96/192/384/512/768/1024/4096 and loses at
    nt=48/128/256 (where its tail wave runs 44-79% full against 79-95%).
    """
    units = num_tokens * triton.cdiv(h_q, _BLOCK_H)
    if units <= 8:
        splits, block_h, block_n, warps = 8, _BLOCK_H, 64, 4
    elif units <= 16:
        splits, block_h, block_n, warps = 4, _BLOCK_H, 64, 4
    elif units <= 48:
        splits, block_h, block_n, warps = 4, 32, 64, 4
    elif units <= 128:
        splits, block_h, block_n, warps = 4, 32, 32, 4
    else:
        wide = _wave_efficiency(num_tokens * triton.cdiv(h_q, 64), _SM_COUNT)
        narrow = _wave_efficiency(num_tokens * triton.cdiv(h_q, 32), _SM_COUNT)
        if wide >= 0.9 * narrow:
            # BLOCK_H=64 requires num_warps=8: at 4 warps ptxas spills ~136
            # slots and the build runs ~1.7x slower.
            splits, block_h, block_n, warps = 1, 64, 64, 8
        else:
            splits, block_h, block_n, warps = 1, 32, 64, 4

    # A BLOCK_H wider than h_q only masks lanes off; drop to the largest tile
    # that carries real heads (and back to the 4-warp build with it).
    if block_h > h_q:
        block_h, warps = max(_BLOCK_H, h_q), _NUM_WARPS
    while splits > 1 and index_topk % splits != 0:
        splits //= 2
    return splits, block_h, block_n, warps


def triton_mla_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
) -> torch.Tensor:
    """Sparse MLA attention over topk indices.

    Args:
        q:         [num_tokens, num_heads_q, dim_qk] bf16
        kv:        [seq_kv, num_heads_kv=1, dim_qk] bf16
        indices:   [num_tokens, num_heads_kv=1, topk] int32
        sm_scale:  softmax scale
        num_kv_splits: override auto-heuristic; None/0 = auto, 1 = force single-pass.
        sm_count:  unused; retained for call-site compatibility.

    Returns:
        out:   [num_tokens, num_heads_q, _BLOCK_DV] bf16
    """
    num_tokens, num_heads_q, dim_qk = q.shape
    assert dim_qk == _DIM_QK, (
        f"sparse MLA kernel requires dim_qk={_DIM_QK} (DeepSeek-V3.2 / GLM-5), "
        f"got {dim_qk}"
    )
    assert kv.shape[1] == 1 and kv.shape[2] == _DIM_QK
    index_topk = indices.shape[2]
    assert index_topk % _MIN_BLOCK_N == 0, (
        f"topk ({index_topk}) must be a multiple of the smallest dispatch "
        f"BLOCK_N ({_MIN_BLOCK_N})"
    )

    kv_group_num = num_heads_q

    if num_kv_splits is None or num_kv_splits == 0:
        num_kv_splits, block_h, block_n, num_warps = _choose_config(
            num_tokens, num_heads_q, index_topk
        )
    else:
        # Manual override (bench/debug): generic tile for either path.
        block_h, block_n, num_warps = _BLOCK_H, 64, _NUM_WARPS
    num_head_groups = triton.cdiv(num_heads_q, min(block_h, num_heads_q))

    out = torch.empty(
        (num_tokens, num_heads_q, _BLOCK_DV),
        dtype=torch.bfloat16,
        device=q.device,
    )

    if num_kv_splits == 1:
        _sparse_mla_kernel_final[(num_tokens, num_head_groups)](
            q_buffer=q,
            k_buffer=kv,
            indices_ptr=indices,
            out_ptr=out,
            seq_kv=kv.shape[0],
            h_q=num_heads_q,
            stride_q_token=q.stride(0),
            stride_q_head=q.stride(1),
            stride_kv_token=kv.stride(0),
            stride_kv_head=kv.stride(1),
            stride_out_token=out.stride(0),
            stride_out_head=out.stride(1),
            stride_indices_token=indices.stride(0),
            stride_indices_head=indices.stride(1),
            sm_scale=sm_scale * LOG2E,
            index_topk=index_topk,
            kv_group_num=kv_group_num,
            BLOCK_H=block_h,
            BLOCK_N=block_n,
            BLOCK_DV=_BLOCK_DV,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            BLOCK_DPE=_BLOCK_DPE,
            num_warps=num_warps,
            num_stages=_NUM_STAGES,
        )
        return out

    # Split-KV: partial fp32 output + LSE per (token, head, split).
    mid_out = torch.empty(
        (num_tokens, num_heads_q, num_kv_splits, _BLOCK_DV + 1),
        dtype=torch.float32,
        device=q.device,
    )
    _sparse_mla_kernel_split[(num_tokens, num_head_groups, num_kv_splits)](
        q_buffer=q,
        k_buffer=kv,
        indices_ptr=indices,
        mid_out_ptr=mid_out,
        seq_kv=kv.shape[0],
        h_q=num_heads_q,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_kv_token=kv.stride(0),
        stride_kv_head=kv.stride(1),
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        sm_scale=sm_scale * LOG2E,
        index_topk=index_topk,
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DMODEL=_BLOCK_DMODEL,
        BLOCK_DPE=_BLOCK_DPE,
        LOGE2=LOGE2,
        num_warps=num_warps,
        num_stages=_NUM_STAGES,
    )

    _sparse_mla_merge_kernel[(num_tokens, num_heads_q, _NUM_MERGE_DV_TILES)](
        mid_out_ptr=mid_out,
        out_ptr=out,
        h_q=num_heads_q,
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_MERGE_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DV_TILE=_MERGE_BLOCK_DV_TILE,
        num_warps=2,
    )
    return out
