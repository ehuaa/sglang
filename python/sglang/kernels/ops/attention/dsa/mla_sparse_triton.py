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

# Launch configs come from a measured decode sweep on A100 (108 SMs), GLM-5.2
# shapes (h_q=64, dim_qk=576, topk=2048); ctx 2.8k and 16k picked the same
# winners, all at num_warps=4 / num_stages=2. `_choose_config` holds the
# per-shape winner table. Explicit configs (no @triton.autotune) keep dispatch
# deterministic and avoid inline sweeps during CUDA-graph warmup.
_NUM_WARPS = 4
_NUM_STAGES = 2


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
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dpe = offs_dpe < BLOCK_DMODEL + BLOCK_DPE

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
        mask=(mask_h[:, None]) & (mask_dpe[None, :]),
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

        offs_k = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_d[:, None]
        )
        k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        offs_kpe = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dpe[:, None]
        )
        kpe = tl.load(
            k_buffer + offs_kpe,
            mask=(mask_kv[None, :]) & (mask_dpe[:, None]),
            other=0.0,
        )
        qk += tl.dot(qpe, kpe.to(q.dtype))

        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE)

        offs_v = (
            indices[:, None] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dv[None, :]
        )
        v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
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


@functools.lru_cache(maxsize=256)
def _choose_config(
    num_tokens: int, h_q: int, index_topk: int
) -> tuple[int, int, int]:
    """Pick `(num_kv_splits, block_h, block_n)`; splits == 1 selects the
    single-pass final kernel.

    Winner table from the A100 sweep, in `units = num_tokens * cdiv(h_q, 16)`
    (the BH=16 single-pass grid size; 108 SMs). Per-unit timings at h_q=64 vs
    the previous auto dispatch:
      units<=16  : split x4, BN=32   (nt<=4:  67-72us vs 89us)
      units<=48  : split x2, BN=64   (nt<=12: 86-103us vs 258-262us)
      units<=80  : split x4, BN=64   (nt<=16: ~150us vs 270us)
      units<=112 : final BH=16 BN=64 (nt<=24: 184us vs 290us)
      else       : final BH=32 BN=32 (nt>=32: 236-773us vs 399-1278us;
                   BH=32 halves the per-head-group re-read of the topk KV)
    """
    units = num_tokens * triton.cdiv(h_q, _BLOCK_H)
    if units <= 16:
        splits, block_h, block_n = 4, _BLOCK_H, 32
    elif units <= 48:
        splits, block_h, block_n = 2, _BLOCK_H, 64
    elif units <= 80:
        splits, block_h, block_n = 4, _BLOCK_H, 64
    elif units <= 112 or h_q < 32:
        splits, block_h, block_n = 1, _BLOCK_H, 64
    else:
        splits, block_h, block_n = 1, 32, 32
    while splits > 1 and index_topk % splits != 0:
        splits //= 2
    return splits, block_h, block_n


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
        num_kv_splits, block_h, block_n = _choose_config(
            num_tokens, num_heads_q, index_topk
        )
    else:
        # Manual override (bench/debug): generic tile for either path.
        block_h, block_n = _BLOCK_H, 64
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
            num_warps=_NUM_WARPS,
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
        num_warps=_NUM_WARPS,
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
