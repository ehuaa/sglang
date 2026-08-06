# SPDX-License-Identifier: Apache-2.0
"""Symmetric-memory dim-0 all-gather and reduce-scatter for pre-Hopper GPUs.

``triton_symm_mem_ag`` gathers with one ``multimem.st`` per 128-bit chunk: the
fabric fans that single store out to every peer. Multicast is Hopper-and-later,
so on Ampere ``symm_mem_hdl.multicast_ptr`` is 0 and that kernel is disabled --
every ``all_gather_into_tensor`` falls back to an NCCL ring.

This module keeps the symmetric-memory substrate and replaces only the store:
each thread issues plain ``st.relaxed.sys.global.v4.b32`` through the peer base
pointers in ``buffer_ptrs_dev``, which is P2P over NVLink and legal on sm80.
Both collectives are one-shot -- a single push to every peer, then a barrier --
so they cost one hop where the ring costs ``world_size - 1`` sequential ones.

What that trades away, stated plainly: both move ``(world_size - 1) *
shard_bytes`` off a rank, the same wire volume as the NCCL ring they replace.
Neither can win on bandwidth, so ``_MAX_BYTES_PER_RANK`` keeps large payloads
on NCCL and they compete only in the latency-bound regime decode lives in.

The two differ in how much headroom there is, and it is worth being precise
because it decides whether either is worth enabling:

* All-gather replaces ``ncclAllGather``, which is already a tight ring. On 8x
  A100 the measured irreducible cost (min across ranks of one collective, i.e.
  the rank that never waits) is ~35 us of a ~174 us attributed average -- 80%
  of what the profiler charges to the kernel is rank skew, which a faster
  kernel cannot recover.
* Reduce-scatter replaces ``pynccl``'s ``reduce_scatterv``, which is *not* a
  ring reduce-scatter: it is ``world_size`` grouped ``ncclReduce`` calls, one
  per root (see ``PyNcclCommunicator.reduce_scatter``). Measured irreducible
  cost is ~80 us of a ~162 us average -- only 50% skew, and the LL protocol
  doubles bytes on the wire, so it lands near 51 GB/s where the link offers
  several times that. That is the one with real headroom.

Only intra-node groups qualify -- peer pointers are a P2P mapping, and a
cross-node rank has no address to store to.
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# One 128-bit store per thread.
_BLOCK_THREADS = 1024
_BYTES_PER_CHUNK = 16
# Measured crossover on 8x A100 (NVLink, bf16): P2P beats the NCCL ring below
# ~1.5 MB per rank and loses above it, because it moves world_size times the
# bytes. Keep a margin.
_MAX_BYTES_PER_RANK = 1 << 20
_MAX_BLOCKS = 32
# Reduction over the staging slots is plain streaming work, unrelated to the
# push's one-chunk-per-thread layout, so it gets its own tile.
_SUM_BLOCK = 2048


@triton.jit
def _st_128(ptr, x, y, z, w, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            st.relaxed.sys.global.v4.b32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[ptr, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ld_128(in_ptr, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[in_ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_flat_tid():
    tid_x, tid_y, tid_z = tl.inline_asm_elementwise(
        "mov.u32 $0, %tid.x; mov.u32 $1, %tid.y; mov.u32 $2, %tid.z;",
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    ntid_x, ntid_y, _ = tl.inline_asm_elementwise(
        "mov.u32 $0, %ntid.x; mov.u32 $1, %ntid.y; mov.u32 $2, %ntid.z;",
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def _sync_threads():
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _send_signal_release(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %tmp32_<1>;
            .reg .pred %p<1>;
            send_signal:
                atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal_acquire(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %tmp32_<1>;
            .reg .pred %p<1>;
            wait_signal:
                atom.global.sys.acquire.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _blockwise_barrier(signal_pad_ptrs, rank: tl.constexpr, world_size: tl.constexpr):
    """Exit barrier only. Every rank must launch the same grid: the pad slot is
    indexed by block id, so a divergent block count deadlocks."""
    block_id = (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )
    flat_tid = _get_flat_tid()
    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote = tl.load(signal_pad_ptrs + remote_ranks).to(tl.pointer_type(tl.uint32))
    local = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))
    if flat_tid < world_size:
        _send_signal_release(remote + block_id * world_size + rank)
        _wait_signal_acquire(local + block_id * world_size + remote_ranks)


@triton.jit
def _p2p_all_gather_kernel(
    input_ptr,
    buffer_ptrs,  # device array of world_size peer base pointers
    signal_pad_ptr,
    num_chunks,  # 128-bit chunks in this rank's shard
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    # dim-0 gather: this rank's whole shard lands contiguously at slot RANK in
    # every peer, so the destination offset is a single scalar.
    dst_base = RANK * num_chunks
    bp = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    in_base = input_ptr.to(tl.pointer_type(tl.uint64))

    pid = tl.program_id(axis=0)
    tid = _get_flat_tid()
    block_start = pid * BLOCK_SIZE
    while block_start < num_chunks:
        chunk = block_start + tid
        mask = chunk < num_chunks
        x, y, z, w = _ld_128(in_base + chunk * 2, mask)
        for p in tl.static_range(WORLD_SIZE):
            peer = tl.load(bp + p).to(tl.pointer_type(tl.uint64))
            _st_128(peer + (dst_base + chunk) * 2, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    _sync_threads()
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE)


@triton.jit
def _p2p_scatter_kernel(
    input_ptr,
    buffer_ptrs,  # device array of world_size peer base pointers
    signal_pad_ptr,
    num_chunks,  # 128-bit chunks in one rank's slice
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    # Transpose of the all-gather push. There, one shard is broadcast to slot
    # RANK of every peer; here, peer p's slice of the input goes to slot RANK of
    # peer p. Both keep a single scalar destination offset, and both leave block
    # b of every rank holding exactly the chunks block b of the peers wrote --
    # which is what makes the block-indexed barrier sufficient.
    dst_base = RANK * num_chunks
    bp = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    in_base = input_ptr.to(tl.pointer_type(tl.uint64))

    pid = tl.program_id(axis=0)
    tid = _get_flat_tid()
    block_start = pid * BLOCK_SIZE
    while block_start < num_chunks:
        chunk = block_start + tid
        mask = chunk < num_chunks
        for p in tl.static_range(WORLD_SIZE):
            x, y, z, w = _ld_128(in_base + (p * num_chunks + chunk) * 2, mask)
            peer = tl.load(bp + p).to(tl.pointer_type(tl.uint64))
            _st_128(peer + (dst_base + chunk) * 2, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    _sync_threads()
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE)


@triton.jit
def _sum_slots_kernel(
    stage_ptr,
    out_ptr,
    n_elems,
    WORLD_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum the world_size contributions parked in the local staging slots.

    Runs as its own launch rather than fused into the push: the exit barrier
    there is block-indexed, so a fused reduction would have to re-derive which
    blocks may read which chunks, and stream ordering already gives the same
    guarantee for free. Accumulates in fp32 -- NCCL's bf16 ring sums pairwise in
    bf16, so this is the more accurate of the two, not merely a different one.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p in tl.static_range(WORLD_SIZE):
        acc += tl.load(stage_ptr + p * n_elems + offs, mask=mask, other=0.0).to(
            tl.float32
        )
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


class Sm80P2PCollectives:
    """Lazily-built symmetric buffer, shared by both collectives, plus the
    guards that decide whether a given call may use it.

    One buffer serves both because each op fully consumes it before returning
    and both run on the caller's stream: the gather's ``copy_`` out and the
    scatter's ``_sum_slots_kernel`` are stream-ordered after their own push, so
    a later op cannot observe a partially overwritten staging area. The signal
    pad is likewise reusable -- the barrier's paired CAS (0->1 on send, 1->0 on
    wait) leaves every slot back at 0.
    """

    def __init__(self, group: dist.ProcessGroup, rank_in_group: int, max_bytes: int):
        self.group = group
        self.rank = rank_in_group
        self.world_size = group.size()
        self.max_bytes = max_bytes
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        pad_bytes = _MAX_BLOCKS * self.world_size * 4
        symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))
        with torch.inference_mode(False), torch.no_grad():
            self.buf = symm_mem.empty(
                (max_bytes // 2,), dtype=torch.bfloat16, device=device
            )
        self.hdl = symm_mem.rendezvous(self.buf, group=group)
        assert self.hdl.rank == rank_in_group, (
            f"symm_mem handle rank {self.hdl.rank} != {rank_in_group}; the "
            f"slot offset both collectives index by would be wrong"
        )

    def _accepts_shard(self, shard: torch.Tensor, whole: torch.Tensor) -> bool:
        """Common guard for both collectives, phrased on the per-rank shard --
        the small side for the gather (its input), the small side for the
        scatter (its output). ``whole`` is the world_size-times-larger side.

        Reads only TP-replicated quantities plus pointer alignment, so every
        rank reaches the same verdict and none is left in a collective the
        others skipped. The caching allocator returns 512-byte-aligned storage,
        so the alignment test never diverges in practice -- the same assumption
        ``triton_symm_mem_ag`` already makes.
        """
        nbytes = shard.numel() * shard.element_size()
        return (
            shard.is_contiguous()
            and whole.is_contiguous()
            and shard.element_size() == 2
            and nbytes % _BYTES_PER_CHUNK == 0
            and nbytes <= _MAX_BYTES_PER_RANK
            and nbytes * self.world_size <= self.max_bytes
            and shard.data_ptr() % _BYTES_PER_CHUNK == 0
            and whole.data_ptr() % _BYTES_PER_CHUNK == 0
            and whole.numel() == shard.numel() * self.world_size
        )

    def _launch(self, kernel, input: torch.Tensor, num_chunks: int) -> None:
        num_blocks = min(_MAX_BLOCKS, max(1, triton.cdiv(num_chunks, _BLOCK_THREADS)))
        kernel[(num_blocks, 1, 1)](
            input_ptr=input,
            buffer_ptrs=self.hdl.buffer_ptrs_dev,
            signal_pad_ptr=self.hdl.signal_pad_ptrs_dev,
            num_chunks=num_chunks,
            BLOCK_SIZE=_BLOCK_THREADS,
            RANK=self.rank,
            WORLD_SIZE=self.world_size,
            num_warps=_BLOCK_THREADS // 32,
        )

    def accepts_all_gather(self, output: torch.Tensor, input: torch.Tensor) -> bool:
        return self._accepts_shard(shard=input, whole=output)

    def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None:
        nbytes = input.numel() * input.element_size()
        self._launch(_p2p_all_gather_kernel, input, nbytes // _BYTES_PER_CHUNK)
        gathered = self.buf.view(torch.uint8)[: output.numel() * output.element_size()]
        output.view(torch.uint8).view(-1).copy_(gathered)

    def accepts_reduce_scatter(self, output: torch.Tensor, input: torch.Tensor) -> bool:
        # Equal-split only. pynccl's reduce_scatterv also serves ragged splits,
        # where output.numel() * world_size != input.numel(); those fail here and
        # stay on NCCL, which is what the ragged path is for.
        return self._accepts_shard(shard=output, whole=input)

    def reduce_scatter(self, output: torch.Tensor, input: torch.Tensor) -> None:
        n_elems = output.numel()
        nbytes = n_elems * output.element_size()
        self._launch(_p2p_scatter_kernel, input, nbytes // _BYTES_PER_CHUNK)
        # The push moves raw 128-bit chunks, so the staging area holds the
        # caller's bits, not bf16. Reinterpret before summing -- reading fp16
        # payloads through the buffer's bf16 type would decode garbage.
        stage = self.buf.view(output.dtype)[: n_elems * self.world_size]
        _sum_slots_kernel[(triton.cdiv(n_elems, _SUM_BLOCK),)](
            stage_ptr=stage,
            out_ptr=output.view(-1),
            n_elems=n_elems,
            WORLD_SIZE=self.world_size,
            BLOCK=_SUM_BLOCK,
        )


_UNINIT = object()


def try_build(
    group: dist.ProcessGroup, rank_in_group: int, max_bytes: int
) -> Optional[Sm80P2PCollectives]:
    """Build the shared symmetric buffer, or return None if this group can never
    use it.

    Collective (``symm_mem.rendezvous``): every rank of the group must call it,
    which holds because the caller is itself inside a collective.
    """
    if torch.cuda.is_current_stream_capturing():
        # Cannot allocate or rendezvous under capture; caller retries later.
        return _UNINIT
    try:
        state = Sm80P2PCollectives(group, rank_in_group, max_bytes)
    except Exception as e:  # noqa: BLE001 - any failure means "stay on NCCL"
        logger.warning("sm80 P2P collectives disabled (%s)", e)
        return None
    if state.hdl.multicast_ptr != 0:
        # Multicast exists, so triton_symm_mem_ag's multimem path is the better
        # one; this module is only for architectures without it.
        logger.info("sm80 P2P collectives not used: multicast is available")
        return None
    # Positive confirmation on purpose: without it, a silent fallback to NCCL is
    # indistinguishable from a working fast path when A/B-ing the flags.
    logger.info(
        "sm80 P2P collectives ENABLED: world_size=%d, buffer=%.1f MB",
        state.world_size,
        max_bytes / 1e6,
    )
    return state
