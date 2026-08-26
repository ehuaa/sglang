"""A100 (SM80) MoE tile selection for the humming GEMMs.

humming ships `tune/sm8x.py` with only `get_base_config`, so the M-dependent
tile choice falls through to the generic `tune/base.py`, which was written
against SM90 and never measured on Ampere. `tune/sm90_h20.py` shows the intended
extension point: a device that needs different tiles overrides `get_config`.

Measured on 8xA100-80GB with DeepSeek-V4-Flash-0731 tp8 MoE shapes (w13
n=512 k=4096, w2 n=4096 k=256, 256 experts, bf16 activations over MXFP4
weights). Against the generic path the searched optimum is 21-35% faster on w2
across the whole decode range and 5-30% on w13; see the three rules below for
where that comes from. Only the MoE path is touched -- dense GEMMs keep the
generic choice, which was not measured.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

_registered = False


def _ceil_to(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def register_sm80_moe_heuristics() -> None:
    """Install the tuned SM80 class into humming's device map.

    Must run before the first `get_heuristics_config` call: that function is
    `lru_cache`d, so a later swap would be masked by the cached configs.
    """
    global _registered
    if _registered:
        return

    from humming.config import GemmType
    from humming.tune import heuristics_map
    from humming.tune.sm8x import Sm80Heuristics
    from humming.utils.smem import estimate_smem_size_layer

    class Sm80TunedHeuristics(Sm80Heuristics):
        @classmethod
        def get_config(
            cls,
            meta,
            shape_m: int,
            use_f16_accum: bool = False,
            use_batch_invariant: bool = False,
            gemm_type: GemmType = GemmType.DENSE,
        ):
            config = super().get_config(
                meta=meta,
                shape_m=shape_m,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
            )
            # batch-invariant mode fixes the K tile and drops stream-k on
            # purpose; retuning it would break the guarantee it exists for.
            if not meta.num_experts or use_batch_invariant:
                return config
            return cls._retune_moe(meta, shape_m, gemm_type, config)

        @classmethod
        def _retune_moe(cls, meta, shape_m: int, gemm_type, config: dict) -> dict:
            block_m, block_n, block_k = config["block_shape"]
            warp_m, warp_n, warp_k = config["warp_shape"]
            ctas = config["num_ctas_per_sm"]
            stages = config["num_stages"]

            # 1. Size block_m against the real rows per expert. base.py compares
            #    tile sizes using shape_m / experts / 0.9; that inflation biases
            #    the comparison toward the larger tile, and at 192 rows/expert it
            #    picks 128 (padding 33% of the rows) where 96 divides exactly.
            rows = max(1, shape_m // meta.num_experts)
            if block_m == 128 and _ceil_to(rows, 96) < _ceil_to(rows, 128):
                block_m, warp_m = 96, 48

            # 2. One warp along M at block_m 32. base.py splits the tile across
            #    two warps in M, but the expert block is exactly block_m rows
            #    tall, so each warp gets half a tile that is already short. Only
            #    32 is claimed: at 64 this loses 12% on w2 unless block_n also
            #    doubles, and that pair was not measured widely enough.
            if block_m == 32 and warp_m * 2 == block_m:
                warp_m = block_m

            # 3. Give the K loop back its steps. base.py doubles block_k
            #    until one CTA holds 8 warps; on w2 (shape_k 256) that leaves 2
            #    K iterations. Halve it back -- once, and only when the loop is
            #    that short -- keeping warp_k, so the CTA sheds warps rather
            #    than the K loop shedding steps.
            if (
                meta.shape_k // block_k < 4
                and block_k // 2 >= warp_k
                and meta.shape_k % (block_k // 2) == 0
            ):
                block_k //= 2

            # 4. Occupancy comes from CTAs, not from more warps in one CTA.
            #    base.py leaves num_ctas_per_sm at 1 here -- the 16-bit branch of
            #    get_base_config never sets it, and nothing downstream raises it
            #    -- while sm90_h20 sets 2-3 in every branch. Only the short-tile
            #    case is claimed: at block_m > 32 the measured winners kept one
            #    CTA, so leave those alone.
            #    A second CTA is worth more than the deepest pipeline stages, so
            #    give stages up to buy it (never below 3, which still overlaps).
            if block_m <= 32 and ctas == 1:
                for depth in range(stages, 2, -1):
                    if cls._fits(
                        meta, (block_m, block_n, block_k), gemm_type, depth, 2
                    ):
                        ctas, stages = 2, depth
                        break

            # base.py only clamps num_sms when it left num_ctas_per_sm at 1.
            num_sms = config["num_sms"] if ctas == 1 else cls.get_num_sms()

            # With two CTAs competing for shared memory, pipeline depth past the
            # K loop is stages that never fill. At one CTA there is nothing to
            # win back, and trimming there cost 0.6% on the largest shape.
            if ctas > 1:
                stages = min(stages, max(2, meta.shape_k // block_k))
                while stages > 2 and not cls._fits(
                    meta, (block_m, block_n, block_k), gemm_type, stages, ctas
                ):
                    stages -= 1

            out = dict(config)
            out["block_shape"] = (block_m, block_n, block_k)
            out["warp_shape"] = (warp_m, warp_n, warp_k)
            out["num_ctas_per_sm"] = ctas
            out["num_stages"] = stages
            out["num_sms"] = num_sms
            return out

        @classmethod
        def _fits(cls, meta, block_shape, gemm_type, stages: int, ctas: int) -> bool:
            smem = estimate_smem_size_layer(meta, block_shape, gemm_type, stages)
            return smem * ctas <= cls.max_smem_size

    heuristics_map[80] = Sm80TunedHeuristics
    heuristics_map[87] = Sm80TunedHeuristics
    _registered = True
    logger.debug("Installed A100-tuned humming MoE heuristics for SM80/SM87.")
