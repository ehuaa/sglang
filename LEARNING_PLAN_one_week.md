# SGLang PP + 投机解码 一周学习计划

**目标读者**：只启动过 sglang 服务，对源码结构、PP、MTP/投机解码、DFlash 原理均不了解。
**目标**：一周后能独立读懂 `DEV_PLAN_support_pp_mtp_pd.md` 中每个任务引用的代码位置，并有能力开始 Week 1 的开发。
**方法**：每天 = 原理阅读（上午）+ 代码走读（下午）+ 动手实验（傍晚）+ 自测问题。所有文件路径基于 `python/sglang/srt/`。

---

## Day 1：整体架构与一次请求的生命周期

**原理**：sglang 三进程模型——TokenizerManager（HTTP 入口/分词）→ Scheduler（组 batch/调度，每 GPU rank 一个）→ TpModelWorker/ModelRunner（前向执行）；返回路径经 DetokenizerManager。

**代码走读**（按调用链顺序）：
1. `entrypoints/http_server.py`（找 `/generate` 路由）→ `managers/tokenizer_manager.py`
2. `managers/scheduler.py`：只看主循环 `event_loop_normal`——`recv_requests` → `get_next_batch_to_run` → `run_batch` → `process_batch_result`，其余先跳过
3. `managers/tp_worker.py` 的 `forward_batch_generation` → `model_executor/model_runner.py` 的 `forward`

**动手**：单卡起一个小模型（如 Qwen 系列 1.5B），发一个请求，加 `--log-level debug`，对着日志把上面的链路各环节找出来。

**自测**：一个请求从 HTTP 进来到第一个 token 出去，经过了哪几个进程、哪几个函数？prefill 和 decode 在 scheduler 眼里分别是什么 batch？

## Day 2：KV cache、batch 组织与 ForwardMode

**原理**：`req_to_token_pool`（请求→token 槽位表）与 `token_to_kv_pool`（槽位→各层 KV 数据）两级结构；radix cache 前缀复用；chunked prefill。

**代码走读**：
1. `mem_cache/memory_pool.py`：`ReqToTokenPool`、KV pool 的 `get_contiguous_buf_infos`（后面 PD 传输就靠它拿层指针，务必看懂）
2. `managers/schedule_batch.py`：`ScheduleBatch` 字段——`seq_lens`、`out_cache_loc`、`forward_mode`
3. `model_executor/forward_batch_info.py`：`ForwardMode` 枚举，重点记住 EXTEND / DECODE / IDLE / TARGET_VERIFY / PREBUILT 五个的含义（后两个是投机解码和 PD 专用）

**动手**：打日志观察一个多轮请求的 `seq_lens` / `out_cache_loc` 变化；开 `--chunked-prefill-size` 发长 prompt 观察分 chunk。

**自测**：KV "槽位索引" 和 "KV 数据" 分别存在哪个 pool？为什么 draft 模型可以和 target 共用 `req_to_token` 索引（这是 PD draft 传输的关键不变量）？

## Day 3：投机解码原理 + EAGLE/MTP 代码

**原理阅读**：
- 投机解码基本思想：小模型 draft k 个 token → 大模型一次前向 verify → 按拒绝采样/贪心接受前缀，**无损**（输出分布不变）
- EAGLE 论文（v1/v2）：draft 输入 = target 上一步的 hidden state + token embedding；树形 draft（topk>1）
- DeepSeek-V3 技术报告的 MTP 章节：MTP = 训练时的多 token 预测头，推理时当 EAGLE 式 draft 用（sglang 里 NEXTN）

**代码走读**：
1. `speculative/spec_info.py`：`SpeculativeAlgorithm`、`SpecInput` 基类
2. `speculative/eagle_worker_v2.py` 的 `forward_batch_generation`（约 1173 行起）：extend 分支（target prefill → `draft_extend_for_prefill`）和 decode 分支（draft → verify → draft_extend），这是整个投机解码的骨架
3. `speculative/eagle_info.py`：`EagleDraftInput`（draft 的输入状态）、`EagleVerifyInput`（verify 的树/token）
4. `speculative/eagle_worker_common.py` 的 `run_eagle_verify`（约 436 行）：TARGET_VERIFY 前向 + sample + accept

**动手**：起一个带 EAGLE draft 的模型（或 DeepSeek+NEXTN），观察日志里的 `accept len`；改 `--speculative-num-draft-tokens` 看 acceptance 变化。

**自测**：为什么 verify 是一次"定长 prefill"（TARGET_VERIFY）而不是逐 token decode？accept 之后哪些 KV 槽要释放、`seq_lens` 怎么推进？draft KV 和 target KV 分别在何时被写入？

## Day 4：Pipeline Parallelism（PP）

**原理**：按层切分模型到多个 stage；microbatch 流水；stage 间只传 hidden states；lm_head 在 last rank、embed 在 first rank；气泡（bubble）概念。

**代码走读**：
1. `managers/scheduler_pp_mixin.py` 的 `event_loop_pp`（68 行起）：microbatch 环（`mbs[mb_id]`）、`_pp_recv_proxy_tensors` / `_pp_send_dict_to_next_stage`（hidden 接力）、输出 dict 的环形回传（last rank 的采样结果绕回 rank 0）
2. `model_executor/forward_batch_info.py` 的 `PPProxyTensors`
3. 辅助阅读：PR #5724（sglang PP 初版）、#11852（新事件循环）的描述部分

**动手**：两卡 `--pp-size 2` 起服务（`--disable-overlap-schedule`），发请求；对照日志理解 rank 0 和 rank 1 各自在做什么。

**自测**：为什么 PP 下每个 rank 都有自己的 Scheduler 和一份一模一样的 batch 元数据？next_token_ids 在 last rank 产生后如何回到 rank 0？

## Day 5：PD 分离（Prefill/Decode Disaggregation）

**原理**：P 节点只做 prefill、D 节点只做 decode；KV cache 经 RDMA（mooncake/nixl）从 P 传到 D；bootstrap 握手、metadata（首 token、logprob 等）随传。

**代码走读**：
1. `disaggregation/prefill.py`：KV sender 注册（约 155-200 行，`get_contiguous_buf_infos` + `prefill_start_layer/end_layer`）、`send_kv_chunk` + `set_buf`（约 1060-1090 行）
2. `disaggregation/decode.py`：receiver 注册（约 420-470 行）、`_commit_transfer_to_req`（metadata 落到 req，约 1670-1790 行）
3. `disaggregation/utils.py` 的 `MetadataBuffers`（约 290-490 行）：注意 spec 相关的 `output_topk_p / hidden_states` 字段
4. `disaggregation/decode_schedule_batch_mixin.py` 的 `process_prebuilt`：转移来的请求如何以 PREBUILT 模式进入 decode
5. 辅助：PR #17212 的描述（P 不开 spec + D 开 spec 的 bugfix，含完整部署命令，可作为动手模板）

**动手**：单机双实例 PD 部署（照 #17212 的脚本改），跑通一个请求；然后 D 加 `--speculative-algorithm EAGLE` 复现"冷启动模式"。

**自测**：draft KV 什么情况下会随 target KV 一起传？（`prefill.py:186` 的条件）D 端怎么重建投机解码的初始状态？（`build_eagle_disagg_draft_input`）

## Day 6：精读 #31139（colocated PP + 投机解码）

**原理**：读 PR #31139 描述全文，核心设计三条——draft 整体只驻 last rank；每轮跨 rank 只传 token 级 raw（`EaglePPVerifyInputRaw`）；所有 rank 用同一份 accept 结果做延迟的 KV 记账，保证各 rank 的 allocator 账本逐字节一致。

**代码走读**（都在本分支上，已 merge）：
1. `speculative/eagle_info.py` 尾部：`EaglePPVerifyInputRaw`（载体：draft/bonus tokens、树拓扑、accept 结果、`build_dummy_for_decode`）
2. `speculative/eagle_worker_v2.py`：`_build_verify_input_from_pp_raw`（1330 行，各 rank 本地重建树）、decode 分支里 last rank 的 "draft_extend 后立刻跑下轮 draft 并序列化 raw"（1302-1326 行）
3. `speculative/eagle_worker_common.py`：`run_eagle_verify` 的 PP 分支（非 last rank 前向后 early-return；PP 下跳过 worker 内 KV move）
4. `managers/scheduler_components/batch_result_processor.py` 696-785 行：全 rank 延迟 accept 记账（KV move + free + seq_lens advance）
5. `managers/scheduler_pp_mixin.py` 的 `_pp_prep_batch_result`（1124 行）：raw 的接收与 dispatcher `_pp_raw_cls`

**动手**：两卡 PP2 + MTP/EAGLE 起服务，验证 `accept len` 正常；在 `_pp_prepare_tensor_dict` 加临时日志打印 raw 内容。

**自测**：为什么 KV move 必须从 worker 挪到 batch_result_processor 且每个 rank 都执行？首个 decode 步的 raw 从哪来（`build_dummy_for_decode` 的作用）？

## Day 7：DFlash 原理与代码 + 总结

**原理**：DFlash = 块扩散（block diffusion）式 draft——draft 模型一次前向并行生成固定 `block_size` 个候选 token（非自回归），target 线性 verify 接受前缀。与 EAGLE 的区别：无树、verify 窗口定长、draft 是独立小模型（需 `--speculative-draft-model-path`）、draft KV 来自 target hidden 注入。

**代码走读**：
1. `arg_groups/speculative_hook.py` 的 `_handle_dflash`（147 行起）：现有限制清单（CUDA only、无 dp-attention、无 PP——158 行的 reject 就是 Week 2 要放开的）
2. `speculative/dflash_worker_v2.py`：`__init__`（157 行起，`build_draft_tp_worker`、block_size 推导）、`forward_batch_generation`（1295 行起）——extend 分支（FULL hidden capture → `_append_target_hidden_to_draft_kv_by_loc` 铺 draft KV）和 decode 分支（"Draft a fixed block"、用 target 的 embed/lm_head、`_greedy_sample_from_vocab_parallel_head`）
3. `speculative/dflash_info_v2.py`：`DFlashDecodePrepareMixin`（verify 块 KV 预分配，已被 DSpark PP 复用——Week 2 的现成基建）、`DFlashDraftInputV2`
4. 对照读 `speculative/dspark_components/dspark_verify.py` 的 `DSparkPPVerifyInputRaw`：这是"线性 verify 的 PP raw 载体"参考实现

**动手**：若无 DFlash draft 模型权重，则做纸面练习：把 Day 6 的 EAGLE PP 时序图改画成 DFlash 版本，标出哪些环节不同（无树重建、无 KV move、draft 依赖 target embed/lm_head）。

**自测**（直接对应 Week 2 开发）：DFlash 的 verify 布局为什么是静态的？PP 下 last rank 没有 target embed（在 first rank），DFlash draft 怎么办？线性 verify 为什么不需要 accept 后的 KV 搬移？

---

## 贯穿一周的建议

- 每天维护一份自己的笔记（术语表 + 时序图），Day 7 汇总
- 遇到读不懂的函数，用 "打日志 + 跑最小实验" 代替硬读
- 学习期间把 `DEV_PLAN_support_pp_mtp_pd.md` 当索引：每学完一天，回去看哪些任务的引用位置已经能看懂了
- 硬件：Day 1-3 单卡即可，Day 4-6 需 2 卡，PD 实验单机双实例即可
