# support_pp_mtp_pd 两周开发计划

**分支基线**：`support_pp_mtp_pd` = 最新 `origin/main`（7a03d30）merge PR [#31139](https://github.com/sgl-project/sglang/pull/31139)（colocated PP+spec：draft 单份驻留 last PP rank，跨 rank 只 relay token 级 raw，全 rank 延迟 accept 记账）。

**前置**：完成 `LEARNING_PLAN_one_week.md` 的一周学习。

**背景与相关链接**：
- 需求来源：[#11857](https://github.com/sgl-project/sglang/issues/11857)（dongyibo：1M context 需要 P 开 PP，同时不想失去 D 端 MTP 加速）
- 方案拆解：[#23162](https://github.com/sgl-project/sglang/issues/23162) ShangmingCai Step 1 = 本计划 Week 1；Step 2（decode 侧 PP+verify）→ Backlog
- 对照基线：P 不配 MTP + D 配 MTP 的冷启动模式（[#17212](https://github.com/sgl-project/sglang/pull/17212) / [#17306](https://github.com/sgl-project/sglang/pull/17306)）
- 文件路径均基于 `python/sglang/srt/`

---

# Week 1：MTP + PP + PD（b 模式：P 端 PP+MTP draft prefill + draft KV transfer）

**目标形态**：`P (PP>1, MTP/EAGLE) + D (TP, MTP, 无 PP)`，D 端 draft 从第一个 token 起拥有完整 prompt 上下文。

**现状**（merge #31139 后）：
- ✅ 计算侧就绪：PD-prefill PP 事件循环走通用 extend 分支，last rank capture FULL hidden + `draft_extend_for_prefill`，draft KV 写入 last rank 的 draft pool（`pool_configurator.py` 已按 `pp_size==1 or is_last_rank` 门控）
- ❌ 传输侧缺失：#31139 未触碰任何 `disaggregation/` 文件
- ⚠️ 危险现状：PD+PP+spec 现在能启动但会静默走错传输路径

## W1D1：环境基线 + 止血门控（任务 1.0）

- [ ] 搭起三套可复现部署脚本并留档：
  1. P(TP, MTP) + D(MTP)——非 PP 的 b 模式（现有功能，作为传输逻辑参照）
  2. P(PP2, 无 spec) + D(MTP)——冷启动基线，记录 gsm8k 精度 + accept len + TTFT/TPOT
  3. P(PP2, MTP) + D(MTP)——当前会静默出错的形态，记录实际故障现象（预期：D 端 KV 错乱或 accept 异常）
- [ ] 提交止血 PR：在 `arg_groups/speculative_hook.py`（或 disagg 初始化）加显式 reject——`disaggregation_mode != "null" && pp_size > 1 && spec` 报错，错误信息注明 "not yet supported"（Week 1 完成后放开）
- [ ] P/D spec 参数一致性校验（`speculative_num_draft_tokens`、draft 模型路径/结构）加入 PD 握手校验
- **验收**：三套脚本可一键复现；止血 PR 可独立合入

## W1D2：draft KV 层映射协议——设计 + conn 层走读（任务 1.1a）

- [ ] 精读 sender 注册：`disaggregation/prefill.py:155-200`（`prefill_start_layer/end_layer`、draft ptrs 的 append 位置 186-194）；receiver 注册：`disaggregation/decode.py:420-470`
- [ ] 精读 conn 层（mooncake 为主）的层匹配逻辑与 #17306 的 dst/src 层数比例处理，画出现有映射图
- [ ] 写设计短文（半页）：draft 层全局层号约定——draft 层编号从 `total_kv_layers` 起排 `total..total+M-1`；sender 在 `kv_args` 中显式声明自己持有的 draft 段（PP 下只有 last rank 声明）；receiver 将 draft 段映射到自己 draft pool 区域；索引沿用 target 的 `req_to_token`（indices shared 不变量，`prefill.py:187` 注释）
- [ ] **设计时即覆盖 D 端未来开 PP 的情形**（draft 段路由到 D 的 last rank），避免 Backlog 阶段返工
- **验收**：设计文档 + 现状映射图，找一位 disagg 模块 reviewer 过目（或发 #23162 评论区征求意见）

## W1D3：层映射实现 + 单测（任务 1.1b）

- [ ] sender 侧：`transfer_draft_cache` 条件（`prefill.py:164`，现只感知 layer_shard）改为 PP-aware：本 rank 持有 `draft_token_to_kv_pool` 才声明 draft 段
- [ ] conn 层：按 D2 设计实现 draft 段的识别与落位（mooncake 优先，nixl 跟进或列 TODO）
- [ ] 单测：构造 P(PP2/PP4) × D(TP) 的层映射用例（纯映射逻辑，不需要真 RDMA），校验 draft 段落到 D 的 draft 区域、target 分片互不越界
- **验收**：单测通过；P(PP2,MTP)+D(MTP) 下用调试日志确认 D 端 draft KV 区域收到非零数据且层对齐

## W1D4：spec metadata relay（任务 1.2）

- 现状问题：`req.output_topk_p / output_topk_index / hidden_states_tensor` 只在 last rank 写入（`prefill.py:662`，条件 `draft_input is not None`）；而 `set_buf(req)` 每个 rank 发完最后一个 KV chunk 都执行（`prefill.py:1084`），非 last rank 写的是零值
- [ ] 先确认事实：D 实际从哪个 P rank 拉 aux（读 conn 层 aux 通道代码 + 实验验证）
- [ ] 二选一实现（倾向 A）：
  - A. 约定 aux 一律由 last rank 提供（若 conn 层可指定 aux sender，改动最小）
  - B. 把三个字段沿 PP 输出回环 relay 到所有 rank——做法完全参照 #31139 的 `pp_verify_input_raw` 进 `_pp_prepare_tensor_dict`（`scheduler_pp_mixin.py:1014`）；hidden 是 [bs, hidden_size] 小张量，开销可接受
- [ ] 确认 chunked prefill 下"只有 last chunk 写 metadata"的时序在 PP 下不变
- **验收**：D 端 `_commit_transfer_to_req` 收到的 topk_p/hidden 与 P last rank 产出一致（打日志比对）

## W1D5：资源裁剪 + 端到端联调（任务 1.3 + 1.5）

- [ ] P 节点裁剪：prefill-only 角色跳过 draft 的 decode cuda graph 捕获与 verify buffer（省显存/启动时间）；确认 decode 路径不会被触发
- [ ] 放开 W1D1 的门控（限 `P PP + D 无 PP` 组合），端到端跑通
- [ ] 验证矩阵：

| 组合 | 指标 | 通过标准 |
|---|---|---|
| P(PP2,MTP)+D(MTP) | gsm8k 精度 | 与冷启动基线持平（±0.5%） |
| 同上 | accept len | 显著高于冷启动（尤其长 prompt：长文摘要/引用自测集） |
| P(TP,MTP)+D(MTP) | 回归 | 现有 b 模式不被破坏 |
| P(PP2,无spec)+D(MTP) | 回归 | 冷启动模式不被破坏 |

- **验收**：矩阵全绿；整理成 PR（含 #17212 风格的部署脚本与数据）

**Week 1 stretch（放不下则入 Backlog）**：1M chunked prefill 压测（逐 chunk FULL hidden 的 last rank 显存峰值、与 dynamic chunking 兼容性）。

---

# Week 2：PP + DFlash（colocated）

**目标形态**：单集群 `--pp-size N + --speculative-algorithm DFLASH` 跑通（即放开 `speculative_hook.py:158-160` 的 reject）。

**可行性依据**（已完成的代码分析）：DFlash verify 是**静态定长 block**（`block_size` 启动时定死，无 DSpark 式动态预算），PP 化路线与 MTP 同构；且 #31139 已留好基建——`DFlashDecodePrepareMixin` 抽出（`dflash_info_v2.py`）、线性 verify 的 PP raw 载体参考实现（`DSparkPPVerifyInputRaw`，`dspark_components/dspark_verify.py`）、pool 预留已按 last-rank 门控（`pool_configurator.py` 的 dflash_family 分支）。

**已知的 DFlash 特有障碍**（区别于 MTP，来自代码走读）：
1. draft 依赖 target 的 embed：decode 时直接取 `target_model.get_input_embeddings()`（`dflash_worker_v2.py:1404`）——PP 下 embed 在 first rank，last rank 是 PPMissingLayer 壳
2. draft 依赖 target 的 lm_head + TP 分片 argmax 归并（`_greedy_sample_from_vocab_parallel_head`，`dflash_worker_v2.py:725`）——lm_head 恰好在 last rank，**无需处理**，但需验证
3. draft worker 无条件构建（`dflash_worker_v2.py:183` 的 `build_draft_tp_worker`）——需改为 last-rank-only

## W2D1：设计 + 载体（DFlashPPVerifyInputRaw）

- [ ] 精读 `dflash_worker_v2.py` 全文 + `DSparkPPVerifyInputRaw`（作为线性 verify raw 的模板）
- [ ] 设计并实现 `DFlashPPVerifyInputRaw`：字段 = draft block tokens `[bs, block_size]`、bonus_tokens、accept_lens（线性前缀接受，无树拓扑、无 accept_index）；mixin `DFlashDecodePrepareMixin`（复用 verify 块 KV 预分配）；实现 `to_tensor_dict / from_pp_outputs / build_dummy_for_decode / filter_batch / merge_batch`（照 `EaglePPVerifyInputRaw` 逐一对应）
- [ ] `scheduler_pp_mixin.py:1136` 的 `_pp_raw_cls` dispatcher 加 DFLASH 分支
- **验收**：载体单测（序列化/反序列化/filter/merge/dummy 构造）

## W2D2：worker PP 化——构建与权重

- [ ] `dflash_worker_v2.py.__init__`：加 `_pp_enabled / _pp_is_last_rank`；draft worker（`build_draft_tp_worker`）只在 last rank 构建，非 last rank 置 None 并补齐 `block_size / speculative_num_draft_tokens` 等常量（照 `DSparkWorkerV2` 在 #31139 的改法，`dspark_worker_v2.py:87-105`）
- [ ] embed 问题：让 DFlash draft 在 PP 下自带 embed——优先从 draft checkpoint 加载；参照 #31139 对 DSpark 的同款修复（`deepseek_v4_dspark.py`：last rank target embed 是 PPMissingLayer 壳，draft 改为自己 `VocabParallelEmbedding` 加载）
- [ ] lm_head：验证 last rank 的 target lm_head 可用（PP 天然放在 last rank），`_greedy_sample_from_vocab_parallel_head` 的 TP 归并不受 PP 影响
- [ ] `spec_v2_attn_backends` / cuda graph init：非 last rank 跳过 draft 相关（照 `eagle_worker_v2.py:1119-1140` 的 None 分支）
- **验收**：PP2 下服务能完成初始化（模型加载 + cuda graph 捕获不崩）

## W2D3：forward 流程 PP 分支

- [ ] extend 分支：非 last rank 仅 relay target 前向（`pp_proxy_tensors` 透传，early return）；last rank 保持现有逻辑（FULL hidden capture → `_append_target_hidden_to_draft_kv_by_loc` → `_make_next_draft_input_prefill`）
- [ ] decode 分支：
  - 非 last rank：从 `DFlashPPVerifyInputRaw` 重建线性 verify 输入（无树重建，比 EAGLE 简单：block tokens 即 input_ids + 定长窗口），target verify 前向后 early-return relay proxy hidden
  - last rank：verify 裁决后照 #31139 模式——draft KV 注入（`_append_target_hidden_to_draft_kv_by_loc`）→ 立刻跑下一轮 draft block → 序列化 raw 进 `batch_output.pp_verify_input_raw`
  - idle 分支保持现状（已有）
- [ ] 首步衔接：prefill→decode 过渡用 `build_dummy_for_decode`（bonus 复制成 block），或直接用 `_make_next_draft_input_prefill` 的产物构真 raw（last rank 有完整状态，优先后者）
- **验收**：PP2 + DFLASH 能产出 token（先不管加速比）

## W2D4：accept 记账一致性 + cuda graph

- [ ] 确认线性 verify 复用正式 `req_to_token` 槽位 → 无需 KV move（对照 PR #31139 描述中 DSpark 的同款论述），`batch_result_processor.py:696` 的 isinstance 分支加入 `DFlashPPVerifyInputRaw`——只做 free + `seq_lens += accept_lens`，不做 move
- [ ] 各 rank allocator 账本 lockstep 验证：多迭代长跑后比对各 rank 的 `seq_lens` / 空闲槽位数（临时断言或日志）
- [ ] target verify 的 decode cuda graph 在 PP 下的 capture bs / token 数核对（`decode_cuda_graph_runner.py` 在 #31139 已按 token 数 size proxy buffer）
- [ ] `DFlashDecodePrepareMixin` 的 verify 块预分配在非 last rank 的执行路径确认（DSpark PP 已走通同款，比对即可）
- **验收**：长跑（≥1000 迭代、并发 8+）无 hang、无 KV 账本分叉断言触发

## W2D5：放开门控 + 端到端验证 + 文档

- [ ] 删除 `_handle_dflash` 的 `pp_size != 1` reject（`speculative_hook.py:158-160`），改为版本化的支持声明
- [ ] 验证矩阵：

| 组合 | 指标 | 通过标准 |
|---|---|---|
| PP2 + DFLASH | 输出正确性 | 与 TP + DFLASH 逐 token 对齐（贪心）|
| PP2 + DFLASH | accept len / TPOT | accept len 与 TP+DFLASH 持平；TPOT 相对 PP2 无 spec 有净加速 |
| TP + DFLASH | 回归 | 现有路径不破坏 |
| PP2 + MTP | 回归 | Week 1 / #31139 路径不破坏 |

- [ ] 文档与 PR：设计说明（载体、embed 修复、无 KV move 论证）+ 数据
- **验收**：矩阵全绿，PR 提交

**Week 2 stretch**：DFlash mixed-chunk 已默认禁用（hook 里已处理），确认 PP 下同样生效即可。

---

# Backlog（两周之外，按优先级排序）

1. **DFlash + PD**——回答"能否直接复用 Week 1 的 MTP 实现"：**传输协议层可复用，但不是零修改**。三类工作项区分如下：
   - **可直接复用（无需改动）**：
     - draft pool → PD sender 的接线是算法无关的：`mem_cache/kv_cache_builder.py:54` 的 `get_draft_kv_pool` 对 spec 算法不做区分（且已带 #31139 的 PP last-rank guard），`prefill.py:186` 的 draft 段发送条件只看 pool 是否存在
     - Week 1 建立的 draft 层全局层号映射协议（draft KV 传输内容就是 pool 层指针字节，DFlash 的 draft KV 虽由 target hidden 注入产生，但存储形态与 MTP 无异）
   - **需要新增（DFlash 专属，工作量小但必须做）**：
     - **D 端首步 bootstrap**：`spec_info.py:164` 的 `build_disagg_draft_input` 是 EAGLE-only（`if self.is_eagle(): ... else return None`），DFLASH 走 PD 时 D 端 spec 状态不会重建。需写 DFlash 分支——比 EAGLE 简单：只需 bonus token + KV 账本初始化（`DFlashDraftInputV2` + `prepare_for_decode` 预分配），**不需要** topk_p/hidden_states
     - P 端 metadata：`prefill.py:662` 的 `is_eagle()` 分支对 DFLASH 天然不写 topk/hidden——这恰好正确（DFlash 不需要），确认免除并加注释即可
   - **需要验证/限制（首版收窄范围）**：
     - DFlash + PD 连**非 PP 版都从未被支持/测试过**（上面 bootstrap 缺口即证据），建议先打通 `P(TP,DFLASH)+D(DFLASH)` 再叠 PP——PP 侧此时已是 Week 1+2 的既有能力
     - `draft_window_size` 紧凑滑窗缓存的 pool 布局与线性布局不同，且 `get_draft_kv_pool` 取到的是哪个 pool 需确认（`dflash_worker_v2.py:287` 的 `draft_worker` 属性链）；首版要求关闭 window
     - `scheduler.py:1130` 上方有一行 TODO 注释（"should we fix this when enabling mtp..."），说明 P 端 draft 传输的 wiring 本就欠打磨，排查时一并处理
2. **D 端 PP（P PP+MTP + D PP+MTP，原 Phase 2 / ShangmingCai Step 2）**：
   - 层映射升级为 PP×PP 矩阵（Week 1 D2 设计已预留）
   - 首步 bootstrap 硬缺口：PREBUILT 批次不命中 `_pp_prep_batch_result` 的 spec 分支（`scheduler_pp_mixin.py:1151-1176`）——用 P 传来的 topk/bonus 构首步 raw
   - prealloc/retract/fake-transfer 队列操作在全 PP rank 的顺序一致性
   - **性能 PoC 作为 gate**（Shangming 的 accept-length 抖动 → PP bubble 警告）：D(PP2)+MTP vs D(PP2) 无 spec vs D(TP)+MTP 三方对比，不达标即止损停在 Week 1 形态
3. **1M chunked prefill 压测**（Week 1 stretch 顺延项）
4. **CI**：PD+PP+spec、PP+DFLASH 的 CI 用例（参照 `test/` 现有 PD 与 spec 测试组织）

**明确不在范围内**：dp-attention × PP+spec（DSpark 结构性冲突；MTP/DFlash 未验证且 DFlash 本身尚不支持 dp-attention——三个独立缺口：lm_head TP 归并、idle 不跑 target forward、draft 无 DP context）；overlap schedule under PP；DSpark 的 PD 支持。
