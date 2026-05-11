# Anchor README

本目录维护 `StreamWeave` 实验笔记。默认先读 `experiment-log.md`，其中只保留当前结论、关键里程碑和下一步，不再堆完整聊天过程。

## 读取顺序

1. `experiment-log.md`：当前状态和关键里程碑。
2. `00-overview.md`：项目目标、当前实验口径和主路径。
3. `10-source-code-current-state.md`：按当前源码同步的协议、SFT、RL 和脚本口径。
4. `05-rl-training.md`：当前 V5 GRPO/RL 状态、run 记录、checkpoint 和性能问题。
5. `0508实验跑分.md`：当前阶段最新的 OVO 结果快照。
6. `实验跑分.md`：正式跑分主表，只保留可比较结果。
7. `07-key-points.md`：关键结论和避坑。
8. `08-commands-and-tools.md`：当前仍建议使用的命令。
9. `04-sft-training.md`、`数据合成0508.md`、`数据清洗0510.md`：SFT 训练记录、数据历史、新数据清洗和已知问题。
10. `补充实验待办.md`：主线之外的补充实验 checklist。

## 当前口径

- 当前主线：`exp3/streamweave_v5` 的 SFT/RL/eval 和新数据清洗。
- SFT 记录：`04-sft-training.md` 单独记录 answered-full SFT 的数据合成、过滤、LLaMAFactory 训练和 OVO 1/8 评测状态。
- RL 记录：`05-rl-training.md` 单独记录旧 8GPU run、fused/chunked run、checkpoint、reward 指标和性能指标。
- 最新事实：历史 RL 输出已从 `RL/outputs` 清理；旧 GRPO launcher 已删除，只保留最新 fused/chunked GRPO 入口、PPO 入口和 smoke test。
- 当前源码事实：最新 GRPO 入口默认从 answered-full SFT vLLM 兼容模型启动，使用 `save_freq=30`、`resume_mode=auto`、remove padding、fused kernels 和 chunked prefill。
- 当前优先级：等 answered-full SFT 评测落盘；用最新 GRPO 入口启动 reward v2 RL；保证 checkpoint 可恢复；再优化 `old_log_prob` 和 `update_actor` 慢的问题。
- 历史 V4 SFT 数据合成已经不再是当前阻塞项；第一次 SFT 回评显示退化，不能直接当作可靠结论。早期 idea 验证、旧数据构造和旧数据清洗摘要已合并到 `00-overview.md`。

## 更新原则

- 当前状态只写一处：`experiment-log.md`。
- 详细跑分只写一处：`实验跑分.md`。
- 命令只保留当前推荐命令；废弃命令不再展开。
- 历史阶段保留结论，不保留重复执行过程。
- 新的异常和经验结论同步写入 `07-key-points.md`。
- 源码行为变化同步写入 `10-source-code-current-state.md`，再按需要摘到总览和命令页。
