# Note README

本目录维护 `StreamWeave` 实验笔记。默认先读 `experiment-log.md`，其中只保留当前结论、关键里程碑和下一步，不再堆完整聊天过程。

## 读取顺序

1. `experiment-log.md`：当前状态和关键里程碑。
2. `00-overview.md`：项目目标、当前实验口径和主路径。
3. `05-rl-training.md`：当前 V5 GRPO/RL 状态与下一步。
4. `实验跑分.md`：正式跑分主表，只保留可比较结果。
5. `07-key-points.md`：关键结论和避坑。
6. `08-commands-and-tools.md`：当前仍建议使用的命令。
7. `04-sft-training.md`、`02-data-construction.md`、`数据合成.md`：SFT/数据历史和已知问题。

## 当前口径

- 当前主线：`exp3/streamweave_v5/RL` 的 GRPO stepwise 训练。
- 最新事实：GRPO 链路已经跑通，但最近一次 run 在 `39/73` 后非正常中断，没有 checkpoint。
- 当前优先级：先修 checkpoint 保存频率，再优化 `old_log_prob` 和 `update_actor` 慢的问题。
- 历史 V4 SFT 数据合成已经不再是当前阻塞项；第一次 SFT 回评显示退化，不能直接当作可靠结论。

## 更新原则

- 当前状态只写一处：`experiment-log.md`。
- 详细跑分只写一处：`实验跑分.md`。
- 命令只保留当前推荐命令；废弃命令不再展开。
- 历史阶段保留结论，不保留重复执行过程。
- 新的异常和经验结论同步写入 `07-key-points.md`。
