# StreamWeave V5 Notes

本目录记录当前 `streamweave_v5` 的代码、数据、训练和实验结论。旧笔记内容已经整理进本目录，不再保留旧文档堆。

## 阅读顺序

1. `00-代码简介.md`：代码结构、核心协议、eval/SFT/RL 三条链路。
2. `01-数据准备.md`：上游数据、抽帧目录、当前保留文件。
3. `02-SFT数据清洗.md`：SFT 合成、过滤和最终训练文件。
4. `03-RL数据清洗.md`：RL query_events schema、难度清洗和最终筛选数据。
5. `04-SFT训练.md`：LLaMAFactory 训练入口、模型导出和注意事项。
6. `05-RL训练.md`：verl stepwise RL 入口、reward、judge、当前 run 口径和历史 RL 实验归档。
7. `06-实验跑分.md`：可比较的正式结果表。
8. `07-实验记录.md`：时间线、关键结论和下一步。
9. `08-命令和工具使用.md`：当前仍建议使用的命令。
10. `09-文献与历史方案归档.md`：从旧 `docs/` 融合进来的历史方案和论文调研结论。
11. `10-RL方案.md`：当前 RL 方案设计，融合 EXP3 GRPPO proposal 和现有代码实现。

## 当前主线

当前仓库：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
```

当前代码主线是 StreamWeave V5：模型按视频流逐步读取当前帧窗口，维护 `anchor + delta` 形式的图文交错 Memory，并在 QA History 中按时间回答或更新答案。

当前主要训练数据：

```text
dataset2/rl_0516_filter.jsonl
dataset2/sft_0516_4500.jsonl
```

当前 RL 训练入口：

```text
RL/scripts/train_exp9_24.sh
```

当前文档原则：

- 当前事实写在 `note/`。
- 历史材料已经压缩进当前文档；后续只维护 `note/`。
- 顶层 `docs/` 的历史内容已经融合到 `09-文献与历史方案归档.md` 和相关主线笔记中。
- `RL/scripts/EXP3_GRPPO_PROPOSAL.md` 已融合到 `10-RL方案.md` 并删除。
- `RL/scripts/EXPERIMENTS.md` 的关键信息已融合到 `05-RL训练.md`；原文件已删除。
- `RL/verl/` 和 `SFT/LlamaFactory/` 是第三方/框架文档，默认不纳入本目录整理。
- 命令只保留当前可用版本，废弃命令不展开。
- 跑分表和实验时间线分开写，避免结果和过程混在一起。
