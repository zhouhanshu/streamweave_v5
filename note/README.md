# Note README

本目录用于维护 `StreamWeave` 主实验笔记。默认入口文件是 `experiment-log.md`。

## 读取顺序

1. `experiment-log.md`
2. `00-overview.md`
3. 当前阶段文件：目前优先读 `data_engine/sft/README.md`、`数据合成.md`、`04-sft-training.md` 和 `08-commands-and-tools.md`
4. `02-data-construction.md`
5. `实验跑分.md`
6. `06-evaluation.md`
7. `07-key-points.md`
8. `08-commands-and-tools.md`

## 更新原则

- 只保留当前有效信息，避免把聊天过程原样堆进笔记。
- 错误步骤不保留完整过程，只在 `07-key-points.md` 中写“避免再犯”的摘要。
- 命令文档只保留当前推荐命令；旧命令只做一句废弃说明。
- 用户给出新进展后，至少同步更新：
  - `experiment-log.md`
  - 对应阶段文件
- 新的坑、异常、经验结论统一写入 `07-key-points.md`。

## 目录说明

- `00-overview.md`：实验目标、当前策略、总体路线
- `01-idea-validation.md`：Idea 验证阶段历史记录
- `02-data-construction.md`：当前数据构造阶段主记录
- `03-data-cleaning.md`：数据过滤、校验与清洗计划
- `04-sft-training.md`：SFT 数据入口与训练计划
- `05-rl-training.md`：RL 训练计划
- `06-evaluation.md`：评测入口、对比原则与后续回评规则
- `实验跑分.md`：本地跑分、外部参考表、smoke/debug 记录和可比性口径
- `07-key-points.md`：关键结论、避坑记录
- `08-commands-and-tools.md`：当前有效命令、环境与入口
- `09-streamweave-proposal-draft.md`：完整提案原文，作为长期参考，不参与日常压缩
- `数据合成.md`：第二阶段数据下载、解压、抽帧、过滤的详细过程记录

## 当前外部主文档

- `../data_engine/sft/README.md`：StreamWeave V4 当前 SFT 数据合成链路、输出文件、验证逻辑和使用命令。
- `../代码重构.md`：StreamWeave V4 重构背景与对齐事项。
- 历史参考：`../../streamweave_v3/docs/实验计划.md`、`../../streamweave_v3/docs/数据构造.md`。

## Git 说明

- `note/` 是独立 Git 仓库。
- Git 提交和推送默认由用户手动处理。
- 除非用户明确要求，否则不要处理 `git commit` / `git push`。
