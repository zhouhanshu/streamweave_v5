# 实验总览

## 实验目标

- 构建一个在线流式视频理解系统，支持用户在视频中的任意时间提出问题。
- 在保证响应时机判断能力的前提下，保住 OCR、颜色、空间关系、对象身份等关键视觉细节。
- 通过选择性视觉状态保留，在较低视觉保留率下尽量逼近全保留方法的准确率。

## 场景与 benchmark

- `OVO-Bench`：问题到来时，模型需要判断应该依赖过去信息、当前信息，还是等待未来证据。
- `StreamingBench`：要求时机判断、在线响应和连续交互能力。
- 单纯输入最近窗口帧不够，系统必须有显式记忆机制，但又不能让上下文无限膨胀。

## 核心方法

- `note`：把当前 `chunk` 作为新的视觉锚点长期保留。
- `bridge`：只记录相对上一个关键视觉状态的变化。
- `silent / answer`：控制是否在当前时刻主动回答。
- 数据构造阶段拆成两条 teacher：
  - `state teacher`：只负责 `note / bridge`
  - `response teacher`：只负责 `silent / answer`
- 最终学生模型仍然是统一策略，联合输出 `state + response`。
- 第一版以 `1s chunk` 为基本决策单位，并维护一个 `active note` 作为桥接参考锚点。

## 当前策略

- `Idea 验证` 已完成一轮探索：`StreamWeave-v2` 在 `OVO 1/8` 同 ID 正常样本上与 `SimpleStream / Qwen3-VL-32B-Instruct / recent4` 持平，都是 `113/170 = 66.47%`。
- 当前结论不是方法已经成立，而是问题已经定位：
  - `EPM / FPD / ACR` 有收益。
  - `ASI / HLD / STU / OJR / OCR` 退化抵消收益。
  - 长上下文和 forward 类任务路径仍会制造 error。
- 当前主线已经进入 `streamweave_v4` 的 SFT 数据合成和首次训练准备阶段。
- V4 SFT 链路已经打通：
  - teacher 使用详细的合成 prompt、标注关键帧约束和 retry feedback 生成 XML。
  - 中间表示保存 teacher prompt、production prompt、raw output、attempts、quality、metadata 等审计信息。
  - 最终训练导出只使用 production prompt，不把 teacher-only 指令、关键帧标注提示或 retry feedback 暴露给学生模型。
  - 样本必须整条 accepted 才能进入 `sft_steps.jsonl` 和 `llamafactory_sharegpt.jsonl`。
- 当前 SFT 数据入口：
  - `exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl`
  - raw data root：`exp2/streamweave_v4/dataset/streamweave_data`
- 当前已验证小规模合成：
  - `data_engine/sft/outputs/gemini_final_8`
  - `accepted=7/8`，导出 `136` 条 step-level SFT 样本。
- 第二轮 SFT 数据合成正在进行，目标先跑 `1000` 条样本，输出目录建议为 `data_engine/sft/outputs/gemini_final_1000_w64`。
- 下一步是第一次 SFT 训练和评测；之后再使用新的数据集继续合成 SFT 数据，并开始搭建 RL 框架。
- `TimeChat-Online-139K` 第一波子集保留为备用或参考数据，不作为当前主训练数据。

## V4 协议口径

- 每一步输入：`memory + qa_history + current frames`
- 每一步输出固定为：

```xml
<eta>...</eta>
<answer>...</answer>
<bridge t="...">...</bridge>
<note t="..." frame="..."></note>
```

- `note`：只保存被选中的当前 step 视觉帧和时间区间，不保存文字。
- `note` 必须使用成对标签，`<note .../>` 自闭合格式无效。
- `bridge`：保存文本压缩和绝对时间区间。
- `bridge` 必须覆盖 note 之间以及窗口边界之间的合法 gap；若 Memory 尾部是 open-tail bridge，当前 step 需要按规则继承并延展。
- `qa_history`：按时间顺序记录问题与历史答案，不做 active query 配对。
- `eta`：视频内绝对秒级时间戳，不是相对等待时间。

## 当前对照基线

- `SimpleStream / Qwen3-VL-8B / recent4`
- OVO 全量：`Backward 50.48 / Realtime 81.48 / Forward 43.79 / Total 58.59`
- StreamingBench：`REAL 82.63 / SQA 58.80 / Proactive 52.40`

## 当前已拿到的新结果

- `Qwen3-VL-32B-Instruct plain`
- StreamingBench：
  - `REAL 80.12`
  - `SQA 54.80`
  - `Proactive time 61.20 / answer 60.80`
- OVO：
  - `Backward 60.73`
  - `Realtime 78.15`
  - `Forward 44.70`
  - `Total 61.19`
- `gemini-2.5-pro` exploratory：
  - `OVO 1/4 subset`
  - `Backward 69.42`
  - `Realtime 77.65`
  - `Forward 44.09`
  - `Total 63.72`
  - 当前只作为子集摸底结果，不直接与 `Qwen3-VL-32B-Instruct plain` 全量结论横比
- 相比 `SimpleStream / Qwen3-VL-8B / recent4`：
  - `Backward` 明显提升
  - `Realtime` 小幅下降
  - `Forward` 小幅提升
  - `Total` 仍整体更高

## 当前配置口径

- benchmark：`OVO-Bench + StreamingBench`
- recent-window：`recent_frames_only=4`
- `chunk_duration=1.0`
- `fps=1.0`
- 当前 SFT teacher 后端：`Gemini / gemini-2.5-pro`
- 历史本地教师模型服务：本地 `vllm`，模型为 `Qwen3-VL-32B-Instruct`

## 当前执行路线

1. 已完成 `32B plain baseline` 和 `StreamWeave-v2 OVO 1/8` 探索性对照。
2. 已切到 `exp2/streamweave_v4` 作为当前 SFT 数据合成、训练和后续 RL 主线。
3. 已下载并对齐 `VideoXum` 标注与 `ActivityNet_Captions` 视频。
4. 已构造 `streamweave_data`：合并 `train/val/test`、按 1fps 抽帧、保存关键帧分数和关键帧 ID。
5. 已形成当前 QA 标注入口 `annotations_qa_filter_final.jsonl`。
6. 已打通 V4 SFT 数据合成链路，并完成 `gemini_final_8` 小规模验证。
7. 当前正在跑第二轮 `1000` 条样本合成；完成后先做数据巡检，再启动第一次 SFT。
8. 第一次 SFT 完成后在 OVO/StreamingBench 上回评，并据此决定是否扩大合成规模和启动 RL。

## 当前相关路径

- 主笔记目录：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/note`
- 基线代码目录：`/mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream`
- 历史 StreamWeave v2 方案目录：`/mmu_mllm_hdd/zhouhanshu/test/exp1/stream-weave_v2`
- 旧版 StreamWeave 参考目录：`/mmu_mllm_hdd/zhouhanshu/test/exp1/streamweave`
- StreamWeave V3 目录：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v3`
- StreamWeave V4 目录：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4`
- 当前 SFT 目录：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft`
- 教师模型目录：`/mmu_mllm_hdd/Models/Qwen3-VL-32B-Instruct`
- 第二阶段代码与数据目录：`/mmu_mllm_hdd/zhouhanshu/test/exp2`
- 当前主数据集目录：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data`
- 当前主标注文件：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl`
