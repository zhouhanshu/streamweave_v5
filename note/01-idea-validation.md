# 第一部分：Idea 验证

## 状态

- 当前状态：阶段性完成，已转入数据构造阶段
- 当前目标：保留 `Idea 验证` 的实验结论和错误归因，后续不再把本文件作为主进度入口
- 当前口径：`recent4 / chunk_duration=1.0 / fps=1.0`

## 当前有效实现

- 工作目录：`/mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream`
- 当前推荐入口：
  - `main_experiments/eval_qwen3vl_ovo_api.py`
  - `main_experiments/eval_streamingbench_real_api.py`
  - `main_experiments/eval_streamingbench_sqa_api.py`
  - `main_experiments/eval_streamingbench_proactive_api.py`
- 当前接口层：
  - `lib/openai_api_eval.py`
  - `lib/streamingbench_api_eval.py`
- `StreamWeave` 当前主代码目录：
  - `/mmu_mllm_hdd/zhouhanshu/test/exp1/stream-weave_v2`
  - 独立重写版，当前作为后续实验主线
  - 当前已有自包含的 `OVO` 入口：`eval_ovo.py`
  - 当前默认 backend 是 OpenAI-compatible API
  - 下一步是切到本地部署模型并继续实验
  - `anno_path / chunked_dir` 等默认路径可能暂时需要手动校对
- `StreamWeave` 旧版代码目录：
  - `/mmu_mllm_hdd/zhouhanshu/test/exp1/streamweave`
  - 当前只保留作 prompt / trace / memory 设计的历史参考
- 约束：
  - 只替换接口层和并发调度层
  - prompt、切窗、打分、结果格式与旧脚本保持一致

## 已确认事项

- `SimpleStream` 历史复现结果可直接作为参考基线，不需要再重复搭环境。
- 本地 `vllm` 服务可访问，服务模型名为 `Qwen3-VL-32B-Instruct`。
- 新的 OpenAI SDK API-only 脚本已经替代旧的混合式 API 入口，作为当前主入口。

## Smoke 验证

- 已跑通：
  - `OVO backward` 单样本
  - `StreamingBench real` 单题
  - `StreamingBench sqa` 单题
  - `StreamingBench proactive` 单题
- 当前有效 smoke 结果目录：
  - `ovo_qwen3vl32_api_openai_smoke_backward_fix_20260418`
  - `streamingbench_real_qwen3vl32_api_openai_smoke_fix_20260418`
  - `streamingbench_sqa_qwen3vl32_api_openai_smoke_fix_20260418`
  - `streamingbench_proactive_qwen3vl32_api_openai_smoke_fix_20260418`

## 当前结论

- 评测链路已经可用，当前不再是“能不能跑通”的问题。
- 已拿到 `Qwen3-VL-32B-Instruct plain` 的 `StreamingBench REAL / SQA / Proactive` 全量结果。
- 已拿到 `Qwen3-VL-32B-Instruct plain` 的 `OVO` 全量结果：
  - `Backward 60.73`
  - `Realtime 78.15`
  - `Forward 44.70`
  - `Total 61.19`
- `32B plain benchmark baseline` 已经补齐。
- `OVO` 的解码压力明显高于 `StreamingBench`，全量时不要把并发开太高。
- 当前主实验代码线已经从 `exp1/streamweave` 切换到 `exp1/stream-weave_v2`。
- 上一版中反复暴露的 prompt 问题，当前视为已在 `v2` 中重写解决，不再作为当前主阻塞。
- 当前新的主阻塞已经转为：
  - `stream-weave_v2` 先跑通 API 版实验入口
  - 再切到本地部署模型
  - 再统一 benchmark 路径配置
- `exp1/streamweave` 的详细 prompt / trace 调试记录保留作历史参考，不再继续叠加为当前进度。
- `streamweave` 已支持 debug trace：
  - 一条样本一个目录
  - 目录内包含 `trace.jsonl` 和 `frames/`
  - `jsonl` 里记录帧占位符名称，实际帧按同名文件落盘
- `streamweave` 当前 debug 版本已暴露出 4 个需要立即修正的逻辑问题：
  - 帧选择条件没有正确建模 `active note + bridge history -> current chunk` 的关系
  - 当前 `ChunkSignals` 过于粗糙，无法支撑可靠的 `note / bridge` 决策
  - 首帧缺少强制 `note` 规则
  - `note` 文本语义写偏了，当前实现更像简陋标签，不是“当前帧与 query 的推理记录”
- 上述 4 个问题的第一轮修正已经落地到 `exp1/streamweave`：
  - observation 现在显式生成 `dense_caption`
  - 首帧强制 `note`
  - state 决策输入加入 `active note dense caption` 和 `bridge chain since active note`
  - `note` 改成 `observation / query_link / keep_reason`
  - trace 现在分开记录 `observe prompt/raw`、`state prompt/raw`、`response prompt/raw`
  - trace 目录重跑时会自动清空，避免新旧轨迹混在一起
- 第二轮交互与 trace 修正已经落地：
  - 新增 `memory_trace.xml`
  - `response` 输出新增 `query_closed`
  - 如果模型输出 `query_closed=yes`，后续 step 的 `active_query` 会被清空
  - `response prompt` 明确要求“有证据才能答，不能凭未来动作猜答案”
  - `bridge prompt` 明确要求“只写相对上一条 memory 的增量动作/状态变化”
  - `memory trace` 现在只保留每一步最终可读内容：
    - `<memory><note ...>` 或 `<memory><bridge ...>`
    - `<query>`
    - `<response>`
    - `<answer>`
    - `<query_closed>`
- 第三轮选帧逻辑重构已经落地：
  - `state` 决策不再由单个 prompt 直接输出 `note/bridge`
  - 当前流程改为：
    1. observation 生成 `chunk state`
    2. bridge prompt 生成 `bridge candidate`
    3. assess prompt 输出 `change_large / text_compressible / reason`
    4. `scene cut + change_large + text_compressible` 做加权 gate
    5. 若判为 `note`，再单独调用 note prompt 生成 `note`
  - 当前已删除 `reconstruct prompt / reconstruction_error` 整条链路
  - `ocr` 不再参与第一版选帧判据
  - `state_policy.py` 已收成单文件实现，旧的 `bridge_policy / note_policy / reconstruction / state_gate` 已移除
- `StreamWeave-v2 OVO 1/8` 探索性结果已经完成：
  - 正常完成样本 `113/170 = 66.47%`
  - 与同 ID 的 `SimpleStream / Qwen3-VL-32B-Instruct / recent4` 持平
  - `EPM / FPD / ACR` 有收益
  - `ASI / HLD / STU / OJR / OCR` 有退化
  - 主要工程错误来自 context length 超限和 forward 类任务视频读取路径
- 当前阶段结论：
  - 继续堆 prompt 不能直接解决核心问题。
  - 需要进入数据构造，让模型系统学习关键帧选择、bridge 事实性、证据充分性和回答时机。

## 已记录全量结果

### `Qwen3-VL-32B-Instruct plain` on StreamingBench

- `REAL`: `80.12%` (`2003/2500`)
  - `Action Recognition`: `81.02%` (`286/353`)
  - `Attribute Recognition`: `85.95%` (`263/306`)
  - `Causal Reasoning`: `78.91%` (`101/128`)
  - `Clips Summarize`: `88.96%` (`282/317`)
  - `Counting`: `41.45%` (`80/193`)
  - `Event Understanding`: `75.78%` (`122/161`)
  - `Object Recognition`: `86.10%` (`316/367`)
  - `Prospective Reasoning`: `77.78%` (`84/108`)
  - `Spatial Understanding`: `73.17%` (`180/246`)
  - `Text-Rich Understanding`: `90.03%` (`289/321`)
  - `Errors`: `3`
- `SQA`: `54.80%` (`137/250`)
  - `Errors`: `0`
- `Proactive`:
  - `time`: `61.20%` (`153/250`)
  - `answer`: `60.80%` (`152/250`)
  - `Errors`: `0`

## 当前建议

- `OVO`：先用 `--max_concurrency 4`，若出现解码资源错误则降到 `2`
- `StreamingBench`：默认 `--max_concurrency 4`
- 不再使用旧的混合式 API 入口做本地 `vllm` 正式评测

## 下一步

1. 本文件暂停作为实时入口。
2. 新进展写入 `02-data-construction.md`、`数据合成.md` 和 `experiment-log.md`。
3. 重新评测时再回到 `06-evaluation.md` 记录结果。

## 历史记录说明

- 以下详细 smoke、prompt、trace、memory 调试记录主要对应旧目录 `exp1/streamweave`。
- 当前保留这些内容是为了回顾设计演化，不再作为 `stream-weave_v2` 的实时进度。

## 最近一次代码修正后的 smoke 观察

- `streamweave/eval_ovo.py --backend mock --sample-id 1 --max-chunks 3`
  - 已确认首帧输出为 `note`
  - trace 中已包含 `dense_caption`
  - trace 中已包含 `observe_prompt_text / observe_raw_output / state_prompt_text / state_raw_output / response_prompt_text / response_raw_output`
- `streamweave/eval_ovo.py --backend mock --sample-id 1 --max-chunks 6`
  - 当前新 gate 已工作
  - `memory_trace.xml` 中首帧为 `note`
  - 后续前几步已被压成连续 `bridge`
- `streamweave/eval_ovo.py --backend openai --sample-id 1 --max-chunks 2`
  - 已真实打到本地 `vllm`
  - 当前答案仍为空，`score=0`
  - 但首帧 observation 已能正确识别厨房、sink、bowl、counter、cutting board、bottles、utensils、stove、pan
  - 首帧 `note` 已改成围绕 query 的推理式记录，不再是错误的 `screen shows 27`
  - 第 2 帧也能输出更合理的 scene-shift note
- `streamweave/eval_ovo.py --backend openai --sample-id 1 --max-chunks 4`
  - 已真实打到本地 `vllm`
  - 当前新 gate 在真实后端上可运行
  - `chunk 1` 已被压成 `bridge`
  - `chunk 2` 在当前 prompt 下被重新升为 `note`
  - 说明新架构已经接通，但 `note` 触发条件还需要继续收紧
- `streamweave/eval_ovo.py --backend openai --sample-id 1 --max-chunks 244`
  - 完整样本已重新跑完
  - 最终 `response=C`
  - `score=1`
  - `note_count=13`
  - `bridge_count=231`
  - `first_answer_step=9`
  - 仅回答一次，且 `query_closed=yes`
  - 相比旧版本：
    - `note` 数从 `31` 降到 `13`
    - 首次作答从 `step 6` 推迟到 `step 9`
    - query 关闭逻辑已正常工作
  - 当前仍有明显问题：
    - `step 8` 的 `note` 仍偏早，更多是在“手里拿着 eggplant”阶段就升锚点
    - 后段 `215+` 到 `243` 出现多个 `note`，是否过密不能只靠数量判断，需要逐帧核实
    - 后段出现 `zucchini / green vegetables` 之类的类别漂移
  - 对 `bridge` 与 `response` 的进一步检查：
    - `step 15 / 22 / 236` 的 `bridge` 基本起到了文本连接和变化描述的作用，能较稳定描述“继续切菜 / sink 更清晰 / bowl 被移到 faucet 下”这类增量变化
    - `step 4 / 5 / 6 / 7 / 9` 的 `bridge` 仍然偏强，会把未直接出现的空间关系或动作完成态写进去，例如：
      - `where the bowl is located`
      - `positions it over the bowl in the sink`
    - 这类 bridge 会把“候选物体出现”写成“接近最终证据”，从而污染 response 侧判断
    - `step 238` 一类后段 bridge 仍有类别漂移，把 `eggplant` / `zucchini` 混写
    - `response prompt` 文本上已经写了“证据不足必须 silent、不能猜未来动作”，但当前约束还不够强，`step 9` 仍然抢答
    - 更严重的是，`ASSESS_TASK / RECONSTRUCT_TASK` 在部分早期 step 上直接返回了 `C`，说明完整 query 及选项污染了 state-side prompt；当前 gate 在这几步上实际上依赖了解析默认值而非稳定输出
- `query payload / answer instruction` 拆分修正已落地：
  - `state-side` 现在只看到题面与选项
  - `response-side` 继续承担最终答题格式约束
  - `ASSESS_TASK / RECONSTRUCT_TASK` 不再出现直接输出 `C` 的协议违例
  - 短 smoke (`sample_id=1 --max-chunks 12`) 验证结果：
    - `bad_state_outputs=[]`
    - 说明 `ASSESS_OUTPUT / RECONSTRUCT_OUTPUT` 已不再偷跑成最终选项
  - 但新的 response 行为仍有问题：
    - 首次回答提前到了 `step 5`
    - 当前抢答已经不再来自中间任务偷跑，而是 response 本身门槛过低
- prompt 重写当前约定：
  - 先讨论中文版本
  - 用户确认后再转英文落代码
  - `OBSERVE_TASK` 已完成这一轮转换
  - 标签含义已显式写清
  - 示例统一用 `xx`
  - 不允许再写任何当前样本相关示例
  - `BRIDGE_TASK` 已收紧：
    - 输入只保留当前帧、当前激活 note 帧、以及自该 note 以来的 bridge 链
    - 不再额外喂 `chunk_state / active_query / dense_caption` 等多余文本
    - 作用明确为“用文本描述当前帧相对已有 memory 的变化”
    - 明确禁止猜测、推理、意图和未来动作
  - 其余 prompt 中的具体样本示例也已清掉：
    - `ASSESS / NOTE / RESPONSE` 示例统一改成 `xx`
  - prompt 清理后重新跑了 `OVO sample_id=1 --max-chunks 5`
    - 结果：`response=""`, `score=0`
    - 当前前 `5` 个 chunk 没有抢答
    - `memory_trace.xml` 显示：
      - `step 0 -> note`
      - `step 1 -> bridge`
      - `step 2 -> note`
      - `step 3 -> bridge`
      - `step 4 -> note`
  - 已新增批量 OVO 入口：
    - `exp1/streamweave/eval_ovo_batch.py`
    - 一条命令可直接跑整个标注文件
    - 已支持样本级并发参数 `--max-workers`
    - 运行时支持实时写入 `results.jsonl / errors.jsonl`
    - 终端支持批量进度条
    - 已修掉 `results.jsonl` 重复写入的问题
    - 输出：
      - `results.jsonl`
      - `results.json`
      - `scores.json`
      - `errors.json`
  - `streamweave/eval_ovo_batch.py --backend mock --limit 1 --max-chunks 3`
    - 已通过本地 smoke
    - 能正常打印 OVO 汇总并落盘批量结果文件
  - `OVO 1/4 subset` 运行中阶段统计：
    - 当前已落盘有效结果 `23`
    - 当前 `error` 数 `5`
    - 当前有效结果全部来自 `backward`
    - 当前部分结果：
      - `ASI: 5/7 = 71.43%`
      - `HLD: 1/9 = 11.11%`
      - `EPM: 3/7 = 42.86%`
  - 已确认当前 `response` 看到的历史是截断的：
    - `note_history[-4:]`
    - `bridge_history[-6:]`
    - `current chunk_state`
    - 不看全部历史帧
    - 不看全部历史 `note / bridge`
  - 这会直接伤到当前的 `EPM / HLD`

## `OVO sample_id=1 (EPM)` 完整单样本运行分析

- 运行命令：
  - `streamweave/eval_ovo.py --backend openai --sample-id 1 --max-chunks 244`
- 结果：
  - `response=C`
  - `score=1`
- trace 统计：
  - 总步数 `244`
  - `note=31`
  - `bridge=213`
  - `silent=8`
  - `answer=236`
  - 首次输出答案在 `step 6`
  - 最终保持正确答案直到结尾

### 视频阶段划分

- `step 0-4`
  - 正确建立厨房初始锚点
  - bowl、sink、counter、cutting board、bottles、stove、pan 等实体识别明显改善
  - 首帧强制 `note` 的逻辑已经工作
- `step 4-10`
  - 视频进入冰箱取菜阶段
  - `step 6` 已经提前输出 `C`
  - `step 10` 的保存帧中，画面明确出现手持 eggplant 且 bowl 可见，这一段开始有较强支持
- `step 14-200`
  - 大量时间都在切 eggplant
  - 中段大部分 step 都被压成 `bridge`
  - 说明当前 `bridge` 机制在“长时间重复动作”上已经开始起作用
  - 但 `step 100` 一类位置出现了“尚未真正投放，却写成正在转移到 bowl”的过强描述
- `step 219-234`
  - 后段出现两次明显抖动：
    - `219` 从 `answer=C` 掉回 `silent`
    - `233` 再次掉回 `silent`
  - 原因是模型把 eggplant 的浅色切面误读成“green vegetable”
  - `step 234` 的保存帧中，画面明确显示把 eggplant 放进 bowl，这一步提供了真正直接的证据

### 当前样本暴露出的主要问题

- response 太早：
  - `step 6` 仅凭“从冰箱里拿出深色蔬菜”就开始回答 `C`
  - 这条样本最后答对了，但答对路径偏武断
- note 仍然会过强推断：
  - `step 100` 的保存帧主要还是在 cutting board 上处理 eggplant
  - 轨迹却已经写成 “being transferred into the bowl”
  - 这说明 observation / note 仍会提前脑补动作完成
- 后段颜色与类别混淆：
  - `step 219`、`233` 把当前物体写成 `green vegetables`
  - 导致 response 临时从 `answer` 掉回 `silent`
- state 虽然已经能大量产出 `bridge`，但后段转移相关阶段仍有偏多 `note`

### 当前正向结论

- dense caption 明显有效，实体覆盖和场景描述比旧版好很多
- 首帧 `note`、query-aware `note`、trace 分字段记录都已经生效
- 在这条完整样本上，当前版本已经能跑完整条视频并得到正确答案
- `memory_trace.xml` 已经可用，后续可以直接拿它讨论 memory 的演化，不必每次都看完整 `trace.jsonl`
