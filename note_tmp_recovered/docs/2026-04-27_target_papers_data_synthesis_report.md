# 目标论文数据构造与合成复现报告

日期：2026-04-27

阅读对象：

- `05_ReWatch-R1`
- `01_OVO-Bench`
- `15_From Verbatim to Gist`
- `10_VideoThinker`
- `03_StreamBridge`
- `09_Streaming Video Instruction Tuning`
- `07_MMDuet2`
- `21_AURA`
- `Eyes_Wide_Open`

写作口径：

- 只写论文中和数据构造、数据合成、自动标注、人工校验、训练/评测样本格式直接相关的内容。
- 尽量写到可以复现流水线的粒度；如果论文没有公开 prompt、阈值或完整代码细节，会明确标出缺口。
- benchmark 论文和训练论文分开处理：benchmark 重点是题目/时间戳/选项怎么来；训练论文重点是 instruction、CoT、RL 数据怎么合成。

## 05. ReWatch-R1: Boosting Complex Video Reasoning in LVLMs through Agentic Data Synthesis

### 数据目标

这篇的核心不是提出新 benchmark，而是合成一个面向复杂视频推理训练的数据集 `ReWatch`，再用它做 SFT + RLVR。`ReWatch` 有三个互相依赖的部分：

- `ReWatch-Caption-10k`：长视频的时间戳密集描述。
- `ReWatch-QA-170k`：从密集描述中合成的高难度视频依赖 QA。
- `ReWatch-CoT-135k`：用多智能体 ReAct 流水线合成的视频 grounded 推理轨迹。

论文给出的总统计是：`10,989` 个视频，`170,862` 个问题，`135,346` 条 CoT。视频来源分布为 `MiraData 1,748`、`VideoEspresso 1,977`、`VideoMarathon 3,291`、`Video-R1 1,982`、`Vript 1,991`。时长分布为短视频 `<3 min: 3,970`，中等 `3-20 min: 5,472`，长视频 `20-60 min: 1,547`。

### Stage 1: ReWatch-Caption-10k

复现这一步要先准备上述五个公开视频源，然后对每个视频做两级 caption：

1. 用 `Mseg = Gemini2.5-Flash, non-thinking` 低帧率扫描整段视频，把视频切成语义连贯片段 `s_i = [t_start_i, t_end_i]`。目标不是固定长度切块，而是让一个片段尽量覆盖完整事件。
2. 对每个片段再用 `Mcap = Gemini2.5-Flash, non-thinking` 高帧率生成详细描述。输出不是一个段落，而是一组片段内事件 `{c_ij}` 和相对时间戳 `{tau_ij}`。
3. 把片段内相对时间 `tau_ij` 加上该片段起点 `t_start_i`，重对齐成原视频绝对时间 `t_ij = t_start_i + tau_ij`。
4. 把所有片段的 `{事件描述, 绝对时间戳}` 合并成 `C_detail(V)`。

可复现时每条 caption 至少要保存：

- `video_id`
- `duration`
- `source_dataset`
- `segments`: `[start, end, segment_caption]`
- `events`: `[absolute_timestamp_or_interval, event_description]`

论文的 SFT 阶段还给了一个 video-text alignment prompt：要求模型按视频故事线划分 meaningful segments，时间戳连续覆盖完整视频，格式为 `[MM:SS-MM:SS] Description`。但注意这个 prompt 是训练 `ReWatch-Caption` 的响应格式，不等于数据合成时所有内部 prompt 都完整公开。

### Stage 2: ReWatch-QA-170k

这一步从 `C_detail` 合成问题，关键是“详细 caption 能答，简短 summary 不能答”。

1. 用 `Msum = Gemini2.5-Flash-Lite, non-thinking` 把详细 caption 压成 `C_sum`。
2. 用 `Mqa = Gemini2.5-Flash, thinking` 同时读取 `C_detail` 和 `C_sum`，生成原始 `(Q, A)`。生成目标是：答案必须能从 `C_detail` 找到，但不能只靠 `C_sum` 推出。
3. 按预定义 10 类问题控制多样性：事件定位、时间定位、计数、因果、状态变化、OCR 阅读、空间感知、数值推理、物体识别、反事实推理。
4. 三层过滤：
   - `F1 answer verification`：用 `Mverify = GPT-4.1` 检查答案是否由 `C_detail` 支持。
   - `F2 text bias elimination`：把问题单独交给探测模型集合 `Mprobe`，如果模型不看视频/不看 caption 也能答对，就丢弃。论文的 `Mprobe` 是 `Qwen3-235B-A22B-Instruct` 和 `Qwen2.5-VL-72B-Instruct`。
   - `F3 summary bias elimination`：把问题和 `C_sum` 给 `Mprobe`，如果只靠 summary 也能答对，也丢弃。
5. 两个过滤阈值 `theta_text` 和 `theta_sum` 都设为 `1`。按论文公式理解，只要 probe 模型的答对共识达到阈值就不通过；保留的是无法被文本先验或 summary 稳定解出的样本。
6. 通过三层过滤后得到 `85k` 原始 QA，再用 `Mrewrite = Gemini2.5-Flash, non-thinking` 改写成多选题，因此最终是 `85,792` 条多选 + `85,070` 条开放问答，共 `170,862` 条。

各类型数量：事件定位 `21,111`，时间定位 `17,755`，计数 `18,746`，因果 `16,290`，OCR `14,470`，空间感知 `16,417`，物体识别 `18,336`，状态变化 `15,176`，数值推理 `19,252`，反事实 `13,309`。平均问题长度 `70.5 tokens`，平均答案长度 `6.2 tokens`。

### Stage 3: ReWatch-CoT-135k

CoT 不是直接让一个 LLM 写“思考过程”，而是让两个 agent 在 caption 空间中模拟重新看视频：

1. `Reasoner A_R = Gemini2.5-Flash, thinking` 读取问题和历史轨迹，输出下一步 `thought` 和 `action`。
2. `Observer A_O = GPT-4.1` 执行 action，在 `C_detail` 中返回 observation。
3. 论文定义了两个动作：
   - `segment_retrieval(query)`：根据自然语言查询找相关事件时间戳。
   - `segment_query(timestamp)`：根据时间戳取回该事件的详细描述。
4. 循环直到 Reasoner 给出最终答案，得到结构化轨迹 `{T_i, Act_i, Obs_i, A_final}`。
5. 用 `Mconvert = Gemini2.5-Flash-Lite, non-thinking` 把结构化轨迹改写成自然语言 CoT，并显式保留 `<action>...</action>`、`<observation>...</observation>`、`<answer>...</answer>` 标签。
6. 图中还标了 `Rejection Sampling`，但正文没有展开拒绝采样的判据、采样次数或失败处理逻辑。复现时需要自行实现，例如要求最终答案等于 GT、每个 observation 能被 `C_detail` 支持、action 数在合理范围内。

最终 CoT 统计：`135,346` 条；平均/最大 reasoning steps 为 `2.3/11`；平均/最大 reasoning tokens 为 `332.5/2045`。

### 训练数据如何使用

SFT 同时训练三种能力：

- `ReWatch-Caption` 用 video-text alignment prompt 训练基础视频-文本对齐。
- `ReWatch-QA` 用 non-thinking prompt 训练直接回答。
- `ReWatch-CoT` 用 ReWatch thinking prompt 训练 `<action>/<observation>/<answer>` 格式的推理。

RL 阶段不是用全部 QA，而是采样 `40k`：`ReWatch-QA 20k` + `Video-R1-QA 10k` + `LongVideoReason-QA 10k`。RL 用 GRPO，8 rollouts，temperature `0.8`，top_p `0.9`。奖励由最终答案、observation groundedness、reasoning sufficiency 和格式奖励组成：

- `r_acc`：模型最终答案和标准答案是否一致。
- `r_obs`：解析每个 `<action, observation>`，用 judge 模型比较 observation 是否被 `C_detail` 支持。
- `r_rea`：把所有 actions/observations 交给 inference LLM 回答问题，如果能推出 GT，说明推理过程充分。
- `r_fmt`：是否按 `<action>`、`<observation>`、`<answer>` 输出。

论文在 RL 奖励中用 `Qwen3-30B-A3B-Instruct` 作为 `M_infer` 和 `M_judge`。

### 复现缺口

- 没有给出五个源数据集的具体 split、视频去重规则和下载清单。
- 没有给出 Stage 1 的“低帧率/高帧率”具体 fps。
- 没有公开 10 类 QA 生成的完整 prompt，只给了类型定义。
- `Rejection Sampling` 只出现在流程图里，缺少过滤标准。
- 没有每层过滤前后的通过率，因此只能复现流程，难以保证最终分布完全一致。

## 01. OVO-Bench: Online-VideO-Benchmark

### 数据目标

这篇是评测集构造论文，不训练模型。数据合成目标是把普通视频/已有 QA 改造成“在线时刻相关”的样本：问题在某个 `query time` 被提出，模型只能基于当时以前、当时附近或未来逐渐出现的信息作答。

最终规模：

- `644` 个 unique videos。
- `2,814` 个 QA / meta-annotation 样本。
- 视频覆盖 `7` 个大类，时长从几分钟到约半小时，平均 query time 为 `428.89s`。
- 多选题选项数不是固定 4 个，而是 `2-5` 个。

任务分三种在线模式、12 个子任务：

- Backward Tracing：`EPM` episodic memory、`ASI` action sequence identification、`HLD` hallucination detection。
- Real-Time Visual Perception：`STU` spatial understanding、`OJR` object recognition、`ATR` attribute recognition、`ACR` action recognition、`OCR` optical character recognition、`FPD` future prediction。
- Forward Active Responding：`REC` repetition event count、`SSR` sequential steps recognition、`CRR` clues reveal responding。

### 原始视频和标注来源

作者把来源分成两类。

第一类是已有人工标注数据集，只取 `val/test` split，目的是降低和模型训练数据重合的泄漏风险：

- `EPM`：`QA-Ego4D`、`OpenEQA`
- `ASI`：`STAR`、`YouCook2`、`CrossTask`、`HiREST`、`COIN`
- `REC`：`Perception-Test`、`THUMOS14/15`
- `SSR`：`COIN`
- `CRR`：`MovieNet`
- Real-Time Visual Perception 六类任务：`Ego4D`

第二类是自抓取 YouTube 视频，用来补充领域多样性。论文说提供 self-crawled YouTube 视频下载链接，但没有在正文列出完整 URL 清单。

### Meta-annotation 怎么构造

OVO-Bench 的核心中间产物不是最终题面，而是带事件级时间戳的 `meta-annotation`。一条可复现的 meta-annotation 至少应包含：

- `video_id`
- `task_type`
- `query_time`
- `clue_time` 或 `evidence_interval`
- 对 Forward Active Responding，还需要 `answer_time` / `clues_reveal_time`
- `question`
- `answer`
- 必要时保存 `reference_time`、`misleading_time`、`step_interval`

论文用了三条路线得到这些 meta-annotation：

1. 直接复用已有精确事件时间戳。适用于 `QA-Ego4D`、`COIN`、`Ego4D` 这类原本就有 event-level timestamp 的数据。
2. 半自动补时间戳。对只有视频级 QA、缺完整时序定位的数据集，如 `OpenEQA`、`STAR`、`Perception-Test`、`THUMOS`，用时间敏感 Video-LLM，文中点名 `Gemini-1.5`，根据问题和答案生成粗粒度事件时间戳；随后人工修正。
3. 纯人工新标。`SSR` 和 `CRR` 的问题、答案和 ground-truth timestamp 由志愿者构造。

全部 source videos 和 meta-annotations 会经过人工检查，确认时间戳和语义准确后才进入最终集。

### QA 与题面怎么生成

已有 QA 不是简单照搬。作者会筛选能适配在线任务定义的 QA，并按任务框架重写。例如 Real-Time Visual Perception 的题面需要强调当前时刻，人工会把问题改成包含 `What is...`、`What am I...`、`Now`、`Currently` 等实时语境。

对 Real-Time Visual Perception，作者还做自动生成：

1. 从原始长视频中随机采样短 clip。
2. 用 `GPT-4o` 在人类修订 prompt 指导下选择适合出题的候选片段，并生成问题和答案。
3. 混入人工提出的问题以降低 LLM 题型偏置。补充材料明确说 `STU/OJR/ATR` 会邀请志愿者补充候选问题。

志愿者构造 Real-Time Visual Perception 问题的指南包括：

- 判断该视频片段是否适合构造 `STU/OJR/ATR` 类问题。
- 选择合适时刻，例如包含多个对象之间明显空间关系、感兴趣对象、异常属性对象。
- 构造选项时保证选项和视觉内容相关，错误选项应来自视频中的误导信息，且选项长度尽量接近。

`CRR` 不能可靠自动生成。作者尝试用 Video-LLM 直接看视频、或用 LLM 读脚本/字幕生成，但效果不好，因此招募志愿者。志愿者指南是：

- 找有明显“当前动作不完整、结果稍后才揭示”的断裂场景。
- 在 `query_time Q_i` 提问，但答案不能立即确定。
- 继续观看，找到刚好足以回答问题的 `clues reveal time A_i`。
- 时间戳要尽量简洁，`A_i` 应是足够视觉信息刚出现的时刻。

### 多选项怎么合成

Backward Tracing 和 Real-Time Visual Perception 主要用多选题。作者认为 naive options 会泄漏答案，所以做“规则驱动 + 视觉 grounding”的干扰项生成：

1. 输入原始 QA 和对应视频 clip。
2. 提示 Video-LLM 生成视觉相关的错误选项，错误项要来自原视频中的误导信息，而不是随便编。
3. 人工检查选项是否有效。
4. 人工确认后打乱选项顺序，降低位置偏置。

这解释了为什么 OVO-Bench 中很多题有 `misleading time`：同一问题在不同时间附近可能有不同相似视觉线索，模型如果乱采帧会被误导。

### Forward Active Responding 的评测样本

`REC/SSR/CRR` 不只是普通 QA，它们需要模型在时间轴上判断何时回答：

- `REC`：视频一开始提问，让模型在重复动作每次完成时提醒或更新计数。
- `SSR`：视频一开始给出 procedure query，让模型在步骤发生时输出相应步骤。
- `CRR`：在 clue 出现前提问，模型应等到足够信息出现后再答。

对离线模型，作者用 multiple-triggering pipeline 模拟在线：在 `t_i > t_0` 的多个触发时刻重复截取 `Video[0:t_i]`，询问现在是否已有足够信息或让模型输出当前答案。`CRR/SSR` 的答案有效性用 `GPT-4o` 判断，timeliness 用指数衰减分数惩罚过晚回答。

### 复现缺口

- 没有公开完整自动生成 prompt，只在补充材料给了评测 prompt 和人工指南。
- YouTube 自抓视频的完整列表不在论文正文。
- Gemini-1.5 粗时间戳生成后的人工修正标准没有量化，例如允许误差范围、双人标注一致性等。
- 每个子任务的精确样本数主要在图中展示，正文未给完整表格。

## 15. From Verbatim to Gist: MM-Mem / HD-EPIC++

### 数据目标

这篇的“数据构造”有两层：

1. 构造/整理一个新评测与训练资源 `HD-EPIC++`，用于长时第一视角厨房视频理解和 SIB-GRPO 训练。
2. 从任意长视频构造三层 memory tuples：`Sensory Buffer -> Episodic Stream -> Symbolic Schema`。这部分更像模型输入/训练轨迹的合成，而不是传统 QA 数据集。

它同时评测 `Video-MME`、`MLVU`、`VStream-QA` 和自建 `HD-EPIC++`。其中真正新增的是 `HD-EPIC++`。

### HD-EPIC++ 怎么来

`HD-EPIC++` 是基于 `HD-EPIC` 派生出来的 egocentric long-horizon kitchen video benchmark。

复现步骤：

1. 取得原始 `HD-EPIC` 视频和标注。
2. 固定重划分为 `156` 个视频：`105` 个 train，`51` 个 test。
3. train split 用于 `SIB-GRPO` fine-tuning；所有 benchmark 结果只在 held-out test split 上报。
4. 在原 HD-EPIC 基础上扩展更密集、细粒度的监督，强调长时程序理解、实体/状态跟踪、grounded multimodal reasoning。

论文称 `HD-EPIC++` 覆盖 7 类 dense annotation：

- `Recipe`：识别、检索、定位菜谱和步骤。
- `Ingredient`：跟踪食材使用、重量、时间、顺序。
- `Nutrition`：分析食材营养及其在 recipe 过程中的变化。
- `Fine-Grained Action`：理解动作的 what/how/why。
- `3D Perception`：推理物体在 3D 空间中的位置。
- `Object Motion`：跨长视频跟踪物体运动。
- `Gaze`：估计注视点并预测未来交互。

基于这些 dense annotations，作者构造 `5-way multiple-choice VQA benchmark`：

1. 为每类问题设计 question prototypes。
2. 总共设计 `30` 个 question prototypes。
3. 用底层标注实例化成 `26,650` 道多选题。
4. 错误选项从同一数据集内部、基于底层标注采样 hard negatives，以减少捷径学习。

可复现时每道题至少应存：

- `video_id`
- `split`
- `question_type`
- `prototype_id`
- `question`
- `answer`
- `options[5]`
- `correct_option`
- `evidence_annotation`，例如 step interval、ingredient id、object trajectory、gaze point、3D relation 等。
- `negative_source_ids`，用于记录 hard negatives 从哪些标注实体/时间段来。

论文没有列出 30 个 prototypes 的完整模板，这是复现最大缺口。

### Memory tuple 构造

MM-Mem 把视频转成三层记忆，复现时要保存每层中间产物。

第一层 `Sensory Buffer`：

1. 对长视频做内容自适应时间分割，论文举例可用 `PySceneDetect` 得到 clips `c_t`。
2. 对每个 clip 的连续帧计算 inter-frame variation：
   `d_t,i = mean_p || f_t,i(p) - f_t,i-1(p) ||_1`。
3. 计算该 clip 内变化均值 `mu_t` 和标准差 `sigma_t`。
4. 选择 salient indices：`S_t = { i | d_t,i > mu_t + sigma_t }`。
5. 以每个 salient index 为中心截取短 key sub-clip，中心时间记为 `tau_t,i`。
6. 做近重复抑制：按 `d_t,i` 从大到小排序，只有当候选点和已保留点距离至少 `Delta` 帧时才保留。
7. 每个 key sub-clip 编成 memory tuple：`(v_t,i, l_t,i, tau_t,i)`，其中 `v_t,i` 是视频编码器的视觉表示，`l_t,i` 是字幕或自动 caption 文本 trace，`tau_t,i` 是时间位置。

第二层 `Episodic Stream`：

1. 按时间顺序遍历 sensory tuples。
2. 对每个 sensory item `m_t,i` 和当前最新 episode node `e*`，由 memory manager 决定动作 `ADD_NEW / MERGE / DISCARD`。
3. `ADD_NEW` 新建 episode；`MERGE` 把当前 sensory 信息合入最新 episode；`DISCARD` 删除冗余或低 novelty 信息。
4. 对保留下来的视觉表示做聚类，论文举例 `K-means`，并选 representative prototypes 作为 episode summaries。
5. Episodic memory 形态为 `(e_k, l_k, tau_k)`，其中 `l_k` 聚合相关文本 traces，`tau_k` 是时间跨度。

第三层 `Symbolic Schema`：

1. 对每个 episodic unit，用 LVLM extractor `phi` 抽取 salient entities 和 glosses。
2. 用 unifier `pi` 把局部实体映射到全局 prototype set `U`；无法匹配就创建新 prototype。
3. 聚合 linked glosses，构成每个全局 prototype 的文本描述。
4. 构图：episode nodes 和 prototype nodes 之间有 grounding edges；prototype 之间可有 relation edges。
5. 最终 symbolic memory 是 `{(u, t_u)}`。

### SIB-GRPO 的训练数据和奖励

SIB-GRPO 用 `HD-EPIC++ train split` 训练 memory manager。训练样本可理解为：

- state `s = (x, M_old)`：局部 sensory window 加已有 episodic memory。
- action/output `m`：memory manager 生成的 episodic textual trace 或更新决策。
- supervision `Y`：对应 VQA 的 ground-truth answer。

奖励公式由三部分组成：

- `R_vqa(s, m)`：用当前 memory 进行 VQA 的任务奖励，回答越正确越高。
- 长度惩罚：`- beta1 * Length(m)`，抑制冗长 memory。
- 参考策略 KL/约束项：`- beta2 * log(pi_theta(m|s) / pi_ref(m|s))`，防止偏离参考 memory 表达。

训练时从旧策略对同一 state 采样一组 `G` 个候选 traces，计算每个 reward，在组内标准化成 relative advantage，然后用 PPO-style clipped surrogate，也就是 GRPO 风格更新。

论文给出的关键超参：

- base model：`Qwen3-VL-8B`
- text retrieval：`bge-large-en-v1.5` + `bge-reranker-v2-m3`
- visual retrieval：基于 clip-level keyframes，CLIP-style embeddings 来自 base model vision encoder
- LoRA：rank `64`，alpha `128`，dropout `0.05`
- SIB-GRPO：epoch `3`，batch size `8`，learning rate `1e-5`
- `beta = 0.1`，`ppo_clip_epsilon = 0.2`，`kl_penalty_coef = 0.1`
- retrieval：`top_k_sym=5`，`top_k_epi=2`，`top_k_sen=1`，entropy threshold `gamma=0.72`

### 复现缺口

- 30 个 question prototypes 没有逐条公开。
- 7 类 dense annotation 的具体标注来源、自动/人工比例、标注界面和质量控制没有展开。
- hard negative sampling 只说“基于底层标注从数据集内部采样”，没有给具体相似度规则。
- Sensory key sub-clip 的窗口长度和 near-duplicate 的 `Delta` 没有给数值。
- `5-way` 多选和附录 Answer Agent prompt 中 “A-D” 存在不一致，复现时应以数据构造处的 `5-way` 为准，或查看代码仓库确认。

## 10. VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning

### 数据目标

VideoThinker 的核心数据不是人工重新标注视频，而是从已有长视频 QA 中合成“工具调用式、多轮推理轨迹”，再把文本工具轨迹转成 video-interleaved CoT 来训练 VideoLLM。

训练源数据：

- 基于 `CG-Bench`。
- 使用 `10k` multiple-choice QA instances。
- 论文补充材料说合成数据视频时长分布中，超过一半在 `20-40 min`，`40-60 min` 占 `17.1%`。
- 大多数样本包含约 `3-5` 次 tool calls。

教师/学生模型：

- LLM agent：`Qwen3-235B-A22B-MoE`，部署在 `4 x H200`。
- VideoLLM / student：`Qwen2.5-VL-7B`。
- VideoLLM 还被用作 caption 工具。

### 工具体系

合成轨迹前要先实现工具池。论文的工具分两类。

Temporal Retrieval：

- `Clip Retrieval(video path, query, topk)`：把视频切成 `10s` clips，用 `LanguageBind-Video` 编码 clip embedding，按 query 相似度返回 top-k 时间区间。
- `Subtitle Retrieval(video path, query, topk)`：用 `Whisper` 转写音频，再按 query 检索相关字幕片段和时间戳。
- `Subtitle Summary(video path, query)`：用 `Qwen3-30B` 对完整字幕做 query-focused summary。

Temporal Zoom：

- `Frame Zoom(video path, interval)`：给定起止时间，抽取该区间 raw frames。论文例子：如果全局采样 32 帧而 `[0,10]s` 只有 2 帧，则 FrameZoom 可在该区间重采样返回 8 帧，提高局部视觉密度。
- `Subtitle Zoom(video path, interval)`：返回该时间区间内的字幕。
- `Caption Zoom(video path, interval)`：先调用 FrameZoom 抽帧，再用 VideoLLM 生成该区间自然语言 caption，描述关键事件、对象和交互。

VideoLLM 作为 caption 工具的 prompt 很短：要求描述视频中直接有助于回答问题的视觉证据；如果无相关证据，则客观总结视频内容；输入包含 `Question: <question>`。

### 轨迹合成流程

给定训练集 `D = {(v_i, x_i, y_i)}`，其中 `v_i` 是视频，`x_i` 是问题，`y_i` 是标准答案。

1. 对每个视频问题对，先用 VideoLLM 为视频生成全局 caption `c_i`。
2. 把 `x_i`、`c_i` 和 tool system prompt 组合成 LLM agent 的初始输入。
3. 初始化 history `H_0 = {prompt, action_space}`，action space 包括所有工具和 `Answer`。
4. 对每一步 `t = 1..T`：
   - LLM agent 生成 reasoning step。
   - LLM agent 选择 action 和参数，例如 `Clip_Retrieval(text=..., top_k=3)` 或 `CaptionZoom(interval=[a,b])`。
   - 如果 action 是 `Answer`，结束。
   - 否则执行工具，得到 tool observation。
   - 把 `(reasoning, action, observation)` 加入历史。
5. 如果达到最大步数 `T` 仍未结束，就强制让 LLM 基于当前 history 产出最终答案。
6. 合成阶段唯一的视觉访问是 `CaptionZoom`，也就是 LLM 看到的是局部帧的 caption 文本，不直接看 raw frames。

为了增加多样性：

- LLM sampling temperature 设为 `0.7`。
- 每个输入生成 `5` 条不同 reasoning trajectories。
- 只保留预测答案 `ŷ_i` 与 ground truth `y_i` 匹配的轨迹。
- 如果 5 条里没有任何一条答对，则随机选一条候选轨迹保留，而不是丢弃样本。

这一步输出的是文本版 tool trajectory，形如：

- `<think>...</think>`
- `<tool_call>{"name": "Clip_Retrieval", "arguments": ...}</tool_call>`
- `<tool_result>intervals or captions</tool_result>`
- 最终答案

### 转成 video-interleaved CoT

训练 VideoLLM 时不能让模型只学 caption 文本，所以作者把 LLM 合成轨迹中的 `CaptionZoom` 替换为 `FrameZoom`：

1. 保留原来的 reasoning step、tool call 结构和时间区间。
2. 找到所有 `CaptionZoom(interval=[a,b])` 的文本 observation。
3. 用同一区间的 `FrameZoom(interval=[a,b])` 取回实际视频帧。
4. 把原 caption observation 替换为特殊 `<video>` tokens / 对应视频片段。
5. 得到 `D_tool = {(v_i, x_i, r_i, ŷ_i)}`，其中 `r_i` 是 video-interleaved tool CoT。
6. 用 token-level cross entropy 训练 student VideoLLM 复现推理轨迹和最终答案。

这就是论文说的“在 caption space 合成，在 video frame space 训练”。它避免要求教师 LLM 直接理解长视频，同时把工具调用策略蒸馏进 VideoLLM。

### 训练设置

论文给出的 Swift/LoRA 设置：

- model：`Qwen2.5-VL-7B-Instruct`
- dataset：训练脚本中写作 `vcbench/train.jsonl`，正文称基于 `CG-Bench`
- epochs：`3`
- per-device train batch size：`1`
- gradient accumulation：`2`
- learning rate：`1e-4`
- LoRA rank：`64`
- LoRA alpha：`16`
- freeze ViT：`True`
- target modules：`all-linear`
- max length：`200000`
- 最多使用 `4 x H200`

推理阶段另有 confidence-gated tool controller：短于 `600s` 的视频先均匀采 `n` 帧直接答，长视频先检索 top-k clips 直接答；若 confidence `< 0.7`，再触发 tool reasoning。这个不属于数据合成本身，但解释了为什么训练数据要覆盖工具使用。

### 复现缺口

- 没有给出 CG-Bench 10k 样本的具体筛选列表。
- tool system prompt 只在附录中展示了截断版，完整 XML 工具定义可能需看代码。
- 最大步数 `T` 没有在正文明确给值。
- 答案匹配过滤如何判断开放文本等价没有展开；CG-Bench 是多选，复现时可先按选项字母精确匹配。
- 如果 5 条都错仍随机保留，会引入噪声；论文没有给这类样本比例。

## 03. StreamBridge: Turning Offline Video-LLMs into Proactive Streaming Assistants

### 数据目标

StreamBridge 的数据分三块，复现时不要混淆：

1. `Stream-IT`：主模型 streaming instruction tuning 数据，训练 interleaved video-text、多轮实时问答和 proactive response。
2. activation model 数据：约 `180k` temporally annotated video samples，用于训练独立二分类触发器 `ACT`。
3. offline 保能力数据：约 `600k` 样本，来自 `LLaVA-178K`、`VCG-Plus`、`ShareGPT4Video`，用于避免 streaming adaptation 损害普通视频理解能力。

### Stream-IT 的格式

Stream-IT 有两种核心序列格式。

多轮实时理解：

`<V1> <Q1> <A1>, <V2> <Q2> <A2>, ...`

这里每轮问题都紧跟当前视频段，目标是让模型利用历史视觉和文本上下文，同时回答当前时刻问题。

主动响应：

`<Q> <V1> <A1>, <V2> <A2>, ...`

这里 `Q` 是开头的开放式需求或目标指令，例如“show me all steps for cooking”，模型在后续视频片段到达后主动给出适当回答。关键是 `<Q>` 和 `<A>` 中间插入视频段，模拟“先提需求，后等证据出现”。

### Stream-IT 的公开来源部分

作者把带时间戳的公开视频数据重排成 proactive-style interleaved format。组成如下：

- Dense Video Captioning：约 `54k`
  - `ActivityNet`: 约 `10k`，平均 `180s`
  - `Shot2Story`: 约 `36k`，平均 `16s`
  - `ViTT`: 约 `8k`，平均 `210s`
- Sequential Step Recognition：约 `22k`
  - `YouCook2`: 约 `1.3k`，平均 `317s`
  - `COIN`: 约 `11k`，平均 `145s`
  - `HowToStep`: 约 `10k`，平均 `190s`
- Grounded VideoQA：约 `69k`
  - `MovieChat`: 约 `0.8k`，约 `10k frames`
  - `EgoTimeQA`: 约 `10k`，平均 `150s`
  - `QAEgo4D`: 约 `15k`，平均 `495s`
  - `FineVideo`: 约 `43k`，平均 `280s`
- Multi-turn Real-time QA：`StreamingQA-120K`，约 `120k`，平均 `150s`

对 dense captioning 任务，作者额外说明：`ActivityNet` 和 `Shot2Story` 中只有 `20%` 被排成 proactive 格式 `<Q> <V1> <A1>, ...`，其余 `80%` 排成多轮实时格式 `<V1> <Q1> <A1>, ...`，其中 `Q_i` 类似 “What is happening now?”。

### StreamingQA-120K 怎么合成

这是 Stream-IT 中最重要的合成部分。

1. 从 `WebVid-10M`、`Panda-70M`、`InternVid-10M` 取短视频 caption 数据。
2. 用视频-文本语义相似度过滤，保留约 `1.28M` 个 clip，确保视频和 caption 对齐。每个 clip 约 `12s`。
3. 构造长视频：每条合成长视频约由 `10` 个短 clip 拼接，平均长度超过 `150s`。
4. 拼接策略：
   - 随机取一个 clip 作为 anchor `V1`。
   - 用中间帧计算 anchor 与候选池其他 clip 的语义相似度。
   - 选最相似或按相似度分布采样下一个 clip `V2`，无放回。
   - 以 `V2` 作为新 anchor 重复，得到 similarity-ordered list `V1, V2, ...`。
   - 实际取连续 span `V[i:i+k]` 作为一条长视频样本。
5. 保留每个短 clip 的 caption，并给它自然时间戳。
6. 用 `GPT-4o` 基于 clip-level captions 生成 QA，覆盖 8 类任务：
   - `OP` object perception
   - `AR` action recognition
   - `SA` spatial awareness
   - `SR` sequential relationship
   - `CR` causal reasoning
   - `OCR` optical character recognition
   - `UEH` unexpected event handling
   - `EU` event understanding
7. 默认把每个 `<Qi> <Ai>` 插在对应 `<Vi>` 后面，形成 `<V1> <Q1> <A1>, <V2> <Q2> <A2>, ...`。
8. 做两种序列增强：
   - `Random QA Drop`：以 `P_drop=0.55` 把 `<Vi> <Qi> <Ai>` 改成只有 `<Vi>`，防止模型学死固定问答位置。
   - `QA Interval Shift`：以 `P_shift=0.1` 把 `<Vi> <Qi> <Ai>` 改成 `<Qi> <Vi> <Ai>`，让视频段成为问题和答案之间的时间延迟，用于 proactive 训练。
9. 另按 OVO-Bench 思路加入少量 hallucination questions，比例 `0.01%`，问题与已有视频输入无关。

GPT-4o 生成 prompt 的关键约束：

- 问答必须高度相关于 caption，不引入 caption 未提到的主题。
- 忽略 caption 中矛盾或不合理部分。
- 鼓励因果和时间推理，问题要多样。
- 问题必须能通过观看视频回答，避免需要假设。
- 不允许写 “according to the caption / image / frame / photo” 等泄漏来源的措辞，要假设自己真正在看视频。
- 每次从 8 类任务中选择最适合的一类，只生成一条 QA，并输出 question、answer、task type。

### Activation model 数据怎么构造

activation model 是一个独立小模型，用来判断“当前帧是否该触发主 VideoLLM 回复”。它不生成答案，只做二分类。

数据来源五类，共约 `180k` video samples：

- Dense Video Captioning：`ActivityNet Captions`、`Shot2Story`
- Sequential Step Recognition：`YouCook2`、`COIN`
- Temporal Action Detection：`FineAction`、`HACS`
- Grounded VideoQA：`Multihop-EgoQA`、`EgoTimeQA`
- Temporal Video Grounding：`Charades`、`ET-Instruct` 的 TVG subset

构造方式：

1. 每个任务准备一组 prompt templates，训练时从对应任务 prompt 池随机抽一个作为开头 `Q`。
2. 输入格式是 `<Q> <V1> <A1> <V2> <A2> ...`。
3. 对每个带标注时间段的视频片段 `V_i`，把响应 `<A_i>` 插在对应标注时间戳的末尾。
4. 只有每个视频片段最后 `P%` 的帧标为正例，即 response-worthy；前面的帧标为负例。
5. `P` 每个训练样本动态从 `0%-50%` 采样，模拟不同触发宽度。
6. 模型结构：以 `LLaVA-OV-0.5B` 为 base，替换 LM head 为二分类 score head，加 learnable `<ACT>` token；每帧视觉 embedding 后追加 `<ACT>`，取最新帧 `<ACT>` hidden state 做分类。

训练细节：

- activation 输入帧以 `1 FPS` 采样。
- frame representations aggressively pool 到每帧 `16` tokens。
- 只训练 LoRA adapters、projector、score head、learnable activation token。
- 训练 `5` epochs；projector 学习率 `2e-5`；LoRA/score head/ACT token 学习率 `2e-4`；AdamW。

### 主模型训练设置

主 VideoLLMs 包括 `LLaVA-OV-7B`、`Oryx-1.5-7B`、`Qwen2-VL-7B`。

训练时：

- Stream-IT 加上约 `600k` offline 样本：`LLaVA-178K`、`VCG-Plus`、`ShareGPT4Video`。
- 所有主模型 fine-tune `1` epoch。
- learning rate `2e-5`，cosine annealing，AdamW。
- image encoder 冻结，visual projector 和 LLM 全量可训。
- `MaxLen=16384`，用于 round-decayed compression。
- 训练采样 `1 FPS`；超过 `256s` 的视频均匀采 `256` 帧以适配最大长度。

### 复现缺口

- 1.28M clip 的语义相似度过滤阈值没有给。
- clip 拼接的相似度模型和“按分布采样/取最相似”的实际实现有轻微表述不一致，需看代码确认。
- GPT-4o 生成 QA 的 few-shot examples 没在正文完整展示。
- public timestamp datasets 被重排时的字段映射规则没有逐数据集说明。
- 600k offline 数据混合比例/采样权重没有详细展开，只给总量和来源。

## 09. Streaming Video Instruction Tuning / Streamo

### 数据目标

Streamo-Instruct-465K 是一个面向端到端 streaming VideoLLM 的 instruction tuning 数据集。它不使用外部 activation model，而是把“是否该说话”直接做成模型要预测的状态 token：

- `<Silence>`：当前视频无关，或相关信息还没出现。
- `<Standby>`：相关事件已经开始，但还没结束或信息还不够。
- `<Response>`：事件已结束，或已有足够信息，可以输出答案。

训练数据结构是多轮流式对话。完整视频被切成连续时间段 `V(1)...V(N)`，每段显式带时间边界，例如 `<2s-3s><video>`。每个 turn 对应一个视频段和一个响应状态/文本：

`D = {(V(1), R(1)), (V(2), R(2)), ..., (V(N), R(N))}`

系统 prompt 的核心规则是：无相关事件或当前输入与问题无关时输出 `<Silence>`；事件进行中或当前相关但还不能回答时输出 `<Standby>`；事件完全结束或信息足够回答时输出 `<Response>` 并给完整描述，不要给部分答案或猜测。

### 总体规模和来源

作者先按统一协议重标注/清洗得到约 `400K` 个有效 streaming 样本，再合并 `LLaVA-Video` 的 offline video QA，最终形成 `Streamo-Instruct-465K`：

- 总样本：约 `465.8K`
- 总视频：`135,875`
- 来源：`Koala`、`LLaVA-Video`、`ActivityNet`、`QVHighlight`、`YouCook2`、`HACS`、`EgoTimeQA`、`DiDeMo`、`COIN`

任务分布：

- Time-sensitive QA：`34.8%`
- Event Grounding：`26.3%`
- Offline QA：`13.8%`
- Narration：`12.7%`
- Event Caption：`6.7%`
- Action Caption：`5.8%`

视频时长分布：

- `0-30s`: `68,273`，约 `50.25%`
- `30-60s`: `19,153`，约 `14.1%`
- `60-120s`: `21,834`，约 `16.07%`
- `120-240s`: `20,529`，约 `15.11%`
- `240s+`: `6,086`，约 `4.48%`

### Real-time Narration 构造

目标是让模型逐秒输出实时旁白。

1. 把视频按 `1s` 间隔切分。
2. 对相邻两个 `1s` 片段组成的 `2s` 窗口，用 `Qwen2.5-VL-72B` 描述这两秒之间最重要的变化。
3. 描述 prompt 要求只写可观察信息，不推测；关注位置、动作、形状、颜色等变化；只描述主要操作或事件，不列小动作。
4. 把逐秒描述串起来，交给 `GLM-4.5` 后处理。
5. 后处理 prompt 要求去重复、过滤无关小细节、把过长描述缩到约 `5` 个词、合并连续动作。
6. 如果描述重复、单调、无有意义变化、混乱或信息不足，输出 `Negative Sample`，即过滤掉。

训练标签上，未到可叙述变化时是 `<Silence>`；有需要播报的时间点输出 `<Response> narration_text`。

### Event Caption 构造

目标是检测事件边界，并在事件结束时输出事件 caption。

1. 用 `ARC-Hunyuan-Video-7B` 生成 segment-level captions。
2. 用同一模型对每个 caption 做 temporal grounding。
3. 只保留那些“所有 segment captions 的时间跨度互相一致、重叠关系合理、且和原始输出对齐”的视频。
4. 这样既过滤噪声，又得到更尖锐的事件边界。

训练时可以按规则生成状态：

- 事件开始前：`<Silence>`
- 事件进行中：`<Standby>`
- 事件结束后：`<Response> event_caption`

补充材料还有 event rewriting prompt：去掉 “Finally/Then/At the beginning/At the end”等过渡词和结构描述，把 caption 改成独立、简洁句子，不增加额外解释。

### Action Caption 构造

Action Caption 与 Event Caption 相似，但粒度从“事件”变成离散动作或步骤。

复现方式：

1. 复用 event-caption pipeline。
2. 换成 action-oriented prompts，例如“Locate and describe a series of actions or steps”。
3. 做 targeted filtering，使边界更贴近动作完成点。
4. 输出 step/action-level 的 `(start_time, end_time, action_caption)`。

它适合训练过程指导类任务，因为响应应发生在一个动作/步骤结束后。

### Event Grounding 构造

这类任务把事件描述先给模型，模型边看视频边定位该事件。

1. 从 Event Caption annotations 中随机采样 caption。
2. 把 caption 改写成 grounding query，例如 “Watch the following video and temporally localize the event. Respond once it has finished... The given event is: {caption}”。
3. 融合已有 temporal grounding 数据集，扩大事件覆盖。
4. 标签是事件时间段 `(start, end)`。
5. 状态标签自然生成：事件开始前 `<Silence>`，事件开始到结束 `<Standby>`，事件结束后 `<Response> Given event occurred between {start}s to {end}s`。

### Time-sensitive QA 构造

这是最能体现 streaming 的部分，问题的正确答案会随时间变化。

1. 用 `GLM-4.5V` 处理视频，检测多个维度的 change points：
   - object attributes：颜色、大小、状态等
   - spatial positions
   - actions / interactions
   - counts
   - scene or context shifts
2. 围绕一个统一问题生成多个 time-specific answers。例如同一个问题“人现在拿着什么”，在不同时间答案不同。
3. 生成 prompt 要求：
   - 每个问题必须有随时间变化的答案，不变化就不要生成。
   - 每题至少 `2` 个不同答案值。
   - 不允许重复相同 answer value；如果状态回到旧值，也要重新记录对应时间。
   - 每个答案必须给精确时间，时间要反映真实变化点。
   - 如果不确定时间，要重新看对应 segment。
4. 输出形态应是：
   - `question`
   - `answers`: `[{value, time}, ...]`
5. 训练时在每个答案变化点输出 `<Response> answer`；其他时刻视上下文为 `<Silence>` 或 `<Standby>`。

### Offline QA 合并

Offline QA 来自 `LLaVA-Video`，主要用于维持普通离线视频理解能力。它不强调逐帧响应时机，因此复现时应和真正 streaming 任务分开标记，避免误以为所有 465K 都有细粒度响应边界。

### 训练损失

Streamo 直接用 next-token SFT 训练，但由于 `<Silence>` 远多于 `<Standby>/<Response>`，作者只对三个特殊状态 token 加权：

- focal weight：降低易分类样本权重。
- frequency alpha：按类别频率倒数增强稀有状态。
- 普通文本 token 仍用标准 cross entropy。

这一步对复现很重要，因为没有加权时模型容易学成一直 `<Silence>`。

### Streamo-Bench 构造

论文还构造了 `Streamo-Bench`，这是评测集，不是训练集：

- `300` 个视频。
- `3,000` 个 task-specific instances。
- 视频来自 `COIN`、`YouCook2`、`ActivityNet`。
- 包括 forward/backward event grounding、dense caption、narration、Time-sensitive QA。

Grounding 用 mIoU；narration/caption 用与 `Qwen2.5-VL-72B` 的 pairwise win rate；TSQA 同时要求内容和时间戳正确。

### 复现缺口

- 各源数据集重标注后的完整 schema 没在论文正文展开。
- `ARC-Hunyuan-Video-7B` grounding 一致性过滤的阈值没有量化。
- `GLM-4.5V` change-point 检测的内部 prompt 只给了 TSQA 生成 prompt，没给模型如何遍历/复查 segment 的完整流程。
- Offline QA 合并比例虽然可由任务分布估计，但采样/去重规则未详细说明。

## 07. MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn RL

### 数据目标

MMDuet2 构造的是 proactive video dialogue 训练数据，不是传统离线 QA。核心要求是：视频持续播放，用户可以在播放中提问，模型在每个流式 turn 里自己决定输出答案还是 `NO REPLY`。

数据规模约 `52k` 视频，分两大来源：

- Web videos：`50,228` 个，平均时长 `92.7s`，平均每视频 `2.0` 个问题、`6.7` 个答案 turn，来源是 `Live-WhisperX`。
- Ego-centric videos：`2,543` 个，平均时长 `164.4s`，平均每视频 `2.1` 个问题、`5.6` 个答案 turn，来源是 `Ego-Exo4D` 和 `EgoExoLearn`。

数据中有两种 proactive dialogue：

- `1QnA`：一个问题对应多个随视频推进出现的答案。
- `nQnA`：同一个视频中有 `2-4` 个问题，每个问题都有自己的答案序列，模拟用户中途不断切换关注点。

### Scene segmentation 和 captioning

复现时先把每个视频切成语义独立 scene `v_1...v_n`，并为每个 scene 得到 caption `c_1...c_n`。

两类视频的切法不同：

1. 对 `Live-WhisperX` web videos，直接使用字幕句子的时间边界作为 scene 边界。对每个字幕句子对应的视频片段，采样该 scene 内的视频帧，并把这些帧和字幕一起输入 MLLM，生成更详细的 scene caption。
2. 对 `Ego-Exo4D` / `EgoExoLearn`，使用原数据中的 segment-level annotations 作为 scene 边界和语义来源。
3. scene 不要求首尾连续覆盖完整视频；论文强调目标是让 scene 内容相对独立、清晰，并覆盖视频的大部分有效时间。

每个 scene 最少应保存：

- `video_id`
- `source_dataset`
- `scene_id`
- `start_time`
- `end_time`
- `subtitle_or_raw_annotation`
- `sampled_frame_paths`
- `scene_caption`

### QA list 怎么生成

给定一个视频的全部 scene captions，作者让 LLM 生成一个问题 `q` 和长度为 `n` 的答案列表 `[a_1...a_n]`。每个 `a_i` 与 scene `v_i` 对齐：

- 如果 `c_i` 能回答 `q`，则 `a_i` 是基于 `c_i` 的答案。
- 如果 `c_i` 不能回答 `q`，则 `a_i = "NO REPLY"`。

每个视频按长度生成 `2-4` 组这样的 question-answer lists。复现时可以把一组记为：

```json
{
  "question": "...",
  "answers": [
    {"scene_id": 1, "reply_timespan": [start, end], "content": "NO REPLY"},
    {"scene_id": 2, "reply_timespan": [start, end], "content": "..."}
  ]
}
```

这里的 `reply_timespan` 就是对应 scene 的时间段。论文不要求标出 scene 内精确到帧的最早回答时刻；SFT 阶段会把答案放在该时间段末尾，RL 再学习更早、更合适的回答时间。

### 1QnA dialogue 构造

`1QnA` 使用一个问题 `q^j` 和对应答案列表 `[a^j_1...a^j_n]` 构造一条训练样本：

1. 用户在视频开始时提出 `q^j`。
2. 视频按固定帧间隔流入。
3. 如果当前 scene 的答案是 `NO REPLY`，assistant 输出 `NO REPLY`。
4. 如果当前 scene 有答案 `a^j_i`，assistant 应在该 scene 的 `reply_timespan` 内输出该答案。

这类样本训练模型持续监听同一个用户目标，并在后续视频中多次主动补充答案。

### nQnA dialogue 构造

`nQnA` 把同一视频里的 `2-4` 组问题都放进一条对话：

1. 用户可以在任意时间提出某个问题 `q^j`。论文没有公开具体 question time 的采样规则。
2. 如果问题在第 `t` 个 scene 附近提出，作者把该问题在之前 scenes 的答案 `[a^j_1...a^j_{t-1}]` 交给 LLM 汇总成一个 immediate answer `a^j_{1...t-1}`。
3. 模型在用户提问当下先输出这个 immediate answer。
4. 随后对当前问题继续监听后续 scenes `[a^j_t...a^j_n]`，并在对应 scene 时间段内输出新增答案。
5. 当用户提出下一个问题时，模型停止围绕旧问题输出，切换到新问题的答案序列。

这一步的关键是保留“用户问题时间”和“当时之前可见内容的汇总答案”。可复现 schema 应额外保存：

- `question_time`
- `active_question_id`
- `immediate_answer`
- `future_answers_until_next_question`
- `next_question_time`

### Chat template 和 SFT 样本落盘

MMDuet2 不训练额外触发器，而是把“是否回答”做成普通 assistant 文本。系统规则大意是：根据连续进入的视频帧回答问题；回答应只包含自上次回复以来的视频信息；如果当前片段不能回答问题，输出 `NO REPLY`。

消息格式：

1. `system`：proactive dialogue 规则。
2. `user`：包含 `1` 或 `2` 帧视频，或者文本，或者二者都有。
3. `assistant`：输出自然语言答案或 `NO REPLY`。
4. 重复上述 user/assistant turn，直到采样帧用完。

时间戳由 turn 前已经输入的帧数乘以帧间隔得到。例如 `1 FPS` 时，第二帧后的 user turn 对应第 `2s`。

SFT 具体设置：

- 初始化模型：`Qwen2.5-VL-3B`。
- 从 proactive 数据中留出 `1,500` 个 web videos 和 `400` 个 ego-centric videos 作为 RL 数据，其余用于 SFT。
- SFT 输入帧每 `2s` 采样一次。
- 每帧使用 `128` visual tokens。
- 每个 user turn 放 `2` 帧。
- 对 SFT，对应答案放在 `reply_timespan` 的末尾，确保相关事件已经发生，避免模型学会提前幻觉。
- 额外混入 `25k` `LLaVA-Video` offline video QA 和 `25k` `Tarsier2` video captioning，以保持离线视频理解能力。

官方 HF 数据卡进一步显示，公开 annotation 至少包含这些字段：`data_source`、`prompt`、`images`、`ability`、`reward_model`、`extra_info`。`reward_model.ground_truth.answer` 内部保存 `content`、`question`、`reply_timespan`、`role`；`extra_info` 中有 `num_images` 和 `question_id`。视频需要从 `Live-WhisperX`、`EgoExoLearn`、`Ego-Exo4D`、`LLaVA-Video`、`Tarsier2` 分别下载，再按仓库脚本抽帧。

### RL 数据和奖励

RL 不是重新合成问答，而是在留出的 proactive videos 上做短片段 rollout：

1. 每个 RL step 从完整视频中采一个 `20-60s` 短 span。
2. 对 span 之前已经发生的 dialogue turns，直接提供 ground-truth replies，减少长视频 credit assignment 难度。
3. span 内让模型 rollout，判断它何时输出、输出什么。
4. 采样仍是每 `2s` 一帧、每帧 `128` tokens、每 user turn `2` 帧。
5. 用 `GRPO`，每个样本 `4` rollouts，框架是 `SGLang` + `verl`。

每个 ground-truth reply span 是 `(gold_g, t_start_g, t_end_g)`。如果模型在该 span 内输出了 `P` 次回复 `(pred_p, tau_p)`，先用 LLM 计算每次回复相对 `gold_g` 的正确性分数 `s_p`，分数范围改成 `0-4`。然后计算修改版 `PAUC`：

- 在 `t_start` 放一个初始分 `0.5`，避免错误输出比完全不输出还难区分。
- 以时间为 x 轴、回复正确性为 y 轴求曲线下面积。
- 用 `(t_end - t_start) * S` 归一化，其中 `S=4`。
- 越早达到高正确性，`r_PAUC` 越高。

总奖励还加三类惩罚：

- `r_rep`：让 LLM 判断新回复内容是否已被历史回复覆盖；已覆盖回复比例越高，奖励越低。
- `r_in_span`：回复落在任何 GT reply span 外的比例越高，奖励越低。
- `r_pfx`：若新回复和历史回复有超过阈值的最长公共前缀，视为 verbose prefix reply；比例越高，奖励越低。

最终权重：

```text
r = 3 * r_PAUC + 2 * r_rep + 0.5 * r_in_span + 2 * r_pfx
```

### 复现缺口

- 论文没有给出 scene captioning / QA generation 的完整 prompt，也没有明确 MLLM 和 LLM 的具体型号。
- `nQnA` 中用户问题时间的采样策略没有公开。
- `2-4` 个问题如何随视频长度决定，没有给具体规则。
- 正确性评分用的 judge LLM、评分 prompt、`r_pfx` 的最长公共前缀阈值没有公开。
- scene caption 质量过滤、QA list 质量过滤、视频去重规则没有细化。
- HF 数据卡公开了 annotation 和抽帧使用方式，但原视频仍需分别从上游数据集下载，无法只靠一个数据仓库直接复现全部视频文件。

## 21. AURA: Always-On Understanding and Real-Time Assistance via Video Streams

### 数据目标

AURA 的目标是训练一个端到端 streaming assistant，使同一个 VideoLLM 支持三种交互：

- `Real-Time QA`：用户在当前时刻提问，模型立即基于当前及历史可见视频回答。
- `Proactive QA`：用户先提出需求，模型等待未来视频证据足够后只回答一次。
- `Multi-Response QA`：用户提出持续监控类问题，模型在后续多个时间点多次回复。

训练数据总量：

- `115k` streaming video QA samples，约 `1.04B` tokens。
- `59k` in-house offline video QA samples，约 `0.16B` tokens。
- 合计约 `174k` samples，`1.2B` tokens。

QA 类型分布来自图 5：

- Streaming：`66.1%`
- Offline：`33.9%`
- Streaming 内部又包括 `Proactive 25.9%`、`Real-Time 25.8%`、`Multi-Response 8.6%`、`Mixed 5.8%`

视频领域分布包括 `Sport 19.4%`、`Life 16.1%`、`Documentary 12.8%`、`Knowledge 12.4%`、`Show 11.3%`、`Movie 10.3%`、`Course 7.1%`、`Game 5.3%`、`Animation 5.1%`。

### Chunk-wise streaming 格式

训练格式和模型上下文管理绑定：

1. 视频按小 chunk 输入，论文实现中 chunk size 是 `1s`。
2. 每个 chunk 构成一个 user message；如果此时用户提问，就把文本问题和对应视频 chunk 放在同一个 user message 里，否则 user message 只有视频 chunk。
3. 每个 user message 后都有 assistant message。
4. 如果该时刻应该回复，assistant message 是自然语言回答。
5. 如果不应回复，assistant message 是特殊 token `<|silent|>`。

长流上下文用双滑窗截断：

- 视频窗口保留最近 `N=30s`。
- 窗口外的 QA 历史保留最近 `M=10` 个 QA groups。一个 QA group 是一个用户问题加它后续所有非 silent assistant replies。
- 如果 QA group 超出视频窗口，只保留文本内容，丢弃对应视频 chunks 和 `<|silent|>`。

### 五阶段 Coarse-to-Fine 数据引擎

#### Stage 1: Video Preparation

作者从 public internet sources 收集公开视频，类别包括 sports、vlogs、documentaries、encyclopedic content、TV shows、movies、courses、games、animation。

预处理固定为：

1. 视频重采样到 `2 FPS`。
2. 重新编码为 `H.264`，减少解码失败和 codec 不一致。

复现时应保存：

- `video_id`
- `domain`
- `source_url_or_source_id`
- `duration`
- `fps_after_resample = 2`
- `codec = H.264`
- `chunk_size = 1s`

#### Stage 2: QA Synthesis

AURA 用两条合成路线。

第一条用于 `Real-Time QA` 和 `Proactive QA`：

1. 对每个视频，用 MLLM 做 scene segmentation。
2. 对每个 scene 生成 scene-level description。
3. 基于原视频和 scene descriptions，MLLM 生成候选 QA，并同时给出 question timestamp 和 answer timestamp。
4. `Real-Time QA` 要求 `question_timestamp = answer_timestamp`。
5. `Proactive QA` 要求 `question_timestamp < answer_timestamp`。
6. 再让 MLLM 只使用截至 answer timestamp 的视频内容验证候选 QA。
7. Real-Time 的验证项是：问题是否合理、答案是否由视频支持、timestamp 是否准确。
8. Proactive 额外验证：问题在 question timestamp 是否能自然提出，以及 answer timestamp 时信息是否已经足够。
9. 只保留验证通过的 QA。

第二条用于 `Multi-Response QA`：

1. MLLM 先做 scene segmentation，生成 scene descriptions 和每段时间区间。
2. 对每个时间区间的视频 clip，把 scene description 给 MLLM，生成候选问题和问题时间戳。
3. 检查候选问题是否合理，以及是否能在同一视频中产生多个不同时间点的有效答案。
4. 只保留满足多答案条件的问题。
5. 对保留的问题，再把对应视频 clip 给 MLLM，生成多个 answers 和各自 timestamps。
6. 验证每个 answer 是否能在指定 timestamp 被正确推断，并且是否真正回答该问题。
7. 保留通过验证的多响应样本。

`Multi-Response` 的样本 schema 至少应是：

```json
{
  "qa_type": "multi_response",
  "question": "...",
  "question_timestamp": 12.0,
  "answers": [
    {"answer": "...", "answer_timestamp": 18.0},
    {"answer": "...", "answer_timestamp": 27.0}
  ],
  "scene_descriptions": [...]
}
```

#### Stage 3: QA Refinement

对不同 QA 类型做不同增强。

`Real-Time QA` 增强难度：

1. 对每个带 timestamp 的原始 QA，再让 MLLM 生成 `4` 个额外问题。
2. 这 `4` 个问题要覆盖从简单感知识别到高级理解/推理的递增难度。
3. 加上原始问题，共 `5` 个候选。
4. 用 balanced sampling ratio 从 `5` 个候选里采一个。
5. 对采中的问题，基于同一个 video prefix 重新生成答案，得到最终 refined QA。

`Proactive QA` 和 `Multi-Response QA` 增强问法：

1. 预定义一组 proactive / multi-response question templates。
2. 对每个问题随机选一个 template。
3. 用 LLM 改写问题。
4. 改写时保留关键实体、动作、时间引用，不能改变语义和视觉 grounding。

#### Stage 4: Streaming Structuring

这一步把 timestamped QA annotation 转成真正训练样本。

1. 先把同一视频内连续 QA interactions 按时间排成一条 streaming sequence。
2. 对同源视频里不同 QA 类型，按 timestamp 混合成真实交互序列，也就是同一段视频里可以既有 Real-Time、又有 Proactive、又有 Multi-Response。
3. 对 Proactive 和 Multi-Response，在用户问题后立即插入一条短 acknowledgment，表示“已收到请求，后面会继续观察再回答”。真正答案仍在未来 timestamp 输出。
4. 把完整 sequence unroll 成多个训练样本。每个样本只锚定一个非 silent assistant message，称为 target answer。
5. 用 target answer 的 timestamp 决定保留哪个视频窗口和 QA 历史窗口。
6. 训练样本里保留 target answer 之前的交互历史，但监督目标主要是该 target answer。

#### Stage 5: Quality Verification

因为 Stage 4 会截断上下文，原本正确的答案在截断窗口里可能已经没有足够证据。AURA 因此再次做 judge verification：

- Real-Time QA：检查 target answer 是否由 response moment 前可见视觉证据和保留 QA history 支持，是否 factual、temporally consistent、无 hallucination。
- Proactive QA / Multi-Response QA：检查 target answer 是否在时间上合适，以及内容是否由保留视频窗口和 QA history 支持。

只有同时满足 grounding 和 timing 的样本才保留。

### 训练损失和数据监督选择

AURA 的 loss 直接依赖 Stage 4 的 unroll 方式。

每个训练样本只保证最后一个 non-silent assistant message 有完整视觉证据，因此：

1. 所有 `<|silent|>` assistant messages 都参与监督。
2. 最后一个 non-silent assistant message 参与监督。
3. 更早的 non-silent messages 不参与 loss，避免模型从截断后证据不足的上下文中学习幻觉回答。

为避免 `<|silent|>` 数量远大于回答文本，作者对 silent message token 降权：

- non-silent response token weight = `1`
- silent message token weight = `1 / N_silent`

其中 `N_silent` 是该训练样本里的 silent assistant message 数量。

### 复现缺口

- 没有列出 public internet videos 的下载清单、去重规则和版权/过滤规则。
- QA synthesis、refinement、verification 使用的 MLLM/LLM/judge 型号和完整 prompt 未公开。
- `balanced sampling ratio` 没有给具体比例。
- Proactive / Multi-Response 的 template 集合没有公开。
- Quality Verification 只有自然语言标准，没有分数阈值或多 judge 一致性规则。
- `59k` in-house offline video QA 不公开；官方项目页和 GitHub 主要开放模型与实时 demo 部署代码，不包含训练数据生成脚本。

## Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video

### 数据目标

这篇有两个相关但不同的数据产物：

1. `ESTP-Bench`：人工验证的 ego-streaming proactive benchmark，用于评估模型是否能在正确时间窗口回答。
2. `ESTP-IT`：用 `Ego4D` training set 自动生成的大规模 instruction tuning 数据，用于训练 `VideoLLM-EyeWO`。

它的任务设定比普通 online QA 更强：用户问题往往在答案出现前提出，模型必须继续看未来视频，并在一个或多个 valid answer intervals 内回答。

### ESTP-Bench 怎么构造

数据源是 `Ego4D` validation set，使用其中 event narrations 和完成目标的 step annotations。作者先按已有工作过滤掉缺失或不确定 annotation 的视频，再把原始 annotation 转成自然语言格式。

过滤后得到：

- `890` 个 videos。
- 覆盖 `100+` distinct scenes / activities。
- 场景包括 indoor home、cooking、cleaning、desk work、labwork、bakery、grocery shopping 等。
- 最终人工验证 QA：`2,264` 条。
- 每条 answer 平均有 `3.96` 个 valid answer intervals。
- 约 `46%` 的问题是 contextually linked，需要依赖前序问答保持一致性。

### 问题类型设计

ESTP-Bench 不是随便生成 QA，而是按三类 proactive types 标注，共 `14` 个任务类型。

`Explicit Proactive Tasks`：直接依赖可见视觉信息，共 8 类：

- `OR` Object Recognition
- `AP` Attribute Perception
- `TRU` Text-Rich Understanding
- `OL` Object Localization
- `OSC` Object State Change
- `EOL` Ego Object Localization
- `EOSC` Ego Object State Change
- `AR` Action Recognition

`Implicit Proactive Tasks`：需要超出表面观察的推理，共 4 类：

- `OFR` Object Function Reasoning
- `IFR` Information Function Reasoning
- `NAR` Next Action Reasoning
- `TU` Task Understanding

`Contextual Proactive Tasks`：需要历史对话和长时视觉一致性，共 2 类：

- `ORC` Object Relative Context
- `TRC` Task Relative Context

### ESTP-Bench 标注流程

标注是自动生成启发 + 人工定稿。

1. 先用 MLLMs 和 LLMs 基于 Ego4D annotation 自动生成 initial QA pairs。
2. 这些自动 QA 不直接作为最终答案，而是给 annotators 提供灵感，帮助他们找到有价值的场景、问题和实例。
3. 人工标注者根据三类 proactive type 和 14 类任务类型编写或修订问题。
4. 为了评估 just-in-time responsiveness，标注者必须为每个答案标出清晰的 valid answer interval。
5. interval 边界依据是：物体在画面中是否完整可见，或事件的明确 start/end。
6. 有歧义指代的问题会被过滤。例如场景里有多个同类物体时，类似“提醒我那个陶瓷碗的位置”这种无法唯一定位的问题会丢弃。
7. 每条样本的 question、answer、answer interval 都由两名 annotators 验证。

可复现 schema：

```json
{
  "video_id": "...",
  "source": "Ego4D-val",
  "proactive_type": "explicit|implicit|contextual",
  "task_type": "OR|AP|...",
  "question": "...",
  "query_timestamp": 0.0,
  "answer": "...",
  "valid_answer_intervals": [[s1, e1], [s2, e2]],
  "context_linked": true,
  "previous_qa_ids": ["..."],
  "annotator_verification": ["annotator_1", "annotator_2"]
}
```

### ESTP-IT / ESTP-Gen 怎么合成

训练数据来自 `Ego4D` training set。论文说用三阶段 `ESTP-Gen` 生成：

- `60K` single-turn questions。
- `20K` multi-turn questions。
- 每条 instance 都包含 question、answer 和对应 valid answer intervals。

三阶段逻辑如下。

#### Stage 1: one-to-one

用 LVLM 生成视频 captions，并从 captions 中抽取初始 QA。这个阶段的样本是 “一个问题 - 一个答案 - 一个时间区间”。

复现时应保存：

- clip / segment caption
- extracted question
- answer
- single valid interval
- supporting caption span

#### Stage 2: one-to-many

对 Stage 1 的单 interval QA，用 RAG 扩展同一个 answer 的多个 valid intervals。直观做法是把 question/answer/caption 当作检索 query，在同一视频或相关 annotation 中找其他能支持同一答案的片段，再把这些片段转成额外 answer intervals。

论文只说“applying RAG to expand each answer into multiple valid intervals”，没有公开检索库、embedding 模型、召回阈值、去重规则或人工过滤方式。因此复现时只能按上述思想实现近似版。

#### Stage 3: many-to-many

把相关 QA pairs 组合成 coherent multi-turn questions。目标是让模型在后续训练中学会跨历史问答维持上下文一致性。

可复现时应按关系图组织：

- 节点：单条 QA。
- 边：共享 object、共享 task goal、时间相邻、答案互相依赖、或都来自同一 Ego4D goal step。
- 采样 connected components，按时间排序，组成 multi-turn QA episode。

论文没有给具体 relation criteria，上面是根据任务目标可操作的复现假设。

### 训练标签如何从 valid intervals 变成 action

`VideoLLM-EyeWO` 的 action space 包括：

- `a_continue` / stay silent
- `a_response` / 输出回答
- `a_ask_high` / 请求当前时刻高分辨率帧

训练分三阶段。

`Stage 1: Passive Interval Responsiveness`

1. 使用 ESTP-IT 中的 valid answer intervals `T_interval = {[s_i, e_i]}`。
2. 如果当前时刻 `t` 落在某个 valid interval 内，就监督模型倾向于 `a_response` 并生成答案。
3. 但不是简单二分类，而是按 `t` 到 interval 末端 `e` 的距离做线性权重 `f(|t-e| / |s-e|)`，让 interval 内不同位置的响应监督强度不同。
4. 如果 `t` 不在任何 valid interval 内，则监督 `a_continue`。

`Stage 2: Proactive just-in-time responsiveness and accurate answering`

1. 引入第三个动作 `a_ask_high`。
2. 在模型不确定的时间戳集合 `T_uncertain` 上，监督模型先请求高分辨率当前帧 `O_t^h`。
3. 拿到高分辨率帧后，再监督模型判断是否应该 `a_response` 并生成答案。
4. 论文说 `T_uncertain` 的识别细节在 Appendix D，但当前 PDF 没有附录正文，因此这一步无法完整复现。

`Stage 3: Coherence across multi-turn QA`

只用 multi-turn questions 训练，强化历史问答和当前视觉上下文之间的一致性，同时保持前两阶段学到的及时响应能力。

### ESTP-Bench 评测相关数据

ESTP-F1 同时考虑答案质量、响应时机和误报：

- 每个 GT item `g_k` 有答案内容 `o_k` 和有效时间窗口。
- 预测 `p_l` 有文本 `o_l` 和时间 `t_l`。
- answer quality 用 LLM 打分。
- timing score 衡量预测是否落入 valid interval，以及是否足够及时。
- False Positive 会作为 precision penalty。

因此构造 benchmark 时必须保留所有 valid answer intervals，而不是只保留一个最佳时间点。

### 复现缺口

- 本地 PDF 多次引用 Appendix，但 15 页 PDF 中没有附录正文；数据引擎细节、prompt、`T_uncertain` 识别规则都缺失。
- MLLM/LLM 的具体型号没有在数据构造段落明确列出。
- Ego4D validation / training 的具体视频 ID、过滤清单、annotation 转自然语言模板没有公开。
- 自动 QA 只是“给人工灵感”还是有多少被直接保留，比例不清楚。
- 人工标注 valid interval 的允许误差、冲突仲裁规则、标注界面未说明。
- RAG 扩展多 interval 的检索模型、阈值、候选过滤和去重规则缺失。
- 项目页当前仍把 `release code`、`release ESTP-Bench and ESTP-IT` 列为 TODO，因此无法用公开 artifact 校验复现结果。
