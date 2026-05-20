# Token 消耗分析

本文记录 OVO-Bench 上图文交错 memory 与纯文本 memory 的 token 消耗估计。当前统计对象是最终/生成过的 memory 存储量，不是每一步推理时反复读入 Memory before 的累计 API prompt token。

## 统计口径

- 数据集：OVO-Bench full，3035 条评测样本。
- 原始帧数：按 `dataset/ovo/video/<video_id>/*.jpg` 真实抽帧文件计数，总计 `641,970` 帧。
- 图像 token：按 Qwen3-VL 448 分辨率理论视觉 token 计算。
- Qwen3-VL 配置：`patch_size=16`，`spatial_merge_size=2`。
- 单张 448x448 图片视觉 token：

```text
(448 / 16 / 2)^2 = 196
```

- 图文交错总 memory token：

```text
image_tokens + memory_text_tokens
= note_count * 196 + memory_text_tokens
```

- 纯文本 memory token：只统计文本 memory token。
- 文本 token 使用 Qwen tokenizer 抽样估计后外推，图像 token 是精确按 note 数计算。

## Note / 原始帧比例

| 方案 | note 数 | 原始帧数 | note / frame |
|---|---:|---:|---:|
| visual base | 142,467 | 641,970 | 22.19% |
| SFT0513 | 67,792 | 641,970 | 10.56% |
| SFT0516 | 82,120 | 641,970 | 12.79% |
| RL20 | 22,568 | 641,970 | 3.52% |
| RL40 | 30,782 | 641,970 | 4.79% |
| streamtext base | 0 | 641,970 | 0.00% |

结论：RL 显著降低了视觉 note 密度。RL20 只有 SFT0513 的约 `33.3%`，只有 visual base 的约 `15.8%`。

## Memory Token 估计

以下表格按 448x448 图像计算，即每张视觉 note `196` visual tokens。

| 方案 | 图像 token | 文本 token | 总 memory token | 相对 RL20 | 相对纯文本 |
|---|---:|---:|---:|---:|---:|
| visual base | 27.92M | 12.02M | 39.94M | 6.21x | 3.21x |
| SFT0513 | 13.29M | 3.96M | 17.24M | 2.68x | 1.38x |
| SFT0516 | 16.10M | 6.10M | 22.19M | 3.45x | 1.78x |
| RL20 | 4.42M | 2.00M | 6.43M | 1.00x | 0.52x |
| RL40 | 6.03M | 2.40M | 8.43M | 1.31x | 0.68x |
| streamtext base | 0 | 12.45M | 12.45M | 1.94x | 1.00x |

结论：

- RL20 memory token 约为纯文本的 `51.6%`，约省 `48.4%`。
- RL40 memory token 约为纯文本的 `67.7%`，约省 `32.3%`。
- SFT 和 visual base 因为 note 过多，虽然文本 memory 更短，但图像 token 叠加后总 token 超过纯文本。

## 如果按 768 图像计算

Qwen3-VL 在 768x768 下的单张图片视觉 token：

```text
(768 / 16 / 2)^2 = 24^2 = 576
```

这比 448x448 的 `196` tokens/图高：

```text
576 / 196 ≈ 2.94x
```

保持 note 数和文本 token 不变，仅把单张图像 token 从 `196` 改成 `576`，得到：

| 方案 | 图像 token @768 | 文本 token | 总 memory token @768 | 相对纯文本 |
|---|---:|---:|---:|---:|
| visual base | 82.06M | 12.02M | 94.08M | 7.56x |
| SFT0513 | 39.05M | 3.96M | 43.00M | 3.45x |
| SFT0516 | 47.30M | 6.10M | 53.40M | 4.29x |
| RL20 | 13.00M | 2.00M | 15.00M | 1.20x |
| RL40 | 17.73M | 2.40M | 20.13M | 1.62x |
| streamtext base | 0 | 12.45M | 12.45M | 1.00x |

结论：如果 memory 里的 note 图片实际按 768 送入模型，那么当前 RL20 / RL40 的图文交错 memory token 会超过纯文本 memory。RL20 约为纯文本的 `1.20x`，RL40 约为纯文本的 `1.62x`。

## 图文交错等于纯文本的阈值

设：

```text
F = 原始帧数 = 641,970
V = 单张 note 图片 token = 196
T_text = 纯文本 memory token
T_inter_text = 图文交错 memory 的文本 token
r = note / 原始帧比例
```

图文交错与纯文本 token 相等时：

```text
r = (T_text - T_inter_text) / (F * V)
```

代入当前 OVO 统计：

```text
F * V = 641,970 * 196 = 125,826,120
T_text ≈ 12.45M
```

以 RL20 的文本部分 `T_inter_text ≈ 2.00M` 计算：

```text
r = (12.45M - 2.00M) / 125.83M
  ≈ 8.30%
```

以 RL40 的文本部分 `T_inter_text ≈ 2.40M` 计算：

```text
r = (12.45M - 2.40M) / 125.83M
  ≈ 7.99%
```

如果图像分辨率改为 768，则 `V = 576`：

```text
F * V = 641,970 * 576 = 369,774,720
```

以 RL20 的文本部分 `T_inter_text ≈ 2.00M` 计算：

```text
r = (12.45M - 2.00M) / 369.77M
  ≈ 2.82%
```

以 RL40 的文本部分 `T_inter_text ≈ 2.40M` 计算：

```text
r = (12.45M - 2.40M) / 369.77M
  ≈ 2.72%
```

如果忽略图文交错自身文本，只问“图片 token 什么时候单独等于纯文本 token”：

```text
r = 12.45M / 125.83M
  ≈ 9.89%
```

768 下同一问题的阈值为：

```text
r = 12.45M / 369.77M
  ≈ 3.37%
```

因此，在当前 OVO 设置和 448 分辨率下：

- 图文交错 note/frame 控制在约 `8%` 以下时，总 memory token 大概率低于纯文本。
- RL20 为 `3.52%`，RL40 为 `4.79%`，都显著低于阈值。
- SFT0513 为 `10.56%`，SFT0516 为 `12.79%`，visual base 为 `22.19%`，都超过阈值，因此不省 token。

在 768 分辨率下：

- 图文交错 note/frame 需要控制在约 `2.7% - 2.8%` 以下，才可能低于纯文本 memory。
- 当前 RL20 为 `3.52%`，RL40 为 `4.79%`，都会超过该阈值。
- 因此，若按 768 视觉 token 计算，当前 RL20 / RL40 不能再声称 memory token 少于纯文本；只能说明它们比普通 visual base / SFT 的图文交错 memory 更省。

## 可用于论文的表述

在 OVO-Bench full 上，纯文本 memory 约消耗 `12.45M` text tokens。按 Qwen3-VL 在 448 分辨率下每张视觉 note `196` visual tokens 计算，RL20 的图文交错 memory 仅保留 `22,568` 个视觉 note，占原始帧数 `3.52%`，总 memory token 约 `6.43M`，为纯文本 memory 的 `51.6%`。RL40 的 note/frame 为 `4.79%`，总 memory token 约 `8.43M`，为纯文本的 `67.7%`。根据阈值公式，当 note/frame 低于约 `8%` 时，图文交错 memory 的总 token 低于纯文本 memory；RL20 和 RL40 均满足该条件。

如果实际部署或评测按 768 分辨率计算视觉 token，则上述表述需要改写：768 下每张视觉 note 为 `576` tokens，RL20 总 memory token 约 `15.00M`，超过纯文本 memory 的 `12.45M`。此时节省 token 的结论不成立，除非进一步把 note/frame 降到约 `2.8%` 以下，或降低 memory note 的图像分辨率。
