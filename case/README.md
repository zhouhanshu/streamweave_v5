# StreamWeave Case Studies

这个目录保存完整可复查的轨迹示例：每个 case 都复制了完整视频帧，并保留 RL、base、streamtext 三组模型的原始 trace。

前四个 case 是 RL 成功且 base/streamtext 失败的对照；`252` 是 RL/base 对照且 streamtext 也正确；`199` 是三组都失败的 hard case。

## Short Presentation Cases

给别人展示时优先看这个入口：[short_cases_summary.md](short_cases_summary.md)。

这批短 case 只对比 RL 和 text-only，并且把两条轨迹分开放：

- `rl_steps.md`：只放 RL 每一步的当前帧链接和模型原始输出。
- `text_steps.md`：只放 text-only 每一步的当前帧链接和模型原始输出。
- `analysis.md`：总结 RL 为什么对、text-only 为什么错。

## Layout

```text
case/
  README.md
  manifest.json
  case_<sample>_<task>_<name>/
    README.md              # 同题对照摘要
    manifest.json           # case 元信息和各模型结果
    frames.md               # 所有帧的本地图片索引
    images/                 # 该视频完整帧图片
    models/
      rl_exp10_step40/
        result.json
        trace.jsonl          # 原始逐步轨迹
        trace.txt            # 原始可读轨迹，含 prompt/raw/applied/memory
        memory.txt
        step_outputs.md      # 本地图片链接 + 每一步真实输出
      base_qwen3vl_8b_0516/
      streamtext_gemini_flash/
```

## Cases

| Case | Task | Frames | RL | Base | StreamText | Contrast |
| --- | --- | ---: | --- | --- | --- | --- |
| [case_590_asi_clothes_before_put_down](case_590_asi_clothes_before_put_down/README.md) | `ASI` | 18 | `B` / 1 | `C` / 0 | `A` / 0 | RL scored `1`; base and streamtext both scored `0` on the same OVO sample. |
| [case_591_asi_sandwich_before_throw_clothes](case_591_asi_sandwich_before_throw_clothes/README.md) | `ASI` | 46 | `A` / 1 | `cannot answer` / 0 | `None of the above options are correct. The person was eating chips.` / 0 | RL scored `1`; base and streamtext both scored `0` on the same OVO sample. |
| [case_493_asi_inner_tube_leak](case_493_asi_inner_tube_leak/README.md) | `ASI` | 53 | `B` / 1 | `C` / 0 | `C` / 0 | RL scored `1`; base and streamtext both scored `0` on the same OVO sample. |
| [case_1266_acr_white_outfit_snowdrift](case_1266_acr_white_outfit_snowdrift/README.md) | `ACR` | 194 | `C` / 1 | `D` / 0 | `A` / 0 | RL scored `1`; base and streamtext both scored `0` on the same OVO sample. |
| [case_252_epm_second_story_bathroom_sink](case_252_epm_second_story_bathroom_sink/README.md) | `EPM` | 234 | `A` / 1 | `cannot answer` / 0 | `A` / 1 | RL scored `1`; base scored `0`; streamtext also scored `1`, so this is primarily an RL/base contrast case. |
| [case_199_epm_gold_object_nightstand](case_199_epm_gold_object_nightstand/README.md) | `EPM` | 110 | `D` / 0 | `cannot answer` / 0 | `The question cannot be answered from the provided video frames.` / 0 | All three compared models scored `0`; this is included as a hard failure example rather than an RL-success case. |

## Source Outputs

- RL: `outputs/ovo_qwen3vl_rl_exp10_step40_full_eval`
- Base: `outputs/qwen3vl_8b_base_full_0516`
- StreamText: `outputs/streamtext/ovo_gemini_flash_full`
