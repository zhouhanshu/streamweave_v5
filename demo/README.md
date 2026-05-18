# StreamWeave 真实 Trace Demo

这个 demo 展示的重点不是“哪些题答对了”，而是模型在真实评测 trace 里如何维护图文交错 memory：

- `<delta>` 是模型写出的文本事件变化。
- `<anchor>` 是模型保留下来的真实视频帧。
- 后续 step 的 prompt 会把这些 `<delta>` 文本和 `<anchor>` 图片交错喂回模型。
- 每个 case 下面的 XML 都来自对应 trace 的 `backend_result.text`，不改写模型输出。

来源：

| 项目 | 路径 |
|---|---|
| 模型 | `models/qwen3vl_sft_0516_step50` |
| 结果 | `outputs/ovo_qwen3vl_sft_0516_step50_1of8/results.jsonl` |
| Trace 根目录 | `outputs/ovo_qwen3vl_sft_0516_step50_1of8/traces/` |

---

## Case 1: `1512_2`，蓝盒子什么时候变成可答

问题：

```text
The man is holding a blue box in his hand, what's in the blue box?
Decide whether the existing visual content ... provides enough information for answering the question.
Answer only with "Yes" or "No".
```

这里最能体现图文交错：模型先把“蓝盒子出现但内容未知”写进 memory，所以答 `No`；等盒子打开，delta 写出 keychain，才答 `Yes`。

### Step 250 的真实输入片段，Memory before

下面是 `trace.jsonl` 里 step 250 之前的 memory 关键片段，按 prompt 里的图文交错形式展示：

```xml
<delta t="1212.0-1231.0">The camera shows a close-up of the man's face, then at 1214s a close-up of a woman's eyes, which remain visible until 1231s, then at 1231s a close-up of hands holding a blue box.</delta>
```

`<anchor t="1231.0-1232.0">`

![](assets/1512_2/001231.jpg)

`</anchor>`

```xml
<delta t="1232.0-1236.0">The hands continue to hold the blue box, then at 1234s the camera shows the man's face again, and at 1236s a close-up of a woman's eyes is shown.</delta>
```

`<anchor t="1236.0-1237.0">`

![](assets/1512_2/001236.jpg)

`</anchor>`

```xml
<delta t="1237.0-1249.0">The close-up of the woman's eyes remains visible, then at 1238s a close-up of the man's eyes is shown, and at 1239s the man and woman are seen again, with the woman holding the small box and pointing at it, and the man looking at it, then at 1247s the camera shows a close-up of the man's face, and at 1249s a close-up of the blue box is shown.</delta>
```

`<anchor t="1249.0-1250.0">`

![](assets/1512_2/001249.jpg)

`</anchor>`

到这里，memory 里只有“蓝盒子”，没有“盒子里是什么”，所以之前的答案都是 `No`。

### Step 249 的真实模型输出：仍然不可答

```xml
<state>The video has shown various scenes, including a car falling into water, people reacting to a trapdoor, and a man and woman interacting with a small box. The current frames show the man holding the blue box and looking at it. The question asks what is in the blue box. The video has not provided any information about the contents of the blue box, so the answer is no.</state>
<answer>No</answer>
<delta t="1237.0-1249.0">The close-up of the woman's eyes remains visible, then at 1238s a close-up of the man's eyes is shown, and at 1239s the man and woman are seen again, with the woman holding the small box and pointing at it, and the man looking at it, then at 1247s the camera shows a close-up of the man's face, and at 1249s a close-up of the blue box is shown.</delta>
<anchor t="1249.0-1250.0"></anchor>
```

Step 249 输出里的 anchor 对应这张图：

![](assets/1512_2/001249.jpg)

### Step 250 的当前帧：盒子被打开

| 1252s | 1253s | 1254s |
|---|---|---|
| ![](assets/1512_2/001252.jpg) | ![](assets/1512_2/001253.jpg) | ![](assets/1512_2/001254.jpg) |

### Step 250 的真实模型输出：答案变成 Yes

```xml
<state>The video has shown various scenes, including a car falling into water, people reacting to a trapdoor, and a man and woman interacting with a small box. The current frames show a close-up of the man holding a blue box, and then the box is opened to reveal a keychain. The question asks what is in the blue box. The current frames provide the answer, as the box is opened to reveal a keychain.</state>
<answer>Yes</answer>
<delta t="1250.0-1253.0">The man continues to hold the blue box, then at 1251s the camera shows the man's face, and at 1252s the box is opened to reveal a keychain.</delta>
<anchor t="1253.0-1254.0"></anchor>
<delta t="1254.0-1255.0">The keychain is visible inside the opened blue box.</delta>
```

把这个真实输出渲染成图文交错，就是：

```xml
<delta t="1250.0-1253.0">The man continues to hold the blue box, then at 1251s the camera shows the man's face, and at 1252s the box is opened to reveal a keychain.</delta>
```

`<anchor t="1253.0-1254.0">`

![](assets/1512_2/001253.jpg)

`</anchor>`

```xml
<delta t="1254.0-1255.0">The keychain is visible inside the opened blue box.</delta>
```

这个 case 的重点：答案不是靠最后一句解释，而是靠 memory 里真的出现了 “box is opened to reveal a keychain” 这个 delta，并且保留了对应视觉 anchor。

---

## Case 2: `580`，ASI，菜谱步骤链

问题：

```text
What does the person do after now add the rolls into the hot oil and cook for 6-8 minutes until they are light golden brown?
Options:
A. put the mixture on edge roll ...
B. once it is done remove from the microwave ...
C. once the rolls turn brown remove and place them on a plate with paper towels and serve
D. broccoli slaw in a bowl add ...
```

这题需要模型把 458 秒的视频压成一条“菜谱步骤 memory”。下面不是人工总结，是 step 91 前 memory 里的相关 delta 摘录。

### Step 91 的真实输入片段，Memory before

```xml
<delta t="1.0-15.0">The screen displays the title "Five Spice Shrimp Spring Roll Recipe" from 1s to 4s, then at 4s transitions to show two women in a kitchen setting, with red rectangular bars appearing and disappearing. The younger woman on the right begins speaking at 6s and continues to speak throughout the current frames.</delta>
```

![](assets/580/000001.jpg)

```xml
<delta t="21.0-27.0">The title card "Prepare the Filling for the Spring Rolls" is displayed from 21s to 24s, then at 24s it changes to "Start with 1 Pound of Large Peeled and Cooked Frozen Shrimp", which remains on screen until 27s.</delta>

<delta t="123.0-127.0">The cut green onions remain on the black surface, then at 127s the screen transitions to a title card for "12oz Bag Broccoli Slaw".</delta>

<delta t="162.0-171.0">A hand enters the frame at 162s and begins to add ingredients to the broccoli slaw in the bowl, continuing to add ingredients until 171s.</delta>

<delta t="282.0-292.0">The hand continues to mix the broccoli slaw in the bowl with a spoon, then at 292s adds chopped shrimp to the broccoli slaw.</delta>

<delta t="303.0-306.0">The screen transitions to a title card "Prepare to Wrap the Spring Rolls" at 302s, which remains on screen until 306s.</delta>

<delta t="307.0-311.0">The screen transitions to a new title card "Make a Sealer - Mix 3 tsp or Cornstarch with Just a Little Water to Dissolve" at 306s, which remains on screen, then at 311s the title card disappears and a plate with cornstarch is visible.</delta>
```

| 馅料阶段 | 准备封口 |
|---|---|
| ![](assets/580/000302.jpg) | ![](assets/580/000393.jpg) |

```xml
<delta t="333.0-337.0">The title card "Place the Spring Roll Wrappers on a Plate and Begin Filling the Rolls" remains on screen, then at 336s it transitions to a visual of a box of lumpia wrappers.</delta>

<delta t="368.0-416.0">The hand continues to fold and roll the lumpia wrapper around the filling, completing the roll by 375s, and continues to adjust the roll on the plate, then at 416s the screen transitions to a title card.</delta>

<delta t="417.0-426.0">The title card "Here is Grammy Rolling her First Spring Roll...." remains on screen, then at 419s an older woman, presumably "Grammy," is shown beginning to roll a spring roll, and continues to roll it on the plate, with another person assisting her at 426s.</delta>

<delta t="427.0-437.0">The older woman continues to roll the spring roll, with the other person assisting her, and at 429s the spring roll is almost fully formed, then at 430s she continues to roll it, and at 434s she is still rolling it, then at 437s the screen transitions to a title card.</delta>
```

![](assets/580/000429.jpg)

```xml
<delta t="442.0-446.0">Grammy continues to speak to the camera, then at 446s the scene transitions to a tray of spring rolls.</delta>
```

`<anchor t="446.0-447.0">`

![](assets/580/000447.jpg)

`</anchor>`

```xml
<delta t="447.0-453.0">The tray of spring rolls remains on screen, then at 453s the scene transitions to a wok.</delta>
```

`<anchor t="453.0-454.0">`

![](assets/580/000453.jpg)

`</anchor>`

```xml
<delta t="454.0-455.0">The wok remains on screen.</delta>
```

### Step 91 当前帧

| 455s | 456s | 457s |
|---|---|---|
| ![](assets/580/000455.jpg) | ![](assets/580/000456.jpg) | ![](assets/580/000457.jpg) |

### Step 91 的真实模型输出

```xml
<state>The video has progressed through preparing the filling for spring rolls, including soaking shrimp, grating ginger, chopping green onions, and mixing broccoli slaw with various spices and ingredients. The next step shown was preparing a sealer (cornstarch and water mixture) and then beginning to wrap the spring rolls. The current frames show a wok, which is the next step in the cooking process. The question asks what the person does after adding the rolls into hot oil and cooking for 6-8 minutes until light golden brown. Based on the memory, the next step after cooking the rolls is to remove them and place them on a plate with paper towels and serve. This corresponds to option C.</state>
<answer>C</answer>
<delta t="454.0-458.0">The wok remains on screen, indicating the next step is to add the spring rolls to the hot oil.</delta>
```

这个 case 的重点：memory 里不是一张最后的锅图，而是“标题 -> 馅料 -> sealer -> wrapper -> rolling -> tray -> wok”的真实 delta 链。ASI 依赖的就是这条图文交错步骤链。

---

## Case 3: `1603_8`，REC，pole vault 计数

问题：

```text
How many times did they pole vault?
Provide your answer as a single number.
```

这题展示“累计事件”怎么存在 memory 里。下面是 step 18 答题前，模型已经写下的真实事件链。

### Step 18 的真实输入片段，Memory before

```xml
<delta t="1.0-6.0">A graphic appears at the bottom of the screen at 1s, indicating "POLE VAULT MEN FINAL", and remains on screen.</delta>

<delta t="11.0-15.0">The camera remains focused on the male pole vaulter, who is holding a pole and looking forward, then at 15s he starts running with the pole.</delta>
```

`<anchor t="15.0-16.0">`

![](assets/1603_8/000015.jpg)

`</anchor>`

```xml
<delta t="16.0-22.0">The pole vaulter continues running with the pole, at 19s he plants the pole and begins to vault, then at 20s he is in mid-air, clearing the bar.</delta>
```

| 19s | 22s | 23s |
|---|---|---|
| ![](assets/1603_8/000019.jpg) | ![](assets/1603_8/000022.jpg) | ![](assets/1603_8/000023.jpg) |

```xml
<delta t="23.0-27.0">The pole vaulter lands on the mat at 23s, then at 24s he raises his arms in celebration, and at 27s he is seen again, raising his arms in celebration.</delta>

<delta t="37.0-41.0">The camera cuts back to the pole vaulter, who is running with the pole, then at 39s he is in mid-air, having planted the pole, and at 41s he is clearing the bar.</delta>
```

| 37s | 41s |
|---|---|
| ![](assets/1603_8/000037.jpg) | ![](assets/1603_8/000041.jpg) |

```xml
<delta t="42.0-46.0">The pole vaulter continues to clear the bar, then at 43s he lands on the mat, and at 44s he raises his arms in celebration, then at 45s he is lying on the mat.</delta>

<delta t="57.0-61.0">Steve Hooker continues running with the pole, preparing for his vault, then at 60s he plants the pole and begins to vault.</delta>
```

| 60s | 63s |
|---|---|
| ![](assets/1603_8/000060.jpg) | ![](assets/1603_8/000063.jpg) |

```xml
<delta t="62.0-67.0">Steve Hooker is in mid-air, clearing the bar, then at 63s he lands on the mat, and at 67s he stands up, holding his pole.</delta>

<delta t="68.0-78.0">Steve Hooker is standing on the mat, holding his pole, and at 69s he is seen again, holding his pole and looking forward, then at 71s he is seen again, holding his pole and looking forward, and at 72s he is seen again, holding his pole and looking forward, then at 75s the camera cuts to a new pole vaulter, Evgeny Lukyanenko, who is running with his pole.</delta>

<delta t="79.0-83.0">Evgeny Lukyanenko continues running with his pole, then at 83s he plants the pole and begins to vault.</delta>
```

| 83s | 86s |
|---|---|
| ![](assets/1603_8/000083.jpg) | ![](assets/1603_8/000086.jpg) |

```xml
<delta t="84.0-86.0">Evgeny Lukyanenko is in mid-air, clearing the bar, then at 86s he lands on the mat.</delta>

<delta t="87.0-90.0">Evgeny Lukyanenko is lying on the mat, then at 88s Steve Hooker is seen looking down.</delta>
```

### Step 18 的真实模型输出

```xml
<state>The video shows a pole vaulting competition. The question asks to count the total number of pole vaults performed by different people. Based on the memory, Evgeny Lukyanenko vaulted twice (at 16s and 37s), and Steve Hooker vaulted twice (at 60s and 75s). The current frames show Steve Hooker looking down and a spectator. Therefore, the total number of pole vaults is 4.</state>
<answer>4</answer>
<delta t="87.0-91.0">Evgeny Lukyanenko is lying on the mat, then at 88s Steve Hooker is seen looking down, and at 91s a spectator is shown.</delta>
<anchor t="91.0-92.0"></anchor>
```

输出 anchor：

![](assets/1603_8/000091.jpg)

这个 case 的重点：真实输出的 state 明确在读 memory 做计数，`answer=4` 不是最后一帧推出来的。

---

## Case 4: `252`，EPM，答案在早期二楼 bathroom

问题：

```text
Where can I wash my hands on the second story of the house?
Options:
A. The bathroom sink
B. The table in the dining room
C. The shelf in the hallway
D. The shelf in the closet
```

这题的提问时间在 234s，但答案依赖 1-89s 的早期楼上 bathroom。当前帧已经是 living room。

### Step 46 的真实输入片段，Memory before

```xml
<delta t="1.0-37.0">The camera slowly pans to the left, revealing more of the empty closet with white shelves, then at 9s the door of the closet is visible, and the camera continues to pan left, revealing a hallway and another room in the background, which remains visible until the end of the current frames. The camera then slowly pans further to the right, revealing more of the hallway and a bathroom on the right side, which remains visible until the end of the current frames, and then the camera slowly pans further to the left, revealing more of the shower area, and at 37s, a bathtub becomes visible.</delta>
```

| 25s | 30s | 37s |
|---|---|---|
| ![](assets/252/000025.jpg) | ![](assets/252/000030.jpg) | ![](assets/252/000037.jpg) |

```xml
<delta t="38.0-89.0">The camera continues to pan left, revealing more of the bathtub and the room beyond it, then at 42s, a vanity and mirror become visible, and at 43s, a doorway leading to another room is visible, which remains visible until the end of the current frames, showing a bedroom with a bed and windows, and the camera slowly pans further to the right, revealing more of the bathroom vanity and mirror, then at 65s, the camera pans further into the bedroom, showing more of the bed and the room, and continues to pan right, revealing more of the room, and at 77s, the camera pans further right, revealing more of the hallway and another room, and continues to pan right, showing a hallway with a closet on the left and a doorway on the right, then at 87s, a dresser and mirror become visible on the left side of the frame.</delta>
```

| 42s：vanity/mirror | 89s：之后转到别处 |
|---|---|
| ![](assets/252/000042.jpg) | ![](assets/252/000089.jpg) |

到 230s 附近，当前画面已经变成客厅：

| 230s | 233s |
|---|---|
| ![](assets/252/000230.jpg) | ![](assets/252/000233.jpg) |

### Step 46 的真实模型输出

```xml
<state>The video has been panning through various rooms of a house, including a closet, hallway, bathroom, bedroom, and living room. The current frames show a living room with a dining area and a console table with a lamp. The question asks where to wash hands on the second story. Based on the memory, a bathroom was visible earlier in the video, which would have a sink. Therefore, option A is the correct answer.</state>
<answer>A</answer>
<delta t="172.0-234.0">The camera continues to pan down and to the right, revealing more of the staircase railing and the room beyond it, which appears to be a living room with furniture and a large window, and the camera remains relatively still, showing the living room from an elevated perspective, with a large painting on the left wall, and at 223s, a second large painting becomes visible on the left wall, and the camera continues to pan right, revealing more of the living room and a console table with a lamp on the right side of the frame.</delta>
```

这个 case 的重点：模型真实输出里明确说 “Based on the memory, a bathroom was visible earlier”。这就是 memory retrieval，不是当前帧答题。

---

## Case 5: `1143`，FPD，drawer 的实体-动作绑定

问题：

```text
What is this person doing with the drawer?
Options:
A. The person is testing the drawer's mechanisms for maintenance.
B. The person is opening the drawer to retrieve or place an item inside.
C. The person is cleaning out the drawer for cleanliness.
D. The person is inspecting the contents for safety.
```

这题展示的是短视频里的细粒度 object-action chain。

### Step 3 的真实输入片段，Memory before

```xml
<delta t="1.0-9.0">The operator holds a piece of fabric and a green object, then places the fabric on a white surface at 4s, and at 9s places the fabric into a drawer.</delta>
```

| 4s | 9s |
|---|---|
| ![](assets/1143/000004.jpg) | ![](assets/1143/000009.jpg) |

```xml
<delta t="10.0-11.0">The operator holds a green object and a blue object, then at 11s places the blue object on a white surface.</delta>
```

![](assets/1143/000011.jpg)

```xml
<delta t="12.0-15.0">The operator continues to hold the green object, then at 14s places it into a drawer.</delta>
```

![](assets/1143/000014.jpg)

### Step 3 当前帧

![](assets/1143/000015.jpg)

### Step 3 的真实模型输出

```xml
<state>Earlier, the operator placed a piece of fabric into a drawer at 9s, then a blue object on a white surface at 11s, and then a green object into a drawer at 14s. The current frame at 15s shows the operator opening a blue cabinet. The question asks what the person is doing with the drawer. Based on the previous actions of placing items into drawers, the most appropriate answer is that the person is opening the drawer to retrieve or place an item inside.</state>
<answer>B</answer>
<delta t="12.0-15.0">The operator continues to hold the green object, then at 14s places it into a drawer, and at 15s opens a blue cabinet.</delta>
<anchor t="15.0-16.0"></anchor>
```

输出里的 anchor：

![](assets/1143/000015.jpg)

这个 case 的重点：FPD 答对不是因为识别了一个 drawer，而是因为 memory 连续保存了 fabric/blue object/green object 与 drawer 的动作关系。

---

## Case 6: `1480_0`，CRR，只有前提，没有后续结果

问题：

```text
The man picked up several books from the ground. What does the man do with the books he picked up?
Decide whether the existing visual content ... provides enough information for answering the question.
Answer only with "Yes" or "No".
```

这题展示“图文交错 memory 也要支持拒答”。模型看到了捡书，但没看到拿起来之后怎么处理，所以答 `No`。

### Step 173 的真实输入片段，Memory before

```xml
<delta t="856.0-862.0">The woman in the grey suit and white gloves continues to interact with the woman in the brown dress, then at 858s she turns and walks away, leaving the woman in the brown dress alone, who continues to look into the bookstore window.</delta>
```

`<anchor t="862.0-863.0">`

![](assets/1480_0/000862.jpg)

`</anchor>`

```xml
<delta t="863.0-865.0">The woman in the brown dress continues to look into the bookstore window, then at 862s a man in a blue shirt and vest is seen picking up books from the floor in the bookstore.</delta>
```

当前帧：

| 865s | 869s |
|---|---|
| ![](assets/1480_0/000865.jpg) | ![](assets/1480_0/000869.jpg) |

### Step 173 的真实模型输出

```xml
<state>The video shows a man in a blue shirt and vest picking up books from the floor in a bookstore. The question asks what the man does with the books he picked up. The current frames show him continuing to pick up books, but there is no visual evidence of him doing anything with them after picking them up. Therefore, the question cannot be answered with the current information.</state>
<answer>No</answer>
<delta t="863.0-870.0">The man in a blue shirt and vest continues to pick up books from the floor in the bookstore.</delta>
```

这个 case 的重点：memory 里有“picking up books”，但没有“does with the books after picked up”。模型真实输出把这个证据缺口写出来了，所以 `No` 是合理的。

---

## 这份 demo 应该怎么看

每个 case 都按同一个结构读：

1. 先看 `Memory before`：这就是模型在该 step 真正能看到的历史，不是人工重新看视频。
2. 再看图片：这些是 `<anchor>` 或当前帧，和 delta 文本交错出现。
3. 最后看 `backend_result.text`：这是模型真实输出的 XML。

如果要展示我们的 idea，重点应该放在这些片段上：模型不是一次性读完整视频，而是在每个 step 把视觉 anchor 和语义 delta 编织成 memory，再用这个 memory 支撑后续答题。
