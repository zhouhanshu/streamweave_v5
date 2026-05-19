# Original OVO Forward Prompts

This file records the original REC / SSR / CRR task prompts before the
target-aware REC and relaxed SSR / CRR prompt edits.

## Original Query Timing

Before the target-aware REC edit, forward query timing in
`evaluation/ovo_adapter.py` was:

```python
if task in {"REC", "SSR"}:
    query_timestamp = target_timestamp
else:
    query_timestamp = float(anno.get("ask_time", 0.0)) if task == "CRR" else 0.0
```

## REC_PROMPT_TEMPLATE

```python
REC_PROMPT_TEMPLATE = """\
You're watching a video in which people may perform a certain type of action repetitively.
The person performing this kind of action is referred to as "they" in the following statement.
Your task is to count how many times different people in the video have performed this kind of action in total.
One complete motion counts as one.
Now, answer the following question: {question}
Provide your answer as a single number (e.g., 0, 1, 2, 3...) indicating the total count.
Do not include any additional text or explanation in your response.
"""
```

## SSR_PROMPT_TEMPLATE

```python
SSR_PROMPT_TEMPLATE = """\
You're watching a tutorial video which contains a sequence of steps.
The following is one step from the whole procedure:
{step}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""
```

## CRR_PROMPT_TEMPLATE

```python
CRR_PROMPT_TEMPLATE = """\
You're responsible for answering questions based on the video content.
The following question is relevant to the latest frames, i.e. the end of the video.
{question}
Decide whether the existing visual content, especially the latest frames near the end of the video, provides enough information for answering the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""
```
