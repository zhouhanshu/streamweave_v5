# Experiment 3 Proposal: GRPPO

This proposal records the intended Experiment 3 algorithm and integration plan before implementation.
The goal is to keep previous experiments reproducible: do not change existing YAML defaults or old launch
scripts for exp1/exp2. GRPPO should be enabled only by the experiment-3 shell script and new code paths.

## Name

User-facing algorithm name: `GRPPO`.

Internal advantage estimator name:

```text
streamweave_stepwise_grppo
```

The `streamweave_stepwise_` prefix is intentional because the trainer passes the full `DataProto` and
algorithm config only to estimators with this prefix.

## High-Level Objective

GRPPO is a step-level RL objective for StreamWeave. It should not use trajectory reward and should not
broadcast final success to all turns.

Each rollout step has two scalar training signals:

```text
step_reward_t
answer_credit_t
```

`step_reward_t` is the GRPPO process signal used for the step component. It combines the judge's
four-dimensional process score with the parser/format score so malformed XML can be penalized before
advantage normalization.

They are normalized separately within the same sample group and same turn index:

```text
step_advantage_t = center(step_reward_t within (group_idx, turn_idx))
answer_advantage_t = center(answer_credit_t within (group_idx, turn_idx))
advantage_t = step_weight * step_advantage_t + answer_weight * answer_advantage_t
```

The final weighted advantage is then broadcast to valid response tokens for that step, as existing
stepwise estimators already do.

## Judge Design

GRPPO uses a new judge prompt version and should not alter the existing step judge behavior used by
previous experiments.

Recommended prompt version:

```text
streamweave_grppo_judge_v1
```

The judge has two prompt templates.

### Process Prompt

Used when the current step does not trigger answer reward.

The judge returns four scalar values from 0.0 to 1.0:

```text
delta_groundedness
anchor_keyframe
semantic_alignment
state_groundedness
```

This prompt must not mention or request `answer_reward`.

The dimensions mean:

- `delta_groundedness`: whether delta text concisely describes frame-to-frame change, includes key
  observable information, and avoids hallucination or pollution from the answer/question/QA history.
- `anchor_keyframe`: whether the selected anchor is a real useful keyframe, whether the step selects
  at most one keyframe, and whether long no-anchor/delta spans are penalized.
- `semantic_alignment`: whether the overall image/text memory aligns with the video frames, preserves
  correct temporal order, and remains semantically coherent.
- `state_groundedness`: whether state is hallucination-free, avoids QA-history pollution, and whether
  any answer reasoning/output is aligned with grounded state reasoning.

The judge process score is:

```text
judge_step_reward_t = 2.0 * mean(all process checklist values)
```

The GRPPO step reward also includes format reward:

```text
step_reward_t =
  grppo_process_weight * judge_step_reward_t
  + grppo_format_weight * format_score_t
  + grppo_note_frequency_weight * note_frequency_score_t
```

### Answer-Aware Prompt

Default/legacy behavior is kept only for reproducibility of earlier GRPPO runs: a step can trigger
answer reward because it has a user query, an answer target, or a non-empty model answer.

Exp6 uses timeline answer supervision. The env derives one supervision state before judging each step:

```text
none    : no query has appeared yet, so no answer reward is requested
silence : a query is active, but this step is not an answer target
answer  : this step has an answer target with a target answer
```

When supervision is `silence` or `answer`, the step uses the answer-aware LLM judge prompt. The trainer
does not overwrite the answer reward with a rule-only forced-answer postprocess in exp6.

The prompt receives the rendered answer supervision label. At a single training step, there is at most
one query and at most one answer target, but a query and its answer target may legitimately fall into
the same step.

The judge returns the same five scalar values:

```text
delta_groundedness
anchor_keyframe
semantic_alignment
state_groundedness
answer_reward
```

Compared with the process prompt, the answer-aware prompt also evaluates:

- whether delta/state hallucination was caused by the question text or answer options;
- whether the model copied query assumptions into memory/state as visual facts;
- whether the answer follows the state reasoning;
- whether the answer is correct under the current step annotation.

The answer-aware prompt receives only a rendered, case-specific answer label section. It should not
receive raw annotation JSON or internal step flags. For example, a silence case is rendered as:

```text
Event type: answer supervision checkpoint.
Current question: ...
Annotation time: ...s
Requirement: the model should not answer now; <answer> should be empty.
```

An answer-target case is rendered as:

```text
Event type: answer checkpoint.
Current question: ...
Options:
A. ...
B. ...
C. ...
Annotation time: ...s
Requirement: the model must answer now. Reference answer: ...
```

`answer_reward` is judged primarily from this label section, then binarized. The answer label's
current requirement and reference answer are authoritative. The model state is checked only for
grounding and consistency with the answer; it must not override or reinterpret the reference answer.
It should be binary:

```text
1.0 = the answer behavior satisfies the label
0.0 = wrong answer, missed required answer, answered when silence was required, contradicted state,
      or used ungrounded state reasoning to support the answer
```

Rule answer correctness is passed only as a reference/metric. It must not directly overwrite
normal answer-correctness decisions in timeline mode.

`answer_reward` is not tied to the number of queries. In timeline mode it is emitted whenever the
derived supervision is `silence` or `answer`. This covers:

- query active, no answer target: judges whether the model correctly stays silent;
- answer target only: checks whether the model answers when the annotation says the current step
  needs an answer;
- query, answer target, and/or model answer in the same step: rewards/penalizes the immediate answer
  decision without dropping the query event;
- before the first query: no answer reward event.

If a cohort has no current query and no answer target, but at least one rollout emits a non-empty
answer, the trainer performs a cheap postprocess before answer-credit computation:

```text
answered rows: grppo_answer_reward = 0
silent rows:   grppo_answer_reward = 1
```

This catches forced-answer behavior without trusting an extra LLM answer score.

This postprocess is a legacy compatibility path. Exp6 disables it with
`grppo_forced_answer_postprocess_enable=false`; answer reward comes from the answer-aware LLM judge.
For silence supervision, the final scalar is `grppo_silence_reward_value * binarize(answer_reward)`.

## Query Annotation Format

RL now accepts one canonical annotation format only. Legacy top-level
`question/query_timestamp/ground_truth`, `queries`, `target_answers`, and draft `question + response`
formats are not accepted by the RL dataset/env path.

Each row must contain a non-empty `query_events` list. Each query explicitly declares one answer type:

```text
answer_type = "mcq"  |  "text"
```

For MCQ, `options` is required and `answer_events[].gt` is the option letter. For natural-language
answers, omit `options` and set `answer_type` to `"text"`.

Canonical one-question-one-answer MCQ row:

```json
{
  "query_events": [
    {
      "qid": "q0",
      "time": 12.0,
      "content": "What object is on the table?\nOptions:\nA. cup\nB. book",
      "answer_type": "mcq",
      "options": ["cup", "book"],
      "answer_policy": "answer_when_asked",
      "answer_events": [
        {
          "time": 18.0,
          "gt": "A",
          "answer": "cup",
          "content": "A. cup"
        }
      ]
    }
  ]
}
```

Canonical natural-language row:

```json
{
  "query_events": [
    {
      "qid": "q0",
      "time": 12.0,
      "content": "What is the person doing?",
      "answer_type": "text",
      "answer_policy": "answer_when_asked",
      "answer_events": [
        {
          "time": 18.0,
          "answer": "The person is cutting vegetables.",
          "content": "The person is cutting vegetables."
        }
      ]
    }
  ]
}
```

At any step, the env should select at most one query annotation and at most one answer-target
annotation. Query and answer target are separate fields and may coexist in the same step. If a row
contains two queries or two answer targets in the same step, the safest behavior is to raise or mark
the rollout invalid rather than silently merge labels.

### One Question, Multiple Target Answers

Some samples ask the model to update an answer as evidence changes over time. The canonical format
should keep target answers nested under the query instead of using two unrelated top-level arrays.
This avoids ambiguous pairing when a row later contains multiple questions, answer-only updates, or
query-only silence supervision.

Recommended annotation:

```json
{
  "query_events": [
    {
      "qid": "q0",
      "time": 127.0,
      "content": "What ingredient is being added...? Update your answer if it becomes different...",
      "answer_type": "mcq",
      "options": ["bacon", "tomato", "lettuce"],
      "answer_policy": "update_when_changed",
      "answer_events": [
        {
          "time": 180.0,
          "gt": "C",
          "answer": "bacon",
          "content": "C. bacon"
        },
        {
          "time": 185.0,
          "gt": "B",
          "answer": "tomato",
          "content": "B. tomato"
        },
        {
          "time": 192.0,
          "gt": "E",
          "answer": "lettuce",
          "content": "E. lettuce"
        }
      ]
    }
  ]
}
```

Rules:

- `time` should be numeric, not a string.
- `qid` is required in the canonical representation even if the current dataset has one query.
- Use `answer_events`, not `target_answers` or `response`.
- `content` is the exact question text shown to the model and may include MCQ options.
- `answer_type` is required. Do not infer MCQ from the presence of options in downstream data.
- For MCQ, `options` stores option text without letters, `gt` stores the option letter, `answer`
  stores the plain option text, and `content` stores the display answer.
- For text QA, `answer` stores the canonical natural-language answer. `content` may duplicate it.
- Every item in `answer_events` becomes its own answer reward event. In the example above, there are
  three answer reward events for the same query.
- For rollout/reward code, the nested format can be expanded into a time-ordered event stream:

```json
[
  {"type": "query", "qid": "q0", "time": 127.0, "content": "..."},
  {"type": "answer_target", "qid": "q0", "time": 180.0, "gt": "C", "content": "C. bacon"},
  {"type": "answer_target", "qid": "q0", "time": 185.0, "gt": "B", "content": "B. tomato"},
  {"type": "answer_target", "qid": "q0", "time": 192.0, "gt": "E", "content": "E. lettuce"}
]
```

## Answer Credit Assignment

After a full trajectory is rolled out, compute answer credit by walking backward over the trajectory.

Let `answer_reward_t` be the answer-aware judge score for step `t`, the legacy forced-answer
postprocess score for older runs that explicitly enable it, or `0.0` if there is no reward event
at that step.

```text
answer_credit_t = answer_reward_t + beta * answer_credit_{t+1}
```

This handles multiple answer/query reward events automatically. Example:

```text
answer_reward = [0.0, 0.6, 0.0, 0.9, 0.0, 0.4]
```

Then `answer_credit_t` contains the discounted sum of all future reward events from `t` onward.

There is no fixed finite window. The effective window is controlled by `beta`.

Recommended configurable overrides:

```text
+algorithm.grppo_answer_decay=0.7
+algorithm.grppo_step_weight=1.0
+algorithm.grppo_answer_weight=0.5
+algorithm.grppo_norm_by_std=false
+algorithm.grppo_min_std=0.03
```

These are the initial experiment defaults. They must be passed from the experiment-3 shell script,
not hard-coded into existing YAML defaults.

## Advantage Normalization

For every unique step row, first compute answer credit:

```text
answer_credit_t = answer_reward_t + beta * answer_credit_{t+1}
```

Then group rows by:

```text
(group_idx, turn_idx)
```

Normalize step and answer signals separately inside that cohort:

```text
step_advantage_t = step_reward_t - mean(step_reward within cohort)
answer_advantage_t = answer_credit_t - mean(answer_credit within cohort)
advantage_t = step_weight * step_advantage_t + answer_weight * answer_advantage_t
```

Do not compare different turns with each other. The weights combine advantages, not raw rewards.

If a signal's cohort has insufficient variance:

```text
std(step_reward over rollout.n) <= grppo_min_std
or
std(answer_credit over rollout.n) <= grppo_min_std
```

then only that signal's component advantage is zeroed for that cohort. A row should still train if
the other signal has enough variance.

The first implementation should support true step-level DAPO filtering before old-log-prob computation:

```text
keep rows whose (group_idx, turn_idx) cohort has enough variance in either step_reward or answer_credit
drop rows only when both component cohorts are low-variance
```

This differs from old trajectory DAPO: the unit is a step cohort, not an entire sample group.

The feature must be configurable from the experiment-3 shell script, for example:

```text
+algorithm.grppo_filter_groups.enable=true
+algorithm.grppo_filter_groups.min_std=0.07
```

When disabled, GRPPO should keep all rows and simply compute advantages for every step cohort.

## Integration Plan

Keep old behavior untouched. Implement GRPPO only through new names and new fields.

### Judge

Reuse existing backend/cache/retry code in `RL/streamweave_rl/judge.py`.

Add a prompt-version branch:

```text
JudgeConfig.prompt_version == streamweave_grppo_judge_v1
```

When this version is active, build either the four-score process prompt or the five-score answer-aware prompt.

The parser should extract GRPPO scores into flat numeric fields. Process prompts may omit
`answer_reward`; in that case it is treated as `0.0`. Avoid putting dict-valued judge outputs into
training/validation metric reducers.

### Environment

Reuse current rollout state from `RL/streamweave_rl/env.py`:

- `raw_action`
- `quality`
- `current_memory_before`
- `qa_history_before`
- `current_frames`
- current query annotation
- current answer target annotation
- current answer text

Add scalar info fields for GRPPO:

```text
grppo_delta_groundedness
grppo_anchor_keyframe
grppo_semantic_alignment
grppo_state_groundedness
grppo_judge_step_reward
grppo_format_score
grppo_step_reward
grppo_answer_reward
grppo_answer_reward_raw
grppo_answer_correctness
grppo_answer_event
grppo_answer_supervision
grppo_answer_reward_scale
grppo_has_query
grppo_has_answer_target
grppo_has_answer
grppo_query_count
grppo_answer_target_count
grppo_forced_answer_postprocess
grppo_prompt_kind
grppo_target_trajectory_score
grppo_target_answer_reward
grppo_target_format_reward
```

Existing `trajectory_score`, `success_score`, and `turn_reward` may remain for old code paths, but
the GRPPO estimator must not read them.

### Agent Loop

Add the GRPPO scalar fields to `reward_extra_info` in `agent_loop_stepwise.py` so they become
`non_tensor_batch` columns.

Do not remove or change old reward fields.

### Advantage Estimator

Add a new function in `advantage.py`:

```text
@register_adv_est("streamweave_stepwise_grppo")
def compute_streamweave_stepwise_grppo(...)
```

Implementation outline:

1. Use `_unique_turn_rows(data)` to deduplicate turn rows.
2. Read `grppo_step_reward` and `grppo_answer_reward`.
3. Legacy-only: if an older run enables forced-answer postprocessing, apply it before component
   computation. Timeline supervision does not use this trainer overwrite.
4. For each `(group_idx, traj_idx)` trajectory, sort by `turn_idx`.
5. Compute backward answer credit with `grppo_answer_decay` on the full rollout batch.
6. For each `(group_idx, turn_idx)` cohort, compute the centered `grppo_step_reward` advantage.
7. For the same cohort, compute the centered `grppo_answer_credit` advantage.
8. Zero a component advantage when that component's cohort variance is below `grppo_min_std`.
9. Compute the final policy advantage as
   `grppo_step_weight * step_advantage + grppo_answer_weight * answer_advantage`.
10. Broadcast the final scalar advantage to response tokens with `response_mask`.

For step-level DAPO, compute both component advantages before old-log-prob, build a row keep mask by
component cohort validity, and drop a step row only when both component cohorts are low-variance.
If rows are filtered, the estimator must reuse precomputed `grppo_answer_credit` and advantages from
the full batch instead of recomputing credit on the filtered trajectory.

### Launch Script

Do not edit YAML. Use the experiment-3 shell script to pass all overrides:

```text
algorithm.adv_estimator=streamweave_stepwise_grppo
+algorithm.grppo_answer_decay=0.7
+algorithm.grppo_step_weight=1.0
+algorithm.grppo_answer_weight=0.5
+algorithm.grppo_norm_by_std=false
+algorithm.grppo_min_std=0.03
+algorithm.grppo_filter_groups.enable=true
+algorithm.grppo_filter_groups.min_std=0.03
+data.streamweave.reward.grppo_process_weight=0.7
+data.streamweave.reward.grppo_format_weight=0.15
+data.streamweave.reward.grppo_note_frequency_weight=0.15
+data.streamweave.reward.grppo_answer_event_mode=timeline
+data.streamweave.reward.grppo_silence_reward_value=0.1
+data.streamweave.reward.judge.prompt_version=streamweave_grppo_judge_v1
```

If needed, create a new `train_exp3_grppo.sh` instead of changing older scripts.

## Current Decisions

1. Timeline `silence` and `answer` steps use the answer-aware prompt.
2. Process-only steps use the four-score process prompt and do not mention answer reward.
3. `grppo_answer_reward` is derived from the LLM judge `answer_reward` for both answer and silence
   labels. Rule correctness is only an auxiliary metric.
4. In timeline mode, silence labels use `grppo_silence_reward_value * binarize(answer_reward)`.
5. Forced answers in no-query/no-target cohorts are a legacy trainer postprocess path and are disabled
   for exp6.
6. `grppo_step_reward` is the unnormalized weighted sum of `grppo_judge_step_reward`,
   `grppo_format_score`, and `grppo_note_frequency_score`; `grppo_judge_step_reward` is a
   process checklist score on a 0-2 scale.
