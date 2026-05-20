# Analysis: Case 590: clothes before put down

Question: What happened before the person put down the clothes?  
Ground truth: `B. Tidied up the blanket.`

| Method | Final answer | Result | Step outputs |
|---|---|---|---|
| RL interleaved memory | `B` | Correct | [rl_steps.md](rl_steps.md) |
| Text-only memory | `A` | Wrong | [text_steps.md](text_steps.md) |

## Why RL Is Correct

RL keeps the earlier folding/tidying action in memory and, at the query step, answers the preceding event rather than the current put-down event.

## Why Text-Only Is Wrong

Text-only describes the current window as placing the folded garment down, then answers A, which repeats the queried event instead of identifying what happened before it.

## What To Show

Step 3 is the query step: RL answers B, while text-only answers A.
Open the two step-output files separately so the viewer sees each model trajectory on its own.
