# Analysis: Case 493: inner tube leak before polishing

Question: What does the person do before use sandpaper/metal to polish rubber near leak?  
Ground truth: `B. look for leaks`

| Method | Final answer | Result | Step outputs |
|---|---|---|---|
| RL interleaved memory | `B` | Correct | [rl_steps.md](rl_steps.md) |
| Text-only memory | `C` | Wrong | [text_steps.md](text_steps.md) |

## Why RL Is Correct

RL keeps the repair sequence as inspection/manipulation of the removed inner tube before later treatment, and answers the preceding action: looking for leaks.

## Why Text-Only Is Wrong

Text-only sees the glue in the final window and jumps to option C, confusing a later preparation/apply-glue step with the action before polishing near the leak.

## What To Show

Step 10 is the query step: RL answers B; text-only answers C.
Open the two step-output files separately so the viewer sees each model trajectory on its own.
