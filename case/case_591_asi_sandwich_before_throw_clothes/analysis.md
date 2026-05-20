# Analysis: Case 591: object eaten before throwing clothes

Question: Which object did the person eat before they threw the clothes?  
Ground truth: `A. The sandwich.`

| Method | Final answer | Result | Step outputs |
|---|---|---|---|
| RL interleaved memory | `A` | Correct | [rl_steps.md](rl_steps.md) |
| Text-only memory | `None of the above options are correct. The person was eating chips.` | Wrong | [text_steps.md](text_steps.md) |

## Why RL Is Correct

RL uses the accumulated memory of the person eating before the later clothing-removal/throwing action and maps that earlier eating event to option A.

## Why Text-Only Is Wrong

Text-only focuses on the visible bag at the final step and outputs a non-option explanation, "None of the above... chips", so it fails the multiple-choice protocol and misses the GT option.

## What To Show

Step 9 is the query step: RL answers A; text-only gives a non-option answer.
Open the two step-output files separately so the viewer sees each model trajectory on its own.
