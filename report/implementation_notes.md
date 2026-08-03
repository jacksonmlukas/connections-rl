# GRPO implementation notes: what we actually ran

Bookkeeping for the reviewer question "is the collapse an artifact of *your* GRPO
implementation rather than a property of the reward?" Dr. GRPO
([2503.20783](https://huggingface.co/papers/2503.20783)) identifies optimization
biases inside GRPO itself, so the specific normalization choices matter and
should be on record rather than reconstructed after the fact.

## What we set explicitly

From `configs/train/grpo*.yaml` via `src/connections_rl/train/grpo.py`:

| Setting | 1.5B (v2) | 7B |
|---|---|---|
| `num_generations` (K) | 8 | 8 |
| `temperature` | 0.9 | 0.9 |
| `beta` (KL to SFT reference) | 0.001 | 0.001 |
| `learning_rate` | 5e-6 | 5e-6 |
| `per_device_train_batch_size` | 2 | 2 |
| `gradient_accumulation_steps` | 8 | 8 |
| `num_train_epochs` | 2 | 1 |
| `max_completion_length` | 512 | 512 |
| LoRA | r=16, α=32, all-linear | same, over 4-bit base (QLoRA) |

## What we did NOT set (and therefore inherited from TRL defaults)

Neither the YAML configs nor `grpo.py` ever pass `scale_rewards`, `loss_type`,
`num_iterations`, or the clipping epsilons. Those took the installed TRL
version's defaults. In current TRL `main`, those defaults are:

- **`scale_rewards="group"`** — advantages are scaled by the standard deviation
  *within each generation group*. This is precisely the choice Dr. GRPO argues
  against: dividing by group std introduces a question-level difficulty bias
  (easy prompts, where the group agrees, get their advantages inflated). The
  Dr. GRPO recommendation is `scale_rewards=False`/`"none"`.
- **`loss_type="dapo"`** — token-level losses normalized by the number of active
  tokens in the global accumulated batch (DAPO), which removes the length bias
  present in the older `"grpo"` aggregation. This one is the *unbiased* choice
  of the available options, so length bias is not a live concern here.

**Action item before submission:** pin the exact TRL version used for each run
and confirm its defaults, rather than assuming current-main behavior. The
version is recoverable from each Kaggle run's environment; `pip freeze | grep
trl` inside a rerun of the training notebook is sufficient, and the value should
be recorded here per run.

## Advantage collapse (computable from existing logs, zero cost)

TRL logs `frac_reward_zero_std`: the fraction of generation groups whose K=8
completions all received the *same* reward. Under group-relative advantage, a
zero-variance group yields zero advantage for every completion in it, so those
groups contribute no gradient signal at all.

From the published 1.5B v2 training log:

| Step | `frac_reward_zero_std` | reward mean | reward std | entropy |
|---|---|---|---|---|
| 5 | 0.3 | −0.002 | 0.152 | 0.339 |
| 120 | 0.0 | 0.203 | 0.287 | 0.253 |
| 240 | 0.4 | 1.289 | 0.427 | 0.040 |
| 250 | 0.7 | 1.563 | 0.150 | 0.007 |
| 270 | **1.0** | **1.600** | **0.000** | 0.0014 |
| 403 (final) | **1.0** | **1.600** | **0.000** | 0.0003 |

By roughly step 270 the run reaches total advantage collapse: every group is
saturated at the maximum reward (1.6), reward variance is exactly zero, and the
policy gradient is therefore zero for the remainder of training. Entropy is
~3e-4 at the end. The last ~130 steps of the 1.5B run performed no meaningful
optimization.

Two implications worth stating in the paper:

1. The collapse is not a case of RL "continuing to push" the policy into a
   degenerate region for the whole run — it pushes hard early, saturates the
   training reward, and then stops receiving signal. This is consistent with the
   checkpoint decomposition, where nothing changes after step 150.
2. Group-std scaling (`scale_rewards="group"`) is *most* unstable exactly as
   groups approach zero variance, which is where this run spent its final third.
   Whether the collapse is causally driven by that normalization choice is a
   testable question: rerunning 1.5B with `scale_rewards="none"` is a cheap
   single-run ablation and is the natural companion to the reward-component
   ablation.

Equivalent numbers for the 7B runs and for seeds 1-2 live in the W&B runs
(`connections-rl-grpo-*`) and should be tabulated the same way before writing.
