# Errata

## E1 (2026-11-24): Success criterion was too loose, so the reported success rate was inflated (90% → 65%)

### What was wrong

In the first version of `scripts/eval_pusht.py`, success was decided like this:

```python
is_success = bool(terminated or info.get("is_success", False) or max_reward >= 0.95)
```

I assumed the PushT `reward` was the raw coverage ratio. It isn't. In `gym_pusht/envs/pusht.py`:

```python
reward = np.clip(coverage / self.success_threshold, 0.0, 1.0)   # success_threshold = 0.95
terminated = is_success = coverage > self.success_threshold
```

So `reward` is coverage divided by 0.95 and clipped to `[0, 1]`. The extra clause `max_reward >= 0.95` really meant
**coverage ≥ ~90.25%**, which is looser than the official criterion (coverage > 95%). A real success always gives
`reward == 1.0` and ends the episode early.

### Impact on the v1 results (seeds 1000–1019)

| Category | Episodes | Count |
| :--- | :--- | :--- |
| Real success (reward 1.0, terminated early) | 1, 2, 3, 4, 6, 8, 9, 10, 11, 16, 17, 18, 19 | 13 |
| **Wrongly counted as success** (ran all 300 steps, coverage ~91–94%) | 5, 7, 13, 14, 20 | 5 |
| Failure | 12 (coverage ~90%), 15 (coverage ~53%) | 2 |

Reported: **90% (18/20)**. Correct by the official criterion: **65% (13/20)**.
The original report is kept at [`outputs/archive/eval_report_v1_threshold_bug.json`](../outputs/archive/eval_report_v1_threshold_bug.json).

### Second issue found during review: what "197 ms/step" actually measured

The timer wrapped every `policy.select_action()` call. With `horizon=16, n_action_steps=8`, DiffusionPolicy
runs a full denoising pass only when its action queue is empty (once every 8 steps). The other 7 calls just pop a
cached action (~0.2 ms). So 197 ms was an **amortized per-step average**. A single full DDPM-100 denoising pass takes
**~1.6 s** on the RTX 4070 Laptop GPU. This was not a wrong number, but the README described it poorly.

### Fix

- Success comes **only** from the environment's `info["is_success"]`. Max coverage is now logged directly from `info["coverage"]`.
- Inference timing is split into "full denoising pass" vs. "queue pop", plus the amortized per-step average.
- `torch.manual_seed(seed)` runs per episode, so diffusion sampling noise is reproducible.
- Added `--scheduler ddim --num-inference-steps N` to study the speed/quality trade-off.

### Lesson

Read the environment source to find what `reward` and `success` actually mean before setting a threshold.
Sanity-check results against officially reported numbers. Here, 90% was noticeably higher than the official figure,
and that gap should have been a red flag.
