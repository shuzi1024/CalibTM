# Shared-weight iterative SPIN + Direct

`one_step` and `two_step` have identical parameters and initialization per seed.
The two-step arm feeds its own clamped first prediction to the same context
encoder and decoder. An immutable true-observation mask distinguishes observed
values from estimates. Direct always reads original true observations only.
The first pass is a latent estimate without a separate supervised loss.

This is inspired by ARI's progressive refinement, at one fixed resolution.
It is not a reproduction of ARI-LLM, and adds no LLM, Sync, gate or loss search.
The original SPIN + Direct experiment is preserved as a historical reference.

## Checks

From a repository environment containing PyTorch:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -m experiments.spin_refine_v1.checks
```

## Training

The runner uses the existing verified CalibTM data bundle and registry. Those
historical data assets remain necessary; this module does not by itself solve
the project's public reproduction gap. Both arms train from scratch.

After a valid GPU allocation has been recorded, run the fit-only smoke before
formal training, using the same explicit authorization deadline:

```bash
python -m experiments.spin_refine_v1.runner --dataset geant --variant two_step --seed 41001 --deadline-unix DEADLINE --smoke
python -m experiments.spin_refine_v1.runner --dataset geant --variant two_step --seed 41001 --deadline-unix DEADLINE
```

Use `--resume` only when a committed `last.pt` exists. The runner restores
model, optimizer and RNG together. A complete epoch is the commit boundary.
The fixed schedule is 120 epochs at 0.001, followed by 40 at 0.0001. Training
updates are matched; compute is not. The primary score remains development
NMAE, averaged equally over the three original conditions.

See `analysis/spin_refine_20260917/PLAN_CN.md` for the bounded comparison and
stop rule. Code availability does not imply the experiment has been run; use
the execution status and completed result files to determine that.
