# ARI-LLM: Autoregressive Imputation with LLM Backbone

## Project Context

This is a research project on **network traffic matrix (TM) imputation** using autoregressive LLM-based methods. The core paper direction is:

> **Reliability-Guided ACIL-ARI**: Adaptive Process Initialization and Reliability-Gated Residual Refinement for Traffic Matrix Imputation

The central claim: instead of unconditional reconstruction, we decompose TM imputation into:
1. **Adaptive prior construction** (ACIL — Anchor-Conditioned Interpolation Layer)
2. **Prior reliability estimation** (Uncertainty Head)
3. **Selective residual refinement** (Reliability-Gated Residual)

## Codebase Structure

- `Imputation/models/ARI_LLM.py` — Main model (~1870 lines). Class `Model` with variant-based conditional architecture.
- `Imputation/models/geoanchor_v2_modules.py` — ACIL prior: `ObservationGeometryExtractorV2` (16 geometry features) + `AnchorConditionedInterpolationLayer` (learned delta_r + offset corrections).
- `Imputation/models/salt_modules.py` — SALT: `LatentTrafficCoupler` (cross-attention), `StageAwareUncertaintyGate` (heuristic gate).
- `Imputation/models/care_modules.py` — CARE: `GlobalCouplingExpert`.
- `Imputation/models/geoattn_modules.py` — GeoAttn: geometry-biased attention.
- `Imputation/exp/exp_imputation.py` — Training/test loop with progressive stage-based imputation.
- `Imputation/run.py` — Entry point with ~200 argparse flags.
- `experiments/` — All experiment scripts, results CSVs, analysis markdown.

## Key Datasets
- **GEANT** (European research network): 23×23 OD flows, seq_len=50
- **Abilene** (US backbone network): 12×12 OD flows, seq_len=50
- Mask types: `random`, `internal_block`, `burst`, `mixed_structured`
- Default mask rate: 0.05 (5% observed)

## GPU Environment
- 4× NVIDIA H800 80GB
- Experiments managed via tmux sessions
- Use `--gpus` flag to specify GPU indices

## Experiment Conventions
- Always use 3 seeds (2021, 2022, 2023) minimum
- Report NMAE, NRMSE, KL divergence
- CSV results go under `experiments/<experiment_name>/`
- Prior checkpoints at `experiments/acil_backbone/checkpoints_cross/priors/`

## Current Paper Status
- ACIL prior: implemented and validated (24/24 cells improved, GEANT delta -0.44%, strongest -3.23%)
- ACIL narrows backbone gap: GEANT internal_block gap shrinks from 0.004615 to 0.000254
- **TODO**: Uncertainty Head, Reliability-Gated Residual, NLL/calibration loss, method ablation experiments
