# CalibTM / ARI-LLM research repository

This is the clean, code-first research repository for traffic-matrix completion.
It contains the original ARI-LLM implementation, the validated lightweight
`value_only` calibrator, selected experiment runners, and the GapCalib
KAN/Attention/Mamba-lite bake-off protocol.

## Contents

- `Imputation/`: original ARI-LLM training and evaluation code.
- `experiments/`: selected experiment runners, configurations, and compact reports.
- `plan/`: current GapCalib module protocol and novelty plan.
- `paper/` and `docs/`: paper materials and evidence handoffs.
- `setup/`: environment and asset preparation helpers.

Large checkpoints, model caches, raw archives, generated predictions, and datasets
are intentionally omitted.

## Environment and assets

Python 3.10 is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup/download_assets.py --datasets abilene geant
python setup/check_setup.py
```

ARI-LLM baselines additionally require GPT-2 weights:

```bash
python setup/download_assets.py --gpt2
```

Datasets belong under `datasets/net_traffic/`; GPT-2 weights belong under `GPT2/`.
Data and model assets are not committed because they are large and may have their
own redistribution terms. WS-DREAM preprocessing is not yet fully automated from
raw data; see the selected experiment reports before attempting that benchmark.

## Current research direction

The stable incumbent is CalibTM v3 (`value_only`). The next experiment is specified
in `plan/CALIBTM_V4_MODULE_BAKEOFF_PROTOCOL_2026-09-11_CN.md`: first validate the
GapCalib geometry host, then compare one MLP/KAN/Attention/Mamba-lite generator
under the same outer constraints. Do not overwrite the v3 implementation.

## Reproducibility note

This is a source snapshot, not a trained-model bundle. Text reports retain the
historical findings, but rerunning experiments requires data/model downloads on
the target GPU server. Some historical reports contain old machine paths; these
are provenance records, not portable launch commands. Prefer current relative
paths and `python -m ...` entry points.

## Moving existing checkpoints

If you want evaluation without retraining, copy the required checkpoint and its
matching dataset from the old machine separately, for example:

```bash
rsync -avP OLD_SERVER:/path/to/checkpoints/ ./checkpoints/
rsync -avP OLD_SERVER:/path/to/datasets/net_traffic/ ./datasets/net_traffic/
```

The checkpoint must match the experiment configuration and preprocessing
registry. A newly designed GapCalib/KAN/Attention/Mamba module has no compatible
old checkpoint and must be trained once; the existing v3 `value_only` model can
be evaluated directly after its checkpoint is copied.
