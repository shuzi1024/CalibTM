# ARI-LLM Imputation

This folder contains the traffic matrix imputation implementation.

Use the project root README for environment and asset preparation. The key points are:

- Install `requirements.txt`.
- Prepare `GPT2/` and real CSV datasets before running.
- `datasets/net_traffic/*/*.csv` files that contain `version https://git-lfs.github.com/spec/v1` are Git LFS pointers, not usable datasets.

## Launchers

From the project root:

```bash
python Imputation/run_abilene.py
python Imputation/run_geant.py
python Imputation/run_wsdream.py
```

or:

```bash
PYTHON=.venv/bin/python bash Imputation/scripts/abilene.sh
PYTHON=.venv/bin/python bash Imputation/scripts/geant.sh
PYTHON=.venv/bin/python bash Imputation/scripts/wsdream.sh
```

`run.py` dispatches by `--data`:

- `net_traffic_abilene` -> Abilene wrapper
- `net_traffic_geant` -> GEANT wrapper
- `net_traffic_trans` -> WS-DREAM wrapper

The shared experiment loop implements the paper protocol: SMAPE training loss, uniform coarse-to-fine interpolation stages, observed-value preservation during progressive validation/testing, and metrics on missing entries only.
