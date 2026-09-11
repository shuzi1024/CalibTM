# Tiny-KAN one-shot development collision

This package tests exactly one cubic B-spline KAN replacement for the current
`value_only` MLP.  It is development evidence, not independent confirmation.
It exposes no sealed-test or legacy-test-cache path.

## Local data-free checks

```bash
PYTHONPATH=. python -m pytest -q experiments/tiny_kan_calibrator_v1/tests
PYTHONPATH=. python -m experiments.tiny_kan_calibrator_v1.queue \
  --output-root /tmp/tiny-kan-preview --max-workers 1
```

The preview command is read-only unless `--launch` is present.

## CUDA smoke

Run the synthetic, data-free smoke first.  It performs one BF16 optimizer
update per registered arm and records only arithmetic/feasibility/memory
telemetry; it never opens A/G or any result artifact.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
python -B -m experiments.tiny_kan_calibrator_v1.smoke \
  --output /tmp/tiny-kan-cuda-smoke.json
```

## Full grid

After the smoke fixes a safe concurrency:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
python -B -m experiments.tiny_kan_calibrator_v1.queue \
  --output-root experiments/tiny_kan_calibrator_v1/results/development_r1 \
  --max-workers 6 --launch
```

All 18 jobs must finish before adjudication:

```bash
python -B -m experiments.tiny_kan_calibrator_v1.adjudicate \
  --results-root experiments/tiny_kan_calibrator_v1/results/development_r1 \
  --output experiments/tiny_kan_calibrator_v1/results/development_r1/adjudication.json
```

If the verdict is `KILL_TINY_KAN_KEEP_VALUE_ONLY`, do not try another KAN
variant.  If accuracy proceeds, run the pre-registered zero-spline attribution
and matched latency/memory checks before considering any fresh confirmation.
