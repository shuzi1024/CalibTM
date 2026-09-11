# Tests

The lightweight unit tests can be run with:

```bash
python -m pytest experiments/minimal_calibrator_v1/tests \
                 experiments/tiny_kan_calibrator_v1/tests
```

Some protocol tests intentionally require registered canonical arrays and
therefore remain skipped or unavailable until the corresponding datasets are
downloaded. GPU smoke tests are opt-in and require one visible CUDA device.
