# Third-party source attribution

This research snapshot reuses and adapts SPIN and Torch Spatiotemporal (TSL).
Their MIT notices apply to the corresponding original/derived code; the repository's root license does not replace them.

- SPIN: Graph Machine Learning Group, pinned commit `7349ba31da7306e7e96c13668a3f1f0a4df90902`, [upstream](https://github.com/Graph-Machine-Learning-Group/spin), [MIT license](experiments/spin_comparison_v1/upstream/spin-7349ba31da7306e7e96c13668a3f1f0a4df90902/LICENSE).
- TSL: `v0.1.1`, [upstream](https://github.com/TorchSpatiotemporal/tsl), [MIT license](experiments/spin_comparison_v1/upstream/tsl-0.1.1/LICENSE).

`experiments/spin_comparison_v1/source_model.py`, `positional.py`, and `scheduler.py` retain upstream structure/source with the adaptations recorded in their README. Attention and primitive implementations preserve the specified upstream computation in local PyTorch code. The reading bundles under `review_packet/spin_direct_sync_20260916/` contain the same attributed code and are covered by these notices as applicable.

Only the pinned source subset used by the CPU oracle, its configuration, and both licenses are included here. The historical `source_manifest.json` records the larger local upstream archive; it is not an inventory of the present publication. The SPIN paper is linked through the official proceedings, separately from the code license. New Chinese manuscript sources and research results are this project's work.
