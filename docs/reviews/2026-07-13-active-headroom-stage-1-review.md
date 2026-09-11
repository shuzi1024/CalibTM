# Active Headroom v1 — Stage 1 prelaunch review

Date: 2026-07-13 UTC

Status: **engineering gate passed; formal scientific gate not yet run**.

## Scope frozen for the oracle stage

- Dataset/split: GEANT validation only; test access is absent from the formal API and CLI.
- Masks: `internal` and `burst`, mask seed `24001`.
- Fixed active partition: seed `27001`; immutable evaluation set `E` and query set `Q`.
- Budgets: balanced `K in {0,1,2,4}` per flow and global `B=462K`, with cap 8 per flow.
- Deployable families: random (seeds `34001..34008`), coverage, geometry uncertainty, and hybrid.
- Diagnostics: exact balanced, exact global DP/MCKP, and truth-greedy global.
- Headline metric: ratio-of-sums NMAE on fixed `E`, normalized trapezoidal AUC.
- Safety metric: fixed initial-missing `H0` degradation and direct/spillover decomposition.

## Verification evidence

- Repository suite: `193 passed in 19.76s`.
- Active-headroom suite before final gate CLI: `126 passed in 12.64s`.
- Manifest/launcher suite: `22 passed`.
- Gate suite after vectorized bootstrap and receipt binding: `31 passed`.
- Exact flow DP and global allocation had independent brute-force review in the prior audit.
- Canonical parsed train/validation hashing opens no raw dataset path and has one protocol hash definition.

## Real-data smoke timing (4 CPU threads per process)

First validation window, start `7572`, all deterministic policies plus exact and truth-greedy:

| mask | deterministic time | random-only time | deterministic rows | random rows |
|---|---:|---:|---:|---:|
| internal | 139.31 s | 0.064 s | 37 | 8 |
| burst | 187.72 s | 0.074 s | 37 | 8 |

Estimated wall time for four concurrent 16-window shards is about 38–51 minutes before filesystem variance. Random-only materialization avoids recomputing exact and truth-greedy for each of eight seeds.

Smoke row hashes:

- internal: `4aaacb88a56cc691967c4dba0c19c922e20628ccef1b6491cfce81674973b8ee`
- burst: `d9cb8748d559978b3a9a2435aa4252b651abb648da8e3caea0f931564e9a20c0`

## Execution and evidence controls

- Full config equality is checked, including policies, cap, reconstructor, bootstrap, and gate.
- Source, parsed train/validation, window cohorts, partitions, preregistration, manifest jobs, and outputs are content-addressed.
- The launcher runs exactly four audited jobs, injects each immutable job identity, writes atomic status receipts, and resumes only when receipt identity and output bytes agree.
- Gate analysis loads only manifest-listed outputs with matching receipts and verifies query legality, exact budget, balanced counts, per-flow cap, fixed scopes, K=0 identity, and `H0` decomposition.
- Zero or negative oracle headroom is reported as `kill`, not treated as an infrastructure exception.

## Current decision

Do not claim that budget-constrained active progressive completion works yet. Launch is permitted only after the final independent read-only audits return without a P0 issue and a fresh source/data/preregistration/manifest snapshot is generated. The next scientific decision is entirely metric-conditional: `proceed`, `revise`, or `kill` according to the preregistered two-mask oracle gate.
