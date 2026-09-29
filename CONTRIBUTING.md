# Contributing

This project treats batch invariance as an end-to-end, bitwise property: the
same target sample, weights, precision, configuration, and per-sample random
inputs must produce exactly the same robot output regardless of batch size or
batch companions.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the CPU-compatible harness tests with:

```bash
python -m pytest -q test_investigation_harness.py
```

Run the complete operator suite on a CUDA machine with:

```bash
python -m pytest -q
```

## Model integrations

New integrations should use a public PyTorch implementation and expose the
adapter contract in `scripts/model_invariance/adapters/`. Test repeatability,
duplicate composition, and unrelated composition. Explicitly control all
per-sample noise for diffusion or flow policies, continue tracing after the
first repaired divergence, and only report support when the final robot output
is bitwise exact.

Commit the compact evidence under `research/<model>/`, including the pinned
source revision, checkpoint revision, environment, baseline and fixed reports,
first-divergence trace, multiple-input result, benchmark, and investigation
notes. Do not commit model weights, datasets, extracted frames, or upstream
source checkouts.
