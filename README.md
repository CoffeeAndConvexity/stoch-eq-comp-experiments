# Fast and Interpretable Dynamics for Fisher Markets

This repository contains the code for the numerical experiments in
[Fast and Interpretable Dynamics for Fisher Markets via Block-Coordinate Updates](https://doi.org/10.1609/aaai.v37i5.25723)
by Tianlong Nan, Yuan Gao, and Christian Kroer (AAAI 2023).

The implementation compares full-information and block-coordinate methods for
computing equilibria in Fisher markets with linear and CES utilities.

## Repository layout

- `code/linear.py` and `code/ces.py`: market models and equilibrium algorithms.
- `code/functions.py`: shared data-generation, projection, logging, and plotting helpers.
- `code/*-experiment.py`: synthetic linear, MovieTweetings, CES, and additional CES experiments.
- `code/*-plot.py`: scripts that produce the figures from experiment CSV files.
- `code/movie-data-process.py`: downloads and preprocesses the MovieTweetings 200K snapshot.
- `data/` and `plots/`: generated outputs; their contents are intentionally not versioned.

## Setup

The current code has been tested with Python 3.12.2 and the pinned package
versions in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MOSEK is recommended for computing the reference equilibria. The requirements
file includes the tested MOSEK Python package, but using it still requires a
valid MOSEK license. If MOSEK is unavailable or cannot solve a problem, the
code automatically falls back to the open-source CLARABEL or SCS solver through
CVXPY. The fallback makes the experiments runnable without a MOSEK license,
although reference values and run times can differ slightly.

## Running the experiments

All commands below are run from the repository root. Experiment CSV files are
written below `data/`, and figures are written below `plots/`.

### Quick reproduction

The quick profiles use smaller markets, fewer iterations, and two or three
random seeds. They exercise every algorithm and reproduce the structure of the
paper figures in a few minutes on a typical development machine.

```bash
# Synthetic linear-utility experiments and plots
python code/linear-experiment.py --quick
python code/linear-plot.py --path quick-sim-data-linear

# Synthetic CES-utility experiments and plots
python code/ces-experiment.py --quick
python code/ces-plot.py --path quick-sim-data-ces

# Additional CES experiments from the appendix
python code/ces-extra-experiment.py --quick
python code/ces-extra-plot.py --path quick-sim-data-ces-extra
```

For the real-data experiment, first create `data/movie_rating.npy` from the
MovieTweetings 200K snapshot, then run the experiment and plotting scripts:

```bash
python code/movie-data-process.py
python code/real-data-experiment.py --quick
python code/real-data-plot.py --path quick-real-data
```

The preprocessing command downloads the upstream snapshot and therefore needs
network access. Use `--base-url`, `--output`, or `--min-ratings` to override its
input location, output file, or filtering threshold.

### Full paper-default runs

Without `--quick`, the drivers use the paper-scale instance sizes, iteration
budgets, and random seeds:

```bash
python code/linear-experiment.py --skip-existing
python code/linear-plot.py

python code/ces-experiment.py --skip-existing
python code/ces-plot.py

python code/ces-extra-experiment.py --skip-existing
python code/ces-extra-plot.py

python code/movie-data-process.py
python code/real-data-experiment.py --skip-existing
python code/real-data-plot.py
```

`--skip-existing` makes experiment runs resumable at the per-algorithm CSV
level. It is safe on a first run and avoids recomputing completed CSV files
after an interruption. The full synthetic linear and MovieTweetings runs are
especially computationally intensive: each block-coordinate method performs
many millions of Python-level coordinate updates across ten seeds. Expect these
runs to take substantially longer than the quick profiles.

The synthetic drivers also accept `--n`, `--m`, `--iterations`, `--seeds`,
`--market-seed`, and `--output`. The real-data driver accepts `--input`,
`--iterations`, `--max-buyers`, `--max-goods`, `--seeds`, and `--output`.
The CES driver additionally accepts `--algorithms` for targeted reruns (for
example, `--algorithms bcdeg_ls,bcpr`).
Seeds and CES volatility values are supplied as comma-separated lists, for
example:

```bash
python code/ces-experiment.py --n 100 --m 100 --iterations 100 --seeds 0,1,2
python code/ces-extra-experiment.py --volatilities 0.4,0.8,1 --skip-existing
```

## Data

No dataset or derived valuation matrix is included in this repository. The
real-data preprocessing script downloads the
[MovieTweetings 200K snapshot](https://github.com/sidooms/MovieTweetings/tree/master/snapshots/200K)
from its upstream project and writes the processed matrix to
`data/movie_rating.npy`. Consult the upstream project for its data terms.

## Citation

```bibtex
@article{nan2023fast,
  title   = {Fast and Interpretable Dynamics for Fisher Markets via Block-Coordinate Updates},
  author  = {Nan, Tianlong and Gao, Yuan and Kroer, Christian},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {37},
  number  = {5},
  pages   = {5832--5840},
  year    = {2023},
  doi     = {10.1609/aaai.v37i5.25723}
}
```
