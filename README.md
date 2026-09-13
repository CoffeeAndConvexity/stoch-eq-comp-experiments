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

The experiments require Python 3 and the packages in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The reference-solution routines call MOSEK through CVXPY. A working MOSEK
installation and license are therefore required to run the experiment drivers
without modifying their solver configuration.

## Running the experiments

Run the scripts from the `code` directory because the original implementation
uses paths relative to that directory:

```bash
cd code

# Synthetic linear-utility experiments and plots
python linear-experiment.py
python linear-plot.py

# Synthetic CES-utility experiments and plots
python ces-experiment.py
python ces-plot.py

# Additional CES experiments from the appendix
python ces-extra-experiment.py
python ces-extra-plot.py
```

For the real-data experiment, first create `data/movie_rating.npy` from the
MovieTweetings 200K snapshot, then run the experiment and plotting scripts:

```bash
cd code
python movie-data-process.py
python real-data-experiment.py
python real-data-plot.py
```

The drivers use the random seeds and instance sizes from the paper. Some runs
are computationally intensive.

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
