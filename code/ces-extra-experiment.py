from ces import *
import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _seed_algorithm(seed, offset):
    """Give each randomized algorithm a stable, resume-independent stream."""
    np.random.seed((int(seed) + 1_000_003 * int(offset)) % (2 ** 32))


def _capture_reference(market):
    return {
        name: np.array(getattr(market, name), copy=True)
        for name in ('opt_x', 'opt_u', 'opt_p', 'opt_b')
    }


def _restore_reference(market, reference):
    for name, value in reference.items():
        setattr(market, name, np.array(value, copy=True))


def run(seed, parameters, save_path, n=200, m=200, iterations=None,
        market_seed=2022, reference=None, skip_existing=False, reuse_seed_zero=False):
    v = generate_market(n, m, parameters, seed=market_seed, has_zero=False)

    print("[  === CES utility ===  ]\n")
    ces = CES(v, rho=0.8)
    if reference is None:
        ces.solve_opt_cvxpy(record=False)
        reference = _capture_reference(ces)
    else:
        _restore_reference(ces, reference)

    os.makedirs(f'{save_path}/{seed}', exist_ok=True)

    iter_ = 4 * int(np.sqrt(n * m)) if iterations is None else int(iterations)

    def should_run(filename):
        return not (skip_existing and os.path.exists(f'{save_path}/{seed}/{filename}'))

    if not should_run('pgls.csv'):
        pass
    elif reuse_seed_zero and os.path.exists(f'{save_path}/{0}/pgls.csv') and seed != 0:
        copy_result(f'{save_path}/{0}/pgls.csv', f'{save_path}/{seed}/pgls.csv')
    else:
        pgls = ces.solve_pgls(num_iter=iter_, factor=(1e12, 0.60, 1.1), record=False, print_=max(iter_ // 10, 1))
        store_in_cvx(pgls, f'{save_path}/{seed}/pgls.csv')

    if not should_run('pr.csv'):
        pass
    elif reuse_seed_zero and os.path.exists(f'{save_path}/{0}/pr.csv') and seed != 0:
        copy_result(f'{save_path}/{0}/pr.csv', f'{save_path}/{seed}/pr.csv')
    else:
        pr = ces.solve_pr(num_iter=iter_, record=False, print_=max(iter_ // 10, 1))
        store_in_cvx(pr, f'{save_path}/{seed}/pr.csv')

    if should_run('bcdeg_ls.csv'):
        _seed_algorithm(seed, 0)
        bcdeg_ls = ces.solve_bcdeg(num_iter=iter_ * m, step_size='line_search', factor=(1, 0.60, 1.1),
                                   record=False, print_=max((iter_ * m) // 10, 1))
        store_in_cvx(bcdeg_ls, f'{save_path}/{seed}/bcdeg_ls.csv')

    if should_run('bcpr.csv'):
        _seed_algorithm(seed, 1)
        bcpr = ces.solve_bcpr(num_iter=iter_ * n, record=False, print_=max((iter_ * n) // 10, 1))
        store_in_cvx(bcpr, f'{save_path}/{seed}/bcpr.csv')

    return reference


def _parse_csv(value, cast):
    return tuple(cast(item.strip()) for item in value.split(',') if item.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the appendix CES volatility sweep.')
    parser.add_argument('--quick', action='store_true', help='run a small two-seed reproduction')
    parser.add_argument('--n', type=int)
    parser.add_argument('--m', type=int)
    parser.add_argument('--iterations', type=int, help='number of full-matrix-equivalent iterations')
    parser.add_argument('--seeds', help='comma-separated seeds')
    parser.add_argument('--volatilities', help='comma-separated volatility values')
    parser.add_argument('--market-seed', type=int, default=2022)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    n = args.n if args.n is not None else (40 if args.quick else 200)
    m = args.m if args.m is not None else (40 if args.quick else 200)
    iterations = args.iterations if args.iterations is not None else (80 if args.quick else None)
    seed_set = _parse_csv(args.seeds, int) if args.seeds else ((0, 1) if args.quick else (0, 1, 2))
    volatilities = _parse_csv(args.volatilities, float) if args.volatilities else (0.4, 0.6, 0.8, 1.0)
    output_root = args.output or (REPO_ROOT / 'data' / ('quick-sim-data-ces-extra' if args.quick else 'sim-data-ces-extra'))

    for v in volatilities:
        paras = ((1, 1), (v, v), v)
        path = str(output_root / f'{v:g}')
        os.makedirs(path, exist_ok=True)

        reference = None
        seed_zero_ready = False
        for s in seed_set:
            reference = run(s, paras, path, n=n, m=m, iterations=iterations,
                            market_seed=args.market_seed, reference=reference,
                            skip_existing=args.skip_existing, reuse_seed_zero=seed_zero_ready)
            if s == 0:
                seed_zero_ready = True
