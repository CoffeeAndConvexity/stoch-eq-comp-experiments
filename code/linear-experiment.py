from linear import *
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


def run(seed, parameters, save_path, n=400, m=400, iterations=None,
        market_seed=2022, reference=None, skip_existing=False, reuse_seed_zero=False):
    v = generate_market(n, m, parameters, seed=market_seed)

    print("[  === linear utility ===  ]\n")
    linear = Linear(v)
    if reference is None:
        linear.solve_opt_cvxpy(record=False)
        reference = _capture_reference(linear)
    else:
        _restore_reference(linear, reference)

    os.makedirs(f'{save_path}/{seed}', exist_ok=True)

    iter_ = int(0.2 * n * m) if iterations is None else int(iterations)

    def should_run(filename):
        return not (skip_existing and os.path.exists(f'{save_path}/{seed}/{filename}'))

    if not should_run('pgls.csv'):
        pass
    elif reuse_seed_zero and os.path.exists(f'{save_path}/{0}/pgls.csv') and seed != 0:
        copy_result(f'{save_path}/{0}/pgls.csv', f'{save_path}/{seed}/pgls.csv')
    else:
        pgls = linear.solve_pgls(num_iter=iter_, factor=(100, 0.80, 1.02), record=False, print_=max(iter_ // 10, 1))
        store_in_cvx(pgls, f'{save_path}/{seed}/pgls.csv')

    if not should_run('pr_ls.csv'):
        pass
    elif reuse_seed_zero and os.path.exists(f'{save_path}/{0}/pr_ls.csv') and seed != 0:
        copy_result(f'{save_path}/{0}/pr_ls.csv', f'{save_path}/{seed}/pr_ls.csv')
    else:
        pr_ls = linear.solve_pr(num_iter=iter_, factor=(1, 0.80, 1.02), record=False, print_=max(iter_ // 10, 1))
        store_in_cvx(pr_ls, f'{save_path}/{seed}/pr_ls.csv')

    if not should_run('pr.csv'):
        pass
    elif reuse_seed_zero and os.path.exists(f'{save_path}/{0}/pr.csv') and seed != 0:
        copy_result(f'{save_path}/{0}/pr.csv', f'{save_path}/{seed}/pr.csv')
    else:
        pr = linear.solve_pr(num_iter=iter_, factor=(1, 1, 1), record=False, print_=max(iter_ // 10, 1))
        store_in_cvx(pr, f'{save_path}/{seed}/pr.csv')

    if should_run('bcdeg.csv'):
        _seed_algorithm(seed, 0)
        bcdeg = linear.solve_bcdeg(num_iter=iter_ * m, step_size='fixed_step', record=False,
                                   print_=max((iter_ * m) // 10, 1))
        store_in_cvx(bcdeg, f'{save_path}/{seed}/bcdeg.csv')

    if should_run('bcdeg_ls.csv'):
        _seed_algorithm(seed, 1)
        bcdeg_ls = linear.solve_bcdeg(num_iter=iter_ * m, step_size='line_search', record=False,
                                      print_=max((iter_ * m) // 10, 1))
        store_in_cvx(bcdeg_ls, f'{save_path}/{seed}/bcdeg_ls.csv')

    if should_run('bcpr.csv'):
        _seed_algorithm(seed, 2)
        bcpr = linear.solve_bcpr(num_iter=iter_ * n, step_size='fixed_step', record=False,
                                 print_=max((iter_ * n) // 10, 1))
        store_in_cvx(bcpr, f'{save_path}/{seed}/bcpr.csv')

    if should_run('a_bcpr.csv'):
        _seed_algorithm(seed, 3)
        a_bcpr = linear.solve_bcpr(num_iter=iter_ * n, step_size='adaptive', record=False,
                                   print_=max((iter_ * n) // 10, 1))
        store_in_cvx(a_bcpr, f'{save_path}/{seed}/a_bcpr.csv')

    if should_run('bcpr_ls.csv'):
        _seed_algorithm(seed, 4)
        bcpr_ls = linear.solve_bcpr(num_iter=iter_ * n, step_size='line_search', record=False,
                                    print_=max((iter_ * n) // 10, 1))
        store_in_cvx(bcpr_ls, f'{save_path}/{seed}/bcpr_ls.csv')

    return reference


def _parse_seeds(value):
    return tuple(int(seed.strip()) for seed in value.split(',') if seed.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the synthetic linear-utility experiment.')
    parser.add_argument('--quick', action='store_true', help='run a small three-seed reproduction')
    parser.add_argument('--n', type=int)
    parser.add_argument('--m', type=int)
    parser.add_argument('--iterations', type=int, help='number of full-matrix-equivalent iterations')
    parser.add_argument('--seeds', help='comma-separated seeds')
    parser.add_argument('--market-seed', type=int, default=2022)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    paras = ((1, 1), (1, 1), 1)
    n = args.n if args.n is not None else (40 if args.quick else 400)
    m = args.m if args.m is not None else (40 if args.quick else 400)
    iterations = args.iterations if args.iterations is not None else (80 if args.quick else None)
    seed_set = _parse_seeds(args.seeds) if args.seeds else ((0, 1, 2) if args.quick else tuple(range(10)))
    path = str(args.output or (REPO_ROOT / 'data' / ('quick-sim-data-linear' if args.quick else 'sim-data-linear')))
    os.makedirs(path, exist_ok=True)

    reference = None
    seed_zero_ready = False
    for s in seed_set:
        reference = run(s, paras, path, n=n, m=m, iterations=iterations,
                        market_seed=args.market_seed, reference=reference,
                        skip_existing=args.skip_existing, reuse_seed_zero=seed_zero_ready)
        if s == 0:
            seed_zero_ready = True
