from linear import *
import argparse
import os
from pathlib import Path


# -----------------------------------------------------------
# Test all algorithms based on real data
# - read data from .npy file
# - store experiment results to data directory
# -----------------------------------------------------------


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


def run(seed, real_data_file, save_path, iterations=None, max_buyers=None,
        max_goods=None, reference=None, skip_existing=False, reuse_seed_zero=False):
    print("[  === real data ===  ]")
    v = np.load(real_data_file)
    if max_buyers is not None:
        v = v[:max_buyers]
    if max_goods is not None:
        v = v[:, :max_goods]
    n, m = v.shape
    print(f"Input size: n * m = {n} * {m}\n")
    linear = Linear(v)  # create a (linear utility) instance defined by valuation matrix v

    # solve market equilibrium using cvxpy + MOSEK
    if reference is None:
        linear.solve_opt_cvxpy(record=False)
        reference = _capture_reference(linear)
    else:
        _restore_reference(linear, reference)

    os.makedirs(f'{save_path}/{seed}', exist_ok=True)

    # set the (total) number of (full-matrix-access) iterations for all algorithms
    iter_ = 10 * int(np.sqrt(n * m)) if iterations is None else int(iterations)

    def should_run(filename):
        return not (skip_existing and os.path.exists(f'{save_path}/{seed}/{filename}'))

    # ---
    # deterministic algorithms (PG-LS + PR + PR-LS)
    # ---

    # solve market equilibrium using Projected Gradient with Linear Search
    if not should_run('pgls.csv'):
        pass
    elif reuse_seed_zero and os.path.exists(f'{save_path}/{0}/pgls.csv') and seed != 0:
        copy_result(f'{save_path}/{0}/pgls.csv', f'{save_path}/{seed}/pgls.csv')
    else:
        pgls = linear.solve_pgls(num_iter=iter_, factor=(100, 0.80, 1.02), record=False, print_=max(iter_ // 10, 1))
        store_in_cvx(pgls, f'{save_path}/{seed}/pgls.csv')

    # solve market equilibrium using Proportional Response
    if not should_run('pr.csv'):
        pass
    elif reuse_seed_zero and os.path.exists(f'{save_path}/{0}/pr.csv') and seed != 0:
        copy_result(f'{save_path}/{0}/pr.csv', f'{save_path}/{seed}/pr.csv')
    else:
        pr = linear.solve_pr(num_iter=iter_, factor=(1, 1, 1), record=False, print_=max(iter_ // 10, 1))
        store_in_cvx(pr, f'{save_path}/{seed}/pr.csv')

    # solve market equilibrium using Proportional Response with Linear Search
    if not should_run('pr_ls.csv'):
        pass
    elif reuse_seed_zero and os.path.exists(f'{save_path}/{0}/pr_ls.csv') and seed != 0:
        copy_result(f'{save_path}/{0}/pr_ls.csv', f'{save_path}/{seed}/pr_ls.csv')
    else:
        pr_ls = linear.solve_pr(num_iter=iter_, factor=(1, 0.80, 1.02), record=False, print_=max(iter_ // 10, 1))
        store_in_cvx(pr_ls, f'{save_path}/{seed}/pr_ls.csv')

    # ---
    # randomized algorithms (BCDEG + BCDEG_LS + BCPR + A_BCPR + BCPR-LS)
    # ---

    # solve market equilibrium using Block Coordinate Descent for Eisenberg-Gale convex programming
    if should_run('bcdeg.csv'):
        _seed_algorithm(seed, 0)
        bcdeg = linear.solve_bcdeg(num_iter=iter_ * m, step_size='fixed_step', record=False,
                                   print_=max((iter_ * m) // 10, 1))
        store_in_cvx(bcdeg, f'{save_path}/{seed}/bcdeg.csv')

    # solve market equilibrium using Block Coordinate Descent for Eisenberg-Gale convex programming with Line Search
    if should_run('bcdeg_ls.csv'):
        _seed_algorithm(seed, 1)
        bcdeg_ls = linear.solve_bcdeg(num_iter=iter_ * m, step_size='line_search', record=False,
                                      print_=max((iter_ * m) // 10, 1))
        store_in_cvx(bcdeg_ls, f'{save_path}/{seed}/bcdeg_ls.csv')

    # solve market equilibrium using Block Coordinate Proportional Response
    if should_run('bcpr.csv'):
        _seed_algorithm(seed, 2)
        bcpr = linear.solve_bcpr(num_iter=iter_ * n, step_size='fixed_step', record=False,
                                 print_=max((iter_ * n) // 10, 1))
        store_in_cvx(bcpr, f'{save_path}/{seed}/bcpr.csv')

    # solve market equilibrium using Adaptive Block Coordinate Proportional Response
    if should_run('a_bcpr.csv'):
        _seed_algorithm(seed, 3)
        a_bcpr = linear.solve_bcpr(num_iter=iter_ * n, step_size='adaptive', record=False,
                                   print_=max((iter_ * n) // 10, 1))
        store_in_cvx(a_bcpr, f'{save_path}/{seed}/a_bcpr.csv')

    # solve market equilibrium using Block Coordinate Proportional Response with Line Search
    if should_run('bcpr_ls.csv'):
        _seed_algorithm(seed, 4)
        bcpr_ls = linear.solve_bcpr(num_iter=iter_ * n, step_size='line_search', record=False,
                                    print_=max((iter_ * n) // 10, 1))
        store_in_cvx(bcpr_ls, f'{save_path}/{seed}/bcpr_ls.csv')

    return reference


def _parse_seeds(value):
    return tuple(int(seed.strip()) for seed in value.split(',') if seed.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the MovieTweetings linear-utility experiment.')
    parser.add_argument('--quick', action='store_true', help='run a small three-seed data subset')
    parser.add_argument('--input', type=Path, default=REPO_ROOT / 'data' / 'movie_rating.npy')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--iterations', type=int, help='number of full-matrix-equivalent iterations')
    parser.add_argument('--max-buyers', type=int)
    parser.add_argument('--max-goods', type=int)
    parser.add_argument('--seeds', help='comma-separated seeds')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    read_path = str(args.input)
    store_path = str(args.output or (REPO_ROOT / 'data' / ('quick-real-data' if args.quick else 'real-data')))
    max_buyers = args.max_buyers if args.max_buyers is not None else (80 if args.quick else None)
    max_goods = args.max_goods if args.max_goods is not None else (80 if args.quick else None)
    iterations = args.iterations if args.iterations is not None else (40 if args.quick else None)

    os.makedirs(store_path, exist_ok=True)

    # random seed we used
    seed_set = _parse_seeds(args.seeds) if args.seeds else ((0, 1, 2) if args.quick else tuple(range(10)))

    reference = None
    seed_zero_ready = False
    for s in seed_set:
        reference = run(s, read_path, store_path, iterations=iterations,
                        max_buyers=max_buyers, max_goods=max_goods,
                        reference=reference, skip_existing=args.skip_existing,
                        reuse_seed_zero=seed_zero_ready)
        if s == 0:
            seed_zero_ready = True
