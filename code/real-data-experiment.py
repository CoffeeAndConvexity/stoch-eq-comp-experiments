from linear import *
import os
import shutil


# -----------------------------------------------------------
# Test all algorithms based on real data
# - read data from .npy file
# - store experiment results to data directory
# -----------------------------------------------------------


def run(seed, real_data_file, save_path):
    print("[  === real data ===  ]")
    v = np.load(real_data_file)
    n, m = v.shape
    print(f"Input size: n * m = {n} * {m}\n")
    linear = Linear(v)  # create a (linear utility) instance defined by valuation matrix v

    # solve market equilibrium using cvxpy + MOSEK
    linear.solve_opt_cvxpy(record=False)

    # set random seed
    np.random.seed(seed)

    os.makedirs(f'{save_path}/{seed}', exist_ok=True)

    # set the (total) number of (full-matrix-access) iterations for all algorithms
    iter_ = 10 * int(np.sqrt(n * m))

    # ---
    # deterministic algorithms (PG-LS + PR + PR-LS)
    # ---

    # solve market equilibrium using Projected Gradient with Linear Search
    if os.path.exists(f'{save_path}/{0}/pgls.csv') and seed != 0:
        shutil.copy(f'{save_path}/{0}/pgls.csv', f'{save_path}/{seed}/')
    else:
        pgls = linear.solve_pgls(num_iter=iter_, factor=(100, 0.80, 1.02), record=False, print_=iter_ // 10)
        store_in_cvx(pgls, f'{save_path}/{seed}/pgls.csv')

    # solve market equilibrium using Proportional Response
    if os.path.exists(f'{save_path}/{0}/pr.csv') and seed != 0:
        shutil.copy(f'{save_path}/{0}/pr.csv', f'{save_path}/{seed}/')
    else:
        pr = linear.solve_pr(num_iter=iter_, factor=(1, 1, 1), record=False, print_=iter_ // 10)
        store_in_cvx(pr, f'{save_path}/{seed}/pr.csv')

    # solve market equilibrium using Proportional Response with Linear Search
    if os.path.exists(f'{save_path}/{0}/pr_ls.csv') and seed != 0:
        shutil.copy(f'{save_path}/{0}/pr_ls.csv', f'{save_path}/{seed}/')
    else:
        pr_ls = linear.solve_pr(num_iter=iter_, factor=(1, 0.80, 1.02), record=False, print_=iter_ // 10)
        store_in_cvx(pr_ls, f'{save_path}/{seed}/pr_ls.csv')

    # ---
    # randomized algorithms (BCDEG + BCDEG_LS + BCPR + A_BCPR + BCPR-LS)
    # ---

    # solve market equilibrium using Block Coordinate Descent for Eisenberg-Gale convex programming
    bcdeg = linear.solve_bcdeg(num_iter=iter_ * m, step_size='fixed_step', record=False, print_=(iter_ * m) // 10)
    store_in_cvx(bcdeg, f'{save_path}/{seed}/bcdeg.csv')

    # solve market equilibrium using Block Coordinate Descent for Eisenberg-Gale convex programming with Line Search
    bcdeg_ls = linear.solve_bcdeg(num_iter=iter_ * m, step_size='line_search', record=False, print_=(iter_ * m) // 10)
    store_in_cvx(bcdeg_ls, f'{save_path}/{seed}/bcdeg_ls.csv')

    # solve market equilibrium using Block Coordinate Proportional Response
    bcpr = linear.solve_bcpr(num_iter=iter_ * n, step_size='fixed_step', record=False, print_=(iter_ * n) // 10)
    store_in_cvx(bcpr, f'{save_path}/{seed}/bcpr.csv')

    # solve market equilibrium using Adaptive Block Coordinate Proportional Response
    a_bcpr = linear.solve_bcpr(num_iter=iter_ * n, step_size='adaptive', record=False, print_=(iter_ * n) // 10)
    store_in_cvx(a_bcpr, f'{save_path}/{seed}/a_bcpr.csv')

    # solve market equilibrium using Block Coordinate Proportional Response with Line Search
    bcpr_ls = linear.solve_bcpr(num_iter=iter_ * n, step_size='line_search', record=False, print_=(iter_ * n) // 10)
    store_in_cvx(bcpr_ls, f'{save_path}/{seed}/bcpr_ls.csv')


if __name__ == "__main__":
    read_path = '../data/movie_rating.npy'
    store_path = "../data/real-data"

    os.makedirs(store_path, exist_ok=True)

    # random seed we used
    seed_set = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    for s in seed_set:
        run(s, read_path, store_path)
