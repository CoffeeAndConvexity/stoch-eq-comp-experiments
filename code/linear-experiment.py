from linear import *
import os
import shutil


def run(seed, parameters, save_path):
    m = n = 400
    v = generate_market(n, m, parameters, seed=2022)

    print("[  === linear utility ===  ]\n")
    linear = Linear(v)
    linear.solve_opt_cvxpy(record=False)

    np.random.seed(seed)

    os.makedirs(f'{save_path}/{seed}', exist_ok=True)

    iter_ = int(0.2 * n * m)

    if os.path.exists(f'{save_path}/{0}/pgls.csv') and seed != 0:
        shutil.copy(f'{save_path}/{0}/pgls.csv', f'{save_path}/{seed}/')
    else:
        pgls = linear.solve_pgls(num_iter=iter_, factor=(100, 0.80, 1.02), record=False, print_=iter_ // 10)
        store_in_cvx(pgls, f'{save_path}/{seed}/pgls.csv')

    if os.path.exists(f'{save_path}/{0}/pr_ls.csv') and seed != 0:
        shutil.copy(f'{save_path}/{0}/pr_ls.csv', f'{save_path}/{seed}/')
    else:
        pr_ls = linear.solve_pr(num_iter=iter_, factor=(1, 0.80, 1.02), record=False, print_=iter_ // 10)
        store_in_cvx(pr_ls, f'{save_path}/{seed}/pr_ls.csv')

    if os.path.exists(f'{save_path}/{0}/pr.csv') and seed != 0:
        shutil.copy(f'{save_path}/{0}/pr.csv', f'{save_path}/{seed}/')
    else:
        pr = linear.solve_pr(num_iter=iter_, factor=(1, 1, 1), record=False, print_=iter_ // 10)
        store_in_cvx(pr, f'{save_path}/{seed}/pr.csv')

    bcdeg = linear.solve_bcdeg(num_iter=iter_ * m, step_size='fixed_step', record=False, print_=(iter_ * m) // 10)
    store_in_cvx(bcdeg, f'{save_path}/{seed}/bcdeg.csv')

    bcdeg_ls = linear.solve_bcdeg(num_iter=iter_ * m, step_size='line_search', record=False, print_=(iter_ * m) // 10)
    store_in_cvx(bcdeg_ls, f'{save_path}/{seed}/bcdeg_ls.csv')

    bcpr = linear.solve_bcpr(num_iter=iter_ * n, step_size='fixed_step', record=False, print_=(iter_ * n) // 10)
    store_in_cvx(bcpr, f'{save_path}/{seed}/bcpr.csv')

    a_bcpr = linear.solve_bcpr(num_iter=iter_ * n, step_size='adaptive', record=False, print_=(iter_ * n) // 10)
    store_in_cvx(a_bcpr, f'{save_path}/{seed}/a_bcpr.csv')

    bcpr_ls = linear.solve_bcpr(num_iter=iter_ * n, step_size='line_search', record=False, print_=(iter_ * n) // 10)
    store_in_cvx(bcpr_ls, f'{save_path}/{seed}/bcpr_ls.csv')


if __name__ == "__main__":
    paras = ((1, 1), (1, 1), 1)
    directory = f"data/sim-data-linear"
    path = os.path.join(os.path.pardir, directory)
    os.makedirs(path, exist_ok=True)

    seed_set = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    for s in seed_set:
        run(s, paras, path)
