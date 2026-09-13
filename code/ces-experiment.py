from ces import *
import os
import shutil


def run(seed, parameters, save_path):
    m = n = 200
    v = generate_market(n, m, parameters, seed=2022, has_zero=False)

    print("[  === CES utility ===  ]\n")
    ces = CES(v, rho=0.8)
    ces.solve_opt_cvxpy(record=False)

    np.random.seed(seed)

    os.makedirs(f'{save_path}/{seed}', exist_ok=True)

    iter_ = int(np.sqrt(n * m))

    if os.path.exists(f'{save_path}/{0}/pgls.csv') and seed != 0:
        shutil.copy(f'{save_path}/{0}/pgls.csv', f'{save_path}/{seed}/')
    else:
        pgls = ces.solve_pgls(num_iter=iter_, factor=(1e8, 0.60, 1.1), record=False, print_=iter_ // 10)
        store_in_cvx(pgls, f'{save_path}/{seed}/pgls.csv')

    if os.path.exists(f'{save_path}/{0}/pr.csv') and seed != 0:
        shutil.copy(f'{save_path}/{0}/pr.csv', f'{save_path}/{seed}/')
    else:
        pr = ces.solve_pr(num_iter=iter_, record=False, print_=iter_ // 10)
        store_in_cvx(pr, f'{save_path}/{seed}/pr.csv')

    bcdeg_ls = ces.solve_bcdeg(num_iter=iter_ * m, step_size='line_search', factor=(1, 0.60, 1.1),
                               record=False, print_=(iter_ * m) // 10)
    store_in_cvx(bcdeg_ls, f'{save_path}/{seed}/bcdeg_ls.csv')

    bcpr = ces.solve_bcpr(num_iter=iter_ * n, record=False, print_=(iter_ * n) // 10)
    store_in_cvx(bcpr, f'{save_path}/{seed}/bcpr.csv')


if __name__ == "__main__":
    paras = ((1, 1), (0.2, 0.2), 0.2)
    directory = f"data/sim-data-ces"
    path = os.path.join(os.path.pardir, directory)
    os.makedirs(path, exist_ok=True)

    seed_set = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    for s in seed_set:
        run(s, paras, path)
