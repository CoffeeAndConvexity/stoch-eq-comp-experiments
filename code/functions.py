import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Simulate a market instance
'''


def generate_market(n, m, paras, seed, has_zero=True):
    np.random.seed(seed)
    mu, var, eps_ = paras
    v_item = np.maximum(np.random.normal(loc=mu[0], scale=var[0], size=m), 0)
    v_buyer = np.maximum(np.random.normal(loc=mu[1], scale=var[1], size=n), 0)
    v = v_buyer.reshape(-1, 1) @ v_item.reshape(1, -1)
    eps = eps_ * np.random.uniform(size=(n, m))
    if has_zero:
        v = np.maximum(np.round(v + eps, 2), 0)
    else:
        v = np.maximum(np.round(v + eps, 2), 0.01)

    return v


'''
Explore data structure
'''


def sparsity(matrix):
    n, m = matrix.shape
    return np.count_nonzero(matrix) / (n * m)


def explore(matrix):
    n, m = matrix.shape
    count_nonzero = np.where(matrix == 0, 0, 1)
    info = "========= VALUATION MATRIX ========="
    info += f"\n# of blank columns:\t\t\t\t{sum(np.sum(matrix, axis=0) == 0)}"
    info += f"\n# of blank rows:\t\t\t\t{sum(np.sum(matrix, axis=1) == 0)}"
    info += f"\n# of 'only one' value columns:\t{sum(np.sum(count_nonzero, axis=0) == 1)} " \
            f"({np.round(100 * sum(np.sum(count_nonzero, axis=0) == 1) / m, 2)}%)"
    info += f"\n# of '<=5' value columns:\t\t{sum(np.sum(count_nonzero, axis=0) <= 5)} " \
            f"({np.round(100 * sum(np.sum(count_nonzero, axis=0) <= 5) / m, 2)}%)"
    info += f"\n# of 'only one' value rows:\t\t{sum(np.sum(count_nonzero, axis=1) == 1)} " \
            f"({np.round(100 * sum(np.sum(count_nonzero, axis=1) == 1) / n, 2)}%)"
    info += f"\n# of '<=5' value rows:\t\t\t{sum(np.sum(count_nonzero, axis=1) <= 5)} " \
            f"({np.round(100 * sum(np.sum(count_nonzero, axis=1) <= 5) / n, 2)}%)"
    info += "\n====================================\n"
    print(info)


'''
Following functions are used to project an array/matrix onto a simplex
'''


def compute_for_price(array, eta, index_list, b=1):
    a = np.sort(array)[::-1]
    cum_a = np.cumsum(a)
    i_ = np.maximum(np.sum(a * index_list > cum_a - b) - 1, 0)
    t = (cum_a[i_] - b) / (i_ + 1)

    return t / eta


def compute_for_price_all(matrix, eta, index_matrix, b=1):
    c = matrix.shape[1]
    a = - np.sort(- matrix, axis=0)
    cum_a = np.cumsum(a, axis=0)
    i_ = np.sum(a * index_matrix > cum_a - b, axis=0) - 1
    t = (cum_a[i_, range(c)] - b) / (i_ + 1)

    return t / eta


def project_weighted_simplex_array(array, w, b=1):
    n = array.shape[0]
    dec_sort = np.argsort(array / w)[::-1]
    array_, w_ = array[dec_sort], w[dec_sort]
    k = 0
    while k <= n - 1 and np.dot(w_[: k + 1], array_[: k + 1]) - b < array_[k] / w_[k] * np.sum(w_[: k + 1] ** 2):
        k += 1
    lambda_ = (np.dot(w_[: k], array_[: k]) - b) / np.sum(w_[: k] ** 2)
    projected_array = np.maximum(array - lambda_ * w, 0)

    return projected_array


def project_weighted_simplex_matrix(matrix, w_matrix, b=None):
    matrix_T = matrix.T
    w_matrix_T = w_matrix.T
    if b is None:
        projected_matrix = np.array(
            [project_weighted_simplex_array(matrix_T[i], w=w_matrix_T[i]) for i in range(matrix.shape[1])]).T
    else:
        projected_matrix = np.array(
            [project_weighted_simplex_array(matrix_T[i], w=w_matrix_T[i], b=b[i]) for i in range(matrix.shape[1])]).T

    return projected_matrix


'''
plot function
'''


def plot_multi_seed(result, x='cost', save_name=f'./plots/unnamed.jpg', metric='dual_gap'):
    plt.figure(figsize=(12, 12))
    y_label_code = {'dual_gap': 'dual gap',
                    'utility_gap(avg)': 'utility gap (average)',
                    'price_gap(avg)': 'price gap (average)'}
    for r in result:
        for algo in r:
            plt.plot(algo[0][x], algo[0][metric], label=algo[1][0], c=algo[1][1], linewidth=0.5, alpha=0.8)
            plt.ylabel(y_label_code[metric], fontsize=15)

    plt.xlabel('touched buyers/items', fontsize=15)
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(prop={'size': 10})
    plt.savefig(save_name)


def create_data():
    cost = 0
    data = {
        'cost': [],
        'Phi': [],
        'dualPhi': [],
        'dual_gap': [],
        'utility_gap(avg)': [],
        'utility_gap(max)': [],
        'price_gap(avg)': [],
        'price_gap(max)': []
    }

    return data, cost


def store_data(data, cost, Phi, dualPhi, ug, pg):
    data['cost'].append(cost)
    data['Phi'].append(Phi)
    data['dualPhi'].append(dualPhi)
    data['dual_gap'].append(dualPhi - Phi)
    data['utility_gap(avg)'].append(np.average(ug))
    data['utility_gap(max)'].append(max(ug))
    data['price_gap(avg)'].append(np.average(pg))
    data['price_gap(max)'].append(max(pg))


def store_in_cvx(data, path):
    df = pd.DataFrame(data)
    df.to_csv(path)
