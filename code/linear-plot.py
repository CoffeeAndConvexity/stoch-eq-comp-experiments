import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os


def tick_spacing_selector(_min, _max, _num_ticks, potential_spacing):
    range = (_max - _min + 1) // 1e9 * 1e9
    for ps in potential_spacing:
        if _num_ticks * ps >= range:
            return ps

def plot(path_, algorithm_set, y_min=None):

    data = dict()
    label_dic = {'bcdeg': 'BCDEG', 'bcdeg_ls': 'BCDEG-LS', 'bcpr': 'BCPR', 'a_bcpr': 'A-BCPR', 'bcpr_ls': 'BCPR-LS',
                 'pgls': 'PGLS', 'pr': 'PR', 'pr_ls': 'PRLS'}

    num_bar = 8

    for algo in algorithm_set:
        for s in range(10):
            col_list = ['cost', 'dual_gap', 'utility_gap(avg)']
            header = {'dual_gap': f'dual_gap{s}',
                      'utility_gap(avg)': f'utility_gap(avg){s}'}
            data_path = os.path.join('../data', path_)
            df = pd.read_csv(f"{data_path}/{s}/{algo}.csv", usecols=col_list)
            df = df.rename(columns=header)
            if algo not in data.keys():
                data[algo] = df
            else:
                data[algo] = pd.merge(data[algo], df, on='cost', how='left')

        for metric in ('dual_gap', 'utility_gap(avg)'):
            columns_list = list([f'{metric}{seed}' for seed in range(10)])
            data[algo][f'{metric}_y'] = data[algo][columns_list].mean(axis=1)
            data[algo][f'{metric}_y_err'] = data[algo][columns_list].std(axis=1)

    def round_select(l1, l2, l3, array):
        selected_l1, selected_l2, selected_l3 = list(), list(), list()
        k = 0
        for i in range(len(l1)):
            if k < num_bar and l1[i] >= array[k]:
                selected_l1.append(l1[i])
                selected_l2.append(l2[i])
                selected_l3.append(l3[i])
                k += 1

        return selected_l1, selected_l2, selected_l3

    save_path = os.path.join('../plots', path_)
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 8))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    max_x = max(data[algorithm_set[0]]['cost'])
    for algo in algorithm_set:
        if max(data[algo]['cost']) < max_x:
            max_x = max(data[algo]['cost'])
    arr = np.floor(np.linspace(0, max_x, num_bar))

    for algo in algorithm_set:
        x, y, y_err = round_select(data[algo]['cost'], data[algo]['dual_gap' + '_y'],
                                   data[algo]['dual_gap' + '_y_err'], arr)
        ax1.errorbar(x=x, y=y, yerr=y_err, marker='o', markersize=3, linewidth=2, label=label_dic[algo], zorder=2)
        x, y, y_err = round_select(data[algo]['cost'], data[algo]['utility_gap(avg)' + '_y'],
                                   data[algo]['utility_gap(avg)' + '_y_err'], arr)
        ax2.errorbar(x=x, y=y, yerr=y_err, marker='o', markersize=3, linewidth=1, linestyle=':', label=label_dic[algo])

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    if y_min is not None:
        ax1.set_ylim(ymax=1, ymin=y_min[0])
        ax2.set_ylim(ymax=0.01, ymin=y_min[1])
    else:
        ax1.set_ylim(ymax=1)
        ax2.set_ylim(ymax=0.01)
    plt.legend(fontsize=20)

    ax1.set_xlabel('cost', fontsize=28)
    ax1.set_ylabel('dual gap', fontsize=28)
    ax2.set_ylabel('utility gap (average)', fontsize=28)

    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    potential_spacing = [2.5e9, 5e9]
    ax1.xaxis.set_ticks(np.arange(min(x), max(x) + 1, tick_spacing_selector(min(x), max(x), 2, potential_spacing)))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))

    plt.savefig(f"{save_path}/sim-data-linear-{'+'.join(algorithm_set)}", bbox_inches='tight', pad_inches=0.02)


if __name__ == "__main__":
    path = "sim-data-linear/"
    algorithms = ('bcdeg', 'bcdeg_ls')
    plot(path, algorithms)
    algorithms = ('bcpr', 'a_bcpr', 'bcpr_ls')
    plot(path, algorithms)
    algorithms = ('pr', 'pr_ls')
    plot(path, algorithms)
    algorithms = ('bcdeg_ls', 'pgls', 'bcpr_ls', 'pr_ls')
    plot(path, algorithms, y_min=(3e-5, 4e-6))
