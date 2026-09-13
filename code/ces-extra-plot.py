import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os

def tick_spacing_selector(_min, _max, _num_ticks, potential_spacing):
    range = (_max - _min + 1) // 1e6 * 1e6
    for ps in potential_spacing:
        if _num_ticks * ps >= range:
            return ps

def plot(path_, algorithm_set, y_min=None):
    data = dict()
    label_dic = {'bcdeg': 'BCDEG', 'bcdeg_ls': 'BCDEG-LS', 'bcpr': 'BCPR', 'a_bcpr': 'A-BCPR', 'bcpr_ls': 'BCPR-LS',
                 'pgls': 'PGLS', 'pr': 'PR', 'prls': 'PRLS'}

    num_bar = 8

    for algo in algorithm_set:
        for s in range(3):
            col_list = ['cost', 'dual_gap', ]
            header = {'dual_gap': f'dual_gap{s}', }
            data_path = os.path.join('../data', path_)
            df = pd.read_csv(f"{data_path}/{s}/{algo}.csv", usecols=col_list)
            df = df.rename(columns=header)
            if algo not in data.keys():
                data[algo] = df
            else:
                data[algo] = pd.merge(data[algo], df, on='cost', how='left')

        for metric in ('dual_gap', ):
            columns_list = list([f'{metric}{seed}' for seed in range(3)])
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

    plt.figure(figsize=(8, 6))
    fig, ax1 = plt.subplots()

    max_x = max(data[algorithm_set[0]]['cost'])
    for algo in algorithm_set:
        if max(data[algo]['cost']) < max_x:
            max_x = max(data[algo]['cost'])
    arr = np.floor(np.linspace(0, max_x, num_bar))

    for algo in algorithm_set:
        x, y, y_err = round_select(data[algo]['cost'], data[algo]['dual_gap' + '_y'],
                                   data[algo]['dual_gap' + '_y_err'], arr)
        plt.errorbar(x=x, y=y, yerr=y_err, marker='o', markersize=3, linewidth=2, label=label_dic[algo])

    plt.yscale('log')
    plt.legend(fontsize=20)

    plt.xlabel('cost', fontsize=28)
    plt.ylabel('dual gap', fontsize=28)


    ax1.tick_params(labelsize=20)
    potential_spacing = [2e6, 4e6, 6e6]
    ax1.xaxis.set_ticks(np.arange(min(x), max(x) + 1, tick_spacing_selector(min(x), max(x), 2, potential_spacing)))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))

    plt.savefig(f"{save_path}/{'+'.join(algorithm_set)}", bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
    os.makedirs("../plots/sim-data-ces-extra", exist_ok=True)
    for v in (0.4, 0.6, 0.8, 1):
        path = f"sim-data-ces-extra/{v}/"
        algorithms = ('bcpr', 'pr')
        plot(path, algorithms)
        algorithms = ('bcdeg_ls', 'pgls')
        plot(path, algorithms)
