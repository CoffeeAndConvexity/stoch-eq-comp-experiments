import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import argparse
import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tick_spacing_selector(_min, _max, _num_ticks, potential_spacing):
    span = max(float(_max) - float(_min), 0.0)
    if span == 0:
        return 1.0
    for ps in potential_spacing:
        if ps <= span and _num_ticks * ps >= span:
            return ps

    target = span / max(_num_ticks, 1)
    magnitude = 10 ** np.floor(np.log10(target))
    for multiplier in (1, 2, 2.5, 5, 10):
        spacing = multiplier * magnitude
        if spacing >= target:
            return spacing


def _available_seeds(path_, algorithm_set):
    data_path = os.path.join(REPO_ROOT, 'data', path_)
    seed_sets = []
    for algo in algorithm_set:
        seeds = {
            int(entry) for entry in os.listdir(data_path)
            if entry.isdigit() and os.path.isfile(os.path.join(data_path, entry, f'{algo}.csv'))
        }
        seed_sets.append(seeds)
    seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if not seeds:
        raise FileNotFoundError(f'No complete seed data found in {data_path} for {algorithm_set}')
    return seeds


def plot(path_, algorithm_set, y_min=None, seeds=None):
    data = dict()
    label_dic = {'bcdeg': 'BCDEG', 'bcdeg_ls': 'BCDEG-LS', 'bcpr': 'BCPR', 'a_bcpr': 'A-BCPR', 'bcpr_ls': 'BCPR-LS',
                 'pgls': 'PGLS', 'pr': 'PR', 'pr_ls': 'PRLS'}

    num_bar = 8

    seeds = _available_seeds(path_, algorithm_set) if seeds is None else list(seeds)
    for algo in algorithm_set:
        for s in seeds:
            col_list = ['cost', 'dual_gap', 'utility_gap(avg)']
            header = {'dual_gap': f'dual_gap{s}',
                      'utility_gap(avg)': f'utility_gap(avg){s}'}
            data_path = os.path.join(REPO_ROOT, 'data', path_)
            df = pd.read_csv(f"{data_path}/{s}/{algo}.csv", usecols=col_list)
            df = df.rename(columns=header)
            if algo not in data.keys():
                data[algo] = df
            else:
                data[algo] = pd.merge(data[algo], df, on='cost', how='left')

        for metric in ('dual_gap', 'utility_gap(avg)'):
            columns_list = [f'{metric}{seed}' for seed in seeds]
            data[algo][f'{metric}_y'] = data[algo][columns_list].mean(axis=1)
            data[algo][f'{metric}_y_err'] = data[algo][columns_list].std(axis=1, ddof=0)

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

    save_path = os.path.join(REPO_ROOT, 'plots', path_)
    os.makedirs(save_path, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 8))
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
        ax1.set_ylim(ymin=y_min[0])
        ax2.set_ylim(ymin=y_min[1])
    ax1.legend(fontsize=20)

    ax1.set_xlabel('cost', fontsize=28)
    ax1.set_ylabel('dual gap', fontsize=28)
    ax2.set_ylabel('utility gap (average)', fontsize=28)

    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    potential_spacing = [1.4e9, 1.5e9, 2.8e9, 3e9, 5e9]
    spacing = tick_spacing_selector(0, max_x, 2, potential_spacing)
    ax1.xaxis.set_ticks(np.arange(0, max_x + spacing, spacing))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
    # ax1.set_yticks(ax1.get_yticks()[::4])

    fig.savefig(f"{save_path}/real-data-{'+'.join(algorithm_set)}.png", bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the MovieTweetings experiment.')
    parser.add_argument('--path', default='real-data', help='directory name below data/ and plots/')
    args = parser.parse_args()
    path = args.path
    algorithms = ('bcdeg', 'bcdeg_ls')
    plot(path, algorithms)
    algorithms = ('bcpr', 'a_bcpr', 'bcpr_ls')
    plot(path, algorithms)
    algorithms = ('pr', 'pr_ls')
    plot(path, algorithms)
    algorithms = ('bcdeg_ls', 'pgls', 'bcpr_ls', 'pr_ls')
    plot(path, algorithms, y_min=(1e-11, 1e-6))
