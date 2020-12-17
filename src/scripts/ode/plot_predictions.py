import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.scripts.ode.integrate_predict import get_info_from_filename
from src.scripts.config import results_path
import numpy as np


def var_target_maxpoly(f: str):
    var_names, n_vars, d_max, poly_max, params_set, init_cond, steps = get_info_from_filename(filename=f)
    return '{}_{}_{}'.format(var_names, d_max, poly_max)


reds = ['#fb6a4a', '#cb181d', '#67000d', '#a50f15', '#ef3b2c']
blues = ['#3182bd', '#9ecae1']


def plot_predictions(path_gpomo, time_horizons, path_pdefind=None, ymax=None, targets_to_plot=[2, 3]):
    plot_data = {}
    if path_gpomo:
        df_gpomo = pd.read_csv(Path.joinpath(results_path, path_gpomo, 'summary.csv'), index_col=0)
    if path_pdefind:
        df_pdefind = pd.read_csv(Path.joinpath(results_path, path_pdefind, 'summary.csv'), index_col=0)

    if path_gpomo and path_pdefind:
        models_list = [['GPoMo', df_gpomo], ['L-ODEfind', df_pdefind]]
        name = path_gpomo
    if path_gpomo and not path_pdefind:
        models_list = [['GPoMo', df_gpomo]]
        name = path_gpomo
    if not path_gpomo and path_pdefind:
        models_list = [['L-ODEfind', df_pdefind]]
        name = path_pdefind
    matplotlib.rcParams.update({'font.size': 25})
    markers = 'ovs^*XD'

    colors = blues[:len(set([2, 3]).intersection(set(targets_to_plot)))] + reds[:len(targets_to_plot)]
    plt.figure(figsize=(15, 10))
    i = 0
    for model, df in models_list:
        df['params'] = df['model'].apply(var_target_maxpoly)
        df = df[df['time_horizon'].isin(time_horizons)]
        #    sns.boxplot(x='time_horizon', y='smape', hue='params', data=df)
        g = df.groupby(['params', 'time_horizon'], as_index=False)['smape'].median()
        pms = g['params'].unique()
        for p in pms:
            my_df = g[g['params'] == p]
            target = int(p.split('_')[1])
            if target in targets_to_plot:
                plt.plot(my_df['time_horizon'], my_df['smape'], label=model + str(target),
                         marker=markers[i], markersize=10, color=colors[i])
                plot_data[f'{model}-{p}'] = [my_df['time_horizon'], my_df['smape']]
                i = i + 1

    plt.xticks([t for i, t in enumerate(time_horizons) if i % 3 == 0])
    # plt.title('{}'.format(path_gpomo))
    plt.xlabel('Time horizon steps')
    plt.ylabel('SMAPE')
    if ymax:
        plt.ylim(None, ymax)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(Path.joinpath(results_path, name + '-smape.jpeg'))
    return plot_data


def target(f: str):
    var_names, n_vars, d_max, poly_max, params_set, init_cond, steps = get_info_from_filename(filename=f)
    return 'Target time derivative {}'.format(d_max)


def plot_times(path_gpomo, path_pdefind=None, targets_to_plot=[]):
    targets_to_plot = ['Target time derivative {}'.format(i) for i in targets_to_plot]
    if path_gpomo:
        df_gpomo = pd.read_csv(Path.joinpath(results_path, path_gpomo, 'times.csv'), index_col=0)
        df_gpomo = df_gpomo.T
        df_gpomo.columns = ['time']
        df_gpomo['model'] = [s.replace('.', '-') for s in df_gpomo.index.values]
        df_gpomo['params'] = df_gpomo['model'].apply(target)
        df_gpomo['method'] = 'GPoMo'
        df_gpomo = df_gpomo[df_gpomo['params'].isin(targets_to_plot)]

    if path_pdefind:
        df_pdefind = pd.read_csv(Path.joinpath(results_path, path_pdefind, 'times.csv'), index_col=0)
        df_pdefind['params'] = df_pdefind['model'].apply(target)
        df_pdefind = df_pdefind[df_pdefind['params'].isin(targets_to_plot)]
        df_pdefind['method'] = 'L-ODEfind'

    if path_gpomo and path_pdefind:
        df = pd.concat([df_pdefind, df_gpomo])
        name = path_gpomo
    if path_gpomo and not path_pdefind:
        df = df_gpomo
        name = path_gpomo
    if not path_gpomo and path_pdefind:
        df = df_pdefind
        name = path_pdefind

    df['params'] = df['params'].apply(lambda x: x.split(' ')[-1])

    df2 = df[['method', 'params', 'time']]
    df2.columns = ['method', 'target', 'time']
    df2.reset_index()
    stats = df2.groupby(['method', 'target']).agg({'time': ['median', 'std']}).reset_index()
    stats.columns = ['method', 'target', 'median', 'std']
    stats['model'] = stats['method'] + stats['target'].astype(str)
    # medians = stats.pivot(index='target', columns='method', values='median')
    # std = stats.pivot(index='target', columns='method', values='std')
    #
    # medians.fillna(0)
    # std.fillna(0)

    colors = blues[:len(
        set(['Target time derivative 2', 'Target time derivative 3']).intersection(set(targets_to_plot)))] + reds[:len(
        targets_to_plot)]

    matplotlib.rcParams.update({'font.size': 25})
    # medians.plot(kind="bar", yerr=std, figsize=(15, 10))

    stats.plot.bar(x='model', y='median', yerr='std', legend=False, figsize=(10, 10), color=colors,
                   error_kw=dict(lw=5, capsize=5, capthick=3),
                   width=1.0)
    plt.yscale('log')
    plt.ylabel("Time")
    plt.yticks([1, 10, 30, 60, 180], [ '1s', '10s', '30s', '1m', '3m'])
    plt.xlabel('')
    plt.grid()
    plt.tight_layout()
    plt.savefig(Path.joinpath(results_path, name + '-times.jpeg'))
    plt.show()
    plt.close()

    return df[['method', 'params', 'time']]


if __name__ == '__main__':
    # path_pdefind = 'LorenzAttractor5001_x_Odefind'
    # path_gpomo = 'LorenzAttractor_5001_x'
    # ymax = None
    # targets = [1, 2, 3]
    # time_horizons = np.arange(5, 196, 5)

    # path_pdefind = 'rosselerOdefind'
    # path_gpomo = 'rosseler'
    # ymax = None
    # targets = [1, 2, 3]
    # time_horizons = np.arange(5, 196, 10)

    # path_pdefind = 'rosseler_y_Odefind'
    # path_gpomo = 'rosseler_y'
    # ymax = None
    # targets = [1, 2, 3]
    # time_horizons = np.arange(5, 196, 10)

    path_pdefind = 'oscilator_x_Odefind'
    path_gpomo = 'oscilator_x'
    ymax = 0.5
    targets = [1, 2]
    time_horizons = np.arange(5, 196, 5)

    # path_pdefind = None
    # path_gpomo = 'rte-39'
    # ymax = None
    # targets = [1, 2, 3]
    # time_horizons = np.arange(1,16)

    plot_data = plot_predictions(path_gpomo, time_horizons, path_pdefind, ymax, targets_to_plot=targets)

    times_data = plot_times(path_gpomo, path_pdefind,
                            targets_to_plot= targets)

    if 'rte' in path_gpomo:
        path = '/home/yamila/projects/rte2020/rte-diff-equations/data/results/rte-39/'
        with open(path + 'plot_data_gpomo.pickle', 'wb') as f:
            pickle.dump(plot_data, f)
        times_data.to_csv(path + 'time_data_gpomo.csv', index=False)

