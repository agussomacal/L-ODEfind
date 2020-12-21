import time
from os import listdir
from pathlib import Path
from typing import List

import pandas as pd

from src.scripts.config import data_path, results_path
from src.lib.skodefind import SkODEFind


def fit_and_save_coeffs(model: str, out: str, targets: List[int], maxpolys: List[int], obs_vars: List[str],
                        testsize=200):
    path_data = Path.joinpath(data_path, model)
    path_results = Path.joinpath(results_path, out + '_' + '_'.join(obs_vars) + '_Odefind')
    Path(path_results).mkdir(parents=True, exist_ok=True)

    times = []
    for f in listdir(str(path_data)):
        for target in targets:
            maxd = target - 1
            for maxpoly in maxpolys:
                if 'solution' in f:
                    df = pd.read_csv(Path.joinpath(path_data, f), index_col=0)[obs_vars]
                    true_solution_train = df.iloc[:-testsize, :]

                    odefind = SkODEFind(
                        target_derivative_order=target,
                        max_derivative_order=maxd,
                        max_polynomial_order=maxpoly,
                        rational=False,
                        with_mean=True, with_std=True, alphas=150, max_iter=10000, cv=20,
                        use_lasso=True,
                    )
                    s = time.time()
                    odefind.fit(true_solution_train)
                    e = time.time()

                    params = f.split('params_')[1].split('_')[0]
                    init_cond = f.split('init_cond_')[1].split('.')[0]
                    filename = "solution-pdefind_coefs-vars_{}-dmax_{}-poly_{}-params_{}-init_cond_{}-steps_0.csv".format(
                        '_'.join(obs_vars), target, maxpoly, params, init_cond)

                    times.append([filename, e - s])
                    coeffs = odefind.coefs_.transpose()

                    coeffs.to_csv(Path.joinpath(path_results, filename))

    pd.DataFrame(times, columns=['model', 'time']).to_csv(Path.joinpath(path_results, 'times.csv'))


if __name__ == '__main__':
    # obs_vars = ['x', 'y', 'z']
    # model = 'LorenzAttractor'
    # out = 'LorenzAttractor'
    # targets = [1]
    # maxpolys = [2]

    # obs_vars = ['x']
    # model = 'LorenzAttractor'
    # out = 'LorenzAttractor'
    # targets = [2, 3]
    # maxpolys = [3]

    obs_vars = ['x']
    model = 'oscilator'
    out = 'oscilator'
    targets = [2, 3]
    maxpolys = [3]

    # obs_vars = ['x']
    # model = 'rosseler'
    # out = 'rosseler'
    # targets = [2, 3]
    # maxpolys = [3]

    # obs_vars = ['y']
    # model = 'rosseler'
    # out = 'rosseler'
    # targets = [2, 3]
    # maxpolys = [3]

    # obs_vars = ['x']
    # model = 'rte-39'
    # out = 'rte-39'
    # targets = [2, 3]
    # maxpolys = [3]


    fit_and_save_coeffs(model, out, targets=targets, maxpolys=maxpolys, obs_vars=obs_vars, testsize=200)
