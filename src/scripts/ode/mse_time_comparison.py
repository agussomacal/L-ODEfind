##
from pathlib import Path
from typing import List
import pandas as pd
from src.scripts.ode.generate_data import poLabs
from src.scripts.ode.integrate_predict import get_info_from_filename
from src.scripts.config import data_path, results_path
from src.lib.simulation_manager import LorenzAttractor
from os import listdir
import matplotlib.pyplot as plt


def xyz(nvar: int) -> List[str]:
    v = ['x', 'y', 'z']
    if nvar <= 3:
        return v[:nvar]
    else:
        return v + [f'x{i + 1}' for i in range(3, nvar)]


def to_python_labels(labs: List[str], nvar: int, dmax: int) -> List[str]:
    python_labels = poLabs(nvar, dmax, Xnote=xyz(nvar))
    python_labels = [a.replace(' ', '') for a in python_labels]
    gpomo_labels = poLabs(nvar, dmax)
    return [python_labels[gpomo_labels.index(l)] for l in labs]


def coeff_matrix(model, nvars: int, dmax: int) -> pd.DataFrame:
    poly_terms = poLabs(nvars, dmax)
    variables = poLabs(nvars, 1)[1:]
    variables.reverse()
    gpomo_m = pd.DataFrame(index=poly_terms, columns=variables)

    for v, v_python in zip(variables, to_python_labels(variables, nvars, 1)):
        for p, p_python in zip(poly_terms, to_python_labels(poly_terms, nvars, dmax)):
            gpomo_m.loc[p, v] = model.d_to_coeff['d' + v_python][p_python]
    return gpomo_m


def translate_odefind_coeff(original_data: str, df: pd.DataFrame, var_name='X'):
    d2 = {'*': ' '}
    d3 = {}

    if 'Lorenz' in original_data:
        d = {
            '1.00000000000000': 'ct',
            'x(t)': var_name + '1',
            'y(t)': var_name + '2',
            'z(t)': var_name + '3',
            '**': '^'
        }
        d_cols = {
            '1.0*Derivative(x(t),t)': var_name + '1',
            '1.0*Derivative(y(t),t)': var_name + '2',
            '1.0*Derivative(z(t),t)': var_name + '3'
        }
    elif 'rosseler' in original_data:
        d = {
            '1.00000000000000': 'ct',
            'Derivative(y(t),t)': var_name + '2',
            'Derivative(y(t),(t,2))': var_name + '3',
            '**': '^'
        }
        d3 = {'y(t)': var_name + '1',
              '1.0 ': ''}

        d_cols = {
            '1.0*Derivative(y(t),(t,3))': var_name + '3'
        }

    elif 'oscilator' in original_data:
        d = {
            '1.00000000000000': 'ct',
            'Derivative(x(t),t)': var_name + '2',
            'Derivative(x(t),(x,2))': var_name + '3',
            '**': '^'
        }
        d3 = {'x(t)': var_name + '1',
              '1.0 ': ''}

        d_cols = {
            '1.0*Derivative(x(t),(t,2))': var_name + '2'
        }

    df.columns = [d_cols[i] for i in df.columns]
    for k, v in d.items():
        df.index = [i.replace(k, v) for i in df.index]
    for k, v in d2.items():
        df.index = [i.replace(k, v) for i in df.index]
    for k, v in d3.items():
        df.index = [i.replace(k, v) for i in df.index]
    return df


def add_times(df: pd.DataFrame, path_results):
    times = pd.read_csv(Path.joinpath(path_results, 'times.csv'), index_col=0)
    if 'Odefind' not in str(path_results):
        # gpomo times are saved in a different way
        times = times.transpose()
        times.columns = ['time']
        times['model'] = [t.replace('.', '-') for t in times.index]
        times['model'] = [t.replace('-csv', '.csv') for t in times['model']]
    df_times = df.join(times.set_index('model'), on='model')
    return df_times


def convert_format(f: str):
    var_names, n_vars, d_max, poly_max, params_set, init_cond, steps = get_info_from_filename(f)
    return 'solution_params_{}_init_cond_{}'.format(params_set, init_cond, steps)


def nvars_polymax(f: str):
    var_names, n_vars, d_max, poly_max, params_set, init_cond, steps = get_info_from_filename(f)
    return n_vars, poly_max


def rosseler_true_coeffs_y(a=0.52, b=2.0, c=4.0):
    return {'ct': -b,
            'Y1': -a * c,
            'Y2': c - a + a * c,
            'Y3': a + 1 - c,
            'Y1^2': - a ** 2,
            'Y1 Y2': 2 * a + a ** 2,
            'Y1 Y3': 1 - a,
            'Y2^2': -(1 + a),
            'Y2 Y3': 1
            }


def oscilator_true_coeffs(a, b, c, d):
    return {
        'ct': 0,
        'X1': -a * d + b * c,
        'X2': a + d,
        'X3': 0,
        'X1^2': 0,
        'X1 X2': 0,
        'X1 X3': 0,
        'X2^2': 0,
        'X2 X3': 0
    }


def compare_coeffs(original_data: str, results_folder: str, var_name: str, d: int):
    path_data = Path.joinpath(data_path, original_data)
    path_results = Path.joinpath(results_path, results_folder)

    model_params = pd.read_csv(Path.joinpath(path_data, 'eq_params.csv'))

    mses = []
    all_adj_coeffs = []
    ix_coeff = True

    for f in listdir(path_results):
        if 'solution' in f and 'dmax_{}'.format(d) in f:
            params_name = convert_format(f)
            nvars, poly_max = nvars_polymax(f)

            params = model_params[model_params['filename'] == params_name]

            adj_coeffs = pd.read_csv(Path.joinpath(path_results, f), index_col=0)
            adj_coeffs.index = [i.strip() for i in adj_coeffs.index]
            adj_coeffs.columns = [i.strip() for i in adj_coeffs.columns]
            if 'Odefind' in results_folder:
                adj_coeffs = translate_odefind_coeff(original_data, adj_coeffs, var_name)
                model_name = 'odefind'
            else:
                model_name = 'gpomo'

            if 'Lorenz' in original_data:
                data_gen = LorenzAttractor(sigma=params['sigma'].values[0], rho=params['rho'].values[0],
                                           beta=params['beta'].values[0])
                true_coeff = coeff_matrix(data_gen, nvars=nvars, dmax=poly_max)
                true_coeff.index = [i.strip() for i in true_coeff.index]
                true_coeff.columns = [i.strip() for i in true_coeff.columns]

            elif 'rosseler' in original_data:
                true_coeff_dict = rosseler_true_coeffs_y(params['a'], params['b'], params['c'])
                true_coeff = pd.DataFrame([float(true_coeff_dict.get(i, 0)) for i in adj_coeffs.index],
                                          index=adj_coeffs.index,
                                          columns=[adj_coeffs.columns[-1]])
                adj_coeffs = adj_coeffs[[adj_coeffs.columns[-1]]]
            elif 'oscilator' in original_data:
                true_coeff_dict = oscilator_true_coeffs(params['a'], params['b'], params['c'], params['d'])
                true_coeff = pd.DataFrame([float(true_coeff_dict.get(i, 0)) for i in adj_coeffs.index],
                                          index=adj_coeffs.index,
                                          columns=[adj_coeffs.columns[-1]])
                adj_coeffs = adj_coeffs[[adj_coeffs.columns[-1]]]

            if 'rosseler' in original_data or 'oscilator' in original_data:
                if ix_coeff:
                    all_adj_coeffs.append(true_coeff)
                    ix_coeff = False
                c = adj_coeffs.copy()
                c.columns = [f]
                all_adj_coeffs.append(c)

            mse = ((true_coeff - adj_coeffs) ** 2).mean().mean()
            mses.append([model_name, f, mse])
    df = pd.DataFrame(mses, columns=['method', 'model', 'mse'])

    if 'rosseler' in original_data or 'oscilator' in original_data:
        all_coeffs = pd.concat(all_adj_coeffs, axis=1)
    else:
        all_coeffs = pd.DataFrame()

    return add_times(df, path_results), all_coeffs


def plot_mse_time(df: pd.DataFrame, model_results: str):
    plt.rcParams.update({'font.size': 20})

    groups = df.groupby('method')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)

    markers = {'gpomo': '^', 'odefind': 'o'}
    colors = {'gpomo': 'blue', 'odefind': 'orange'}
    for name, group in groups:
        ax.plot(group.time, group.mse, marker=markers[name], linestyle='', ms=7, label=name, alpha=0.7, c=colors[name])
        if 'Lorenz' in model_results:
            ax.set_yscale('log')
        ax.set_xscale('log')

    if 'Lorenz' in model_results:
        ax.set_xticks([1, 10, 30, 60, 600, 1800, 3600, 3600 * 3, 3600 * 6])
        ax.set_xticklabels(['1s', '10s', '30s', '1m', '10m', '30m', '1h', '3h', '6h'])
    else:
        ax.set_xticks([1, 10, 30, 60, 360])
        ax.set_xticklabels(['1s', '10s', '30s', '1m', '3m'])

    ax.legend(['GPoMo', 'L-ODEfind' ])
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.grid()
    plt.tight_layout()
    plt.savefig(Path.joinpath(results_path, f'{model_results}_mse.png'))
    # plt.show()


def plot_coeffs(coeffs: pd.DataFrame, model_results: str, var_name: str, d: int):
    coeffs_gpomo = coeffs[[i for i in coeffs.columns if 'gpomo' in i]]
    coeffs_pdefind = coeffs[[i for i in coeffs.columns if 'pdefind' in i]]
    true_coeffs = coeffs[[var_name + str(d)]]
    true_coeffs.columns = ['true']

    if 'rosseler' in model_results:
        cuadratic_var = ['ct', 'Y3', 'Y3^2', 'Y2', 'Y2 Y3', 'Y2^2', 'Y1', 'Y1 Y3', 'Y1 Y2', 'Y1^2']
        true_coeffs = true_coeffs.loc[cuadratic_var]
        coeffs_pdefind = coeffs_pdefind.loc[cuadratic_var]
        coeffs_gpomo = coeffs_gpomo.loc[cuadratic_var]

    # change labels Y3 = Y'', Y2=Y', Y1 = Y
    d_labels = {var_name + '3': var_name + '\'\'', var_name + '2': var_name + '\'', var_name + '1': var_name}
    for k, v in d_labels.items():
        true_coeffs.index = [a.replace(k, v) for a in true_coeffs.index]
        coeffs_gpomo.index = [a.replace(k, v) for a in coeffs_gpomo.index]
        coeffs_pdefind.index = [a.replace(k, v) for a in coeffs_pdefind.index]

    m_gpomo = pd.DataFrame(coeffs_gpomo.median(axis=1), columns=['gpomo'])
    m_pdefind = pd.DataFrame(coeffs_pdefind.median(axis=1), columns=['odefind'])
    medians = pd.concat([true_coeffs, m_gpomo, m_pdefind], axis=1)

    std_gpomo = pd.DataFrame(coeffs_gpomo.std(axis=1), columns=['gpomo'])
    std_pdefind = pd.DataFrame(coeffs_pdefind.std(axis=1), columns=['odefind'])
    true_std = pd.DataFrame(0, index=std_pdefind.index, columns=['true'])
    std = pd.concat([true_std, std_gpomo, std_pdefind], axis=1)

    medians.plot(kind="bar", yerr=std, figsize=(15, 10), color=['green', 'blue', 'red'])
    plt.legend(['True', 'GPoMo3', 'L-ODEfind3'])
    plt.ylabel("Coefficient value")
    plt.savefig(Path.joinpath(results_path, model_results + '_coeffs.jpeg'))


if __name__ == '__main__':
    model_data = 'LorenzAttractor'
    model_results = 'LorenzAttractor' + '_x_y_z'
    var_name = 'X'
    d = 1

    # model_data = 'oscilator'
    # model_results = 'oscilator'
    # var_name = 'X'
    # d = 2

    # model_data = 'rosseler'
    # model_results = 'rosseler_y'
    # var_name = 'Y'
    # d = 3

    df_mse_gpomo, coeffs_gpomo = compare_coeffs(original_data=model_data, results_folder=model_results,
                                                var_name=var_name,
                                                d=d)
    df_mse_odefind, coeffs_odefind = compare_coeffs(original_data=model_data, results_folder=model_results + '_Odefind',
                                                    var_name=var_name, d=d)

    plot_mse_time(pd.concat([df_mse_gpomo, df_mse_odefind]), model_results)

    if 'rosseler' in model_data or 'oscilator' in model_data:
        coeffs = pd.concat([coeffs_gpomo, coeffs_odefind], axis=1)
        coeffs = coeffs.loc[:, ~coeffs.columns.duplicated()]
        plot_coeffs(coeffs, model_results, var_name, d)

# instructions for plot MSE vs time, gpomo vs pdefind
# fit gpomo (gpomo_all_var_observed.R)
# fit pdefind (fit_odefind.py when observed variables are x,y,z for the lorenz attractor)
# make plot using mse_time_comparison.py

# Same idea for Lorenz and Rossler, but Lorenz is fully observed and Rossler(observed x). The code is full of "ifs" :(
