from os import listdir
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from src.scripts.config import data_path, results_path
from src.lib.skodefind import SKIntegrate
from src.scripts.utils.metrics import smape_loss


def get_info_from_filename(filename):
    """
    "solution-gpomo_coefs-vars_x_y_z-dmax_1-poly_2-params_0-init_cond_0-steps_5000.csv"
    "solution-pdefind_coefs-vars_x_y_z-dmax_1-poly_2-params_0-init_cond_0-steps_0.csv"

    :param filename:
    :return:
    """
    filename = filename.split('.')[0]
    info = filename.split('-')[1:]
    var_names = info[1].split('_')[1:]
    n_vars = len(var_names)
    d_max = int(info[2].split('_')[-1])
    poly_max = int(info[3].split('_')[-1])
    params_set = int(info[4].split('_')[-1])
    init_cond = int(info[5].split('_')[-1])
    steps = int(info[6].split('_')[-1])

    return var_names, n_vars, d_max, poly_max, params_set, init_cond, steps


def translate_coeff_gpomo2odefind(df: pd.DataFrame, var_names, n_vars, d_max):
    if n_vars != 1:
        raise Exception('Not implemented.')

    var_name = var_names[0]
    var_name_in_gpomo = var_name + str(1)
    var_name_in_pdefind = '{}(t)'.format(var_name.lower())

    # replace rows
    df.index = [i.strip() for i in df.index]
    d_row = dict()
    d_row['ct'] = '1.00000000000000'
    d_row['^'] = '**'
    d_row[' '] = '*'
    for k, v in d_row.items():
        df.index = [i.replace(k, v) for i in df.index]

    d_row = dict()
    d_row[var_name_in_gpomo] = var_name_in_pdefind
    d_row.update({
        var_name + str(der):
            'Derivative({},{})'.format(var_name_in_pdefind, '(t,{})'.format(der - 1) if der > 2 else 't')
        for der in range(2, d_max + 1)
    })

    for k, v in d_row.items():
        df.index = [i.replace(k, v) for i in df.index]

    df.index = ["1.0*" + i if 'Derivative' in i else i for i in df.index]

    d_col = dict()
    d_col.update({
        var_name + str(der):
            '1.0*Derivative({},{})'.format(var_name_in_pdefind, '(t,{})'.format(der) if der > 1 else 't')
        for der in range(1, d_max + 1)
    })
    df.columns = [d_col[i.strip()] for i in df.columns]
    return df


def integrate_predict_smape(original_data: str, results_folder: str, time_horizons, testsize=200,
                            one=False, verbose=False) -> pd.DataFrame:
    path_data = Path.joinpath(data_path, original_data)
    path_results = Path.joinpath(results_path, results_folder)

    smapes = []
    for f in listdir(str(path_results)):
        if 'solution' == f[:len('solution')]:
            if verbose:
                print(f)
            # get params of run
            var_names, n_vars, d_max, poly_max, params_set, init_cond, steps = get_info_from_filename(filename=f)
            # get true solution
            true_solution = pd.read_csv(
                Path.joinpath(path_data, 'solution_params_{}_init_cond_{}.csv'.format(params_set, init_cond)),
                index_col=0)
            true_solution_train = true_solution[var_names].iloc[:-testsize, :]
            true_solution_test = true_solution[var_names].iloc[-testsize:, :]
            if len(var_names) == 1:
                true_solution_test = (true_solution_test[var_names[0]]).values

            adj_coeffs = pd.read_csv(Path.joinpath(Path(path_results), f), index_col=0)
            adj_coeffs.index = [i.strip() for i in adj_coeffs.index]
            adj_coeffs.columns = [i.strip() for i in adj_coeffs.columns]

            if 'Odefind' in results_folder:
                model_name = 'odefind'
            else:
                model_name = 'gpomo'
                # only consider last column and translate names
                adj_coeffs = translate_coeff_gpomo2odefind(adj_coeffs.iloc[:, [-1]], [s.upper() for s in var_names],
                                                           n_vars, d_max)

            # keep last column of adj_coeffs and integrate
            skintegrate = SKIntegrate(target_derivative_order=d_max,
                                      max_derivative_order=d_max - 1,
                                      max_polynomial_order=poly_max)
            if verbose:
                print('Doing fit and predictions')
            skintegrate.fit(true_solution_train)
            # define the coefficient names in the correct order
            skintegrate.set_coefs(
                coefs=adj_coeffs.loc[skintegrate.data_manager.get_X_dframe().columns, :].iloc[:, -n_vars:].T)

            predictions = (skintegrate.predict(np.arange(1, testsize + 1))).values

            for time_h in time_horizons:
                if one:
                    a = predictions[time_h]
                    t = true_solution_test[time_h]
                else:
                    a = predictions[:time_h]
                    t = true_solution_test[:time_h]
                smape = smape_loss(a, t)
                smapes.append([model_name, f, smape, time_h])
                if time_h == max(time_horizons):
                    Path.joinpath(path_results, 'plots').mkdir(exist_ok=True, parents=True)
                    plt.plot(true_solution_test, label='true')
                    plt.plot(predictions, label='pred')
                    plt.title('SMAPE {:.3f}'.format(smape))
                    plt.xlabel('Time horizon')
                    plt.legend()
                    plt.savefig('{}/plots/{}.png'.format(path_results, f))
                    plt.close()

    df = pd.DataFrame(smapes, columns=['method', 'model', 'smape', 'time_horizon'])
    df.to_csv('{}/summary.csv'.format(path_results))
    return df


if __name__ == '__main__':
    # model_data = 'LorenzAttractor5001'
    # model_results = 'LorenzAttractor5001_x_Odefind'
    # model_results = 'LorenzAttractor5001_x'
    #
    model_data = 'oscilator'
    model_results = 'oscilator_x_Odefind'
    # model_results = 'oscilator_x'

    # model_data = 'rosseler'
    # model_results = 'rosseler_x_Odefind'
    # model_results = 'rosseler_x'

    # model_data = 'rte-39'
    # model_results = 'rte-39_x'
    # model_results ='rte-39_x_Odefind'

    # model_data = 'rosseler'
    # model_results = 'rosseler_y'
    # model_results = 'rosseler_y_Odefind'

    df = integrate_predict_smape(original_data=model_data, results_folder=model_results,
                                 time_horizons=np.arange(1, 200, dtype=int),
                                 testsize=200, one=False, verbose=False)
