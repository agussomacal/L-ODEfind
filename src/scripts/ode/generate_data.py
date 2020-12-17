from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.special import comb

from src.lib.simulation_manager import Integrator, Oscilator, DifferentialModels
from src.scripts.config import  data_path


def generate_data(data_experiment_name,
                  num_experiments_per_param: int, num_time_steps: int, dt: float,
                  model_class: DifferentialModels = Oscilator,
                  list_model_params: List[Dict] = [{'a': 1, 'b': -0.1, 'c': 0.2, 'd': 2}]) -> List[pd.DataFrame]:
    folder_path = Path.joinpath(data_path, data_experiment_name)
    folder_path.mkdir(exist_ok=True, parents=True)
    data_params = []
    model_param_names = list(list_model_params[0].keys())
    np.random.seed(10)
    for j, model_params in enumerate(list_model_params):
        for i in range(num_experiments_per_param):
            model = model_class(**model_params)
            integrator = Integrator(model)
            Xinit = np.random.normal(size=len(model.var_names))
            ode_solution = integrator.integrate_solver(
                Xinit=Xinit,
                time_steps=num_time_steps,
                integration_dt=dt
            )
            ode_solution.index.name = 'time'
            ode_solution.reset_index(inplace=True)
            filename = 'solution_params_{}_init_cond_{}'.format(j, i)
            data_params.append([filename] + [model_params[k] for k in model_param_names])
            ode_solution.to_csv(path_or_buf='{}/{}.csv'.format(folder_path, filename), index=False)
    pd.DataFrame(data_params, columns=['filename'] + model_param_names).to_csv('{}/eq_params.csv'.format(folder_path),
                                                                               index=False)
    return folder_path


def regOrd(nVar: int, dMax: int) -> pd.DataFrame:
    """
    Python traduction of regOrd function from Gpomo (R)
    """
    pExpo = np.arange(dMax + 1).reshape((-1, 1))
    for i in range(nVar - 1):
        pExpotmp = pExpo
        tmpEx = np.reshape(pExpotmp[:, 0] * 0, (-1, 1))
        pExpo = np.concatenate([pExpotmp, tmpEx], axis=1)

        for j in range(dMax):
            tmpEx = np.reshape(pExpotmp[:, 0] * 0 + (j + 1), (-1, 1))
            tmpEx2 = np.concatenate([pExpotmp, tmpEx], axis=1)
            pExpo = np.concatenate([pExpo, tmpEx2], axis=0)

    ltdMax = np.sum(pExpo, axis=1) <= dMax
    pExpo = np.transpose(pExpo[ltdMax, :])[range(nVar - 1, -1, -1), :]
    noms = ["X{}".format(i + 1) for i in range(nVar)]

    return pd.DataFrame(pExpo, index=noms)


def poLabs(nVar: int, dMax: int, Xnote: (str, List) = "X") -> List[str]:
    """
    Python traduction of poLabs function from Gpomo (R)
    """

    if (len(Xnote) == 1):
        Xnote = [Xnote + str(j + 1) for j in range(nVar)]
    elif len(Xnote) != nVar:
        raise Exception("Xnote should be either one single character or a  nVar len vector of character")

    pMax = int(comb(nVar + dMax, dMax))
    pExpo = regOrd(nVar, dMax)
    lbls = [""] * pMax
    lbls[0] = "ct"
    for i in range(1, pMax):
        for j in range(nVar):
            if pExpo.iloc[j, i] != 0:
                if pExpo.iloc[j, i] == 1:
                    lbls[i] = lbls[i] + Xnote[j] + " "
                if pExpo.iloc[j, i] > 1:
                    lbls[i] = str(lbls[i]) + str(Xnote[j]) + "^" + str(pExpo.iloc[j, i]) + " "

    return lbls


if __name__ == '__main__':
    folder_path = generate_data(
        num_experiments_per_param=20,
        num_time_steps=5000,
        dt=0.01,

        # data_experiment_name='LorenzAttractor',
        # model_class=LorenzAttractor,
        # list_model_params=[{'sigma': 10, 'rho': 28, 'beta': 8.0 / 3}]

        # data_experiment_name='rosseler',
        # model_class=RoselerAttractor,
        # list_model_params=[{'a': 0.52, 'b': 2, 'c': 4}]

        data_experiment_name='oscilator',
        model_class=Oscilator,
        list_model_params=[{'a': 0.1, 'b': -1, 'c': 1, 'd': 0}]

    )

    # TODO: decide to include or not the data plots
    # solution = pd.read_csv('{}/solution_params_{}_init_cond_{}.csv'.format(folder_path, 0, 0), index_col=0)
    #
    # solution.plot(y='x')
    # solution.plot(y='y')
    # solution.plot(x='x', y='y', marker='.')
    # plt.show()

    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # solution.plot( ax=ax[0, 0])
    # solution.plot(x='x', y='y', marker='.', ax=ax[0, 1])
    # solution.plot(x='x', y='z', marker='.', ax=ax[1, 0])
    # solution.plot(x='y', y='z', marker='.', ax=ax[1, 1])
    # plt.show()

    #
    # osi = Oscilator(**{'a': 1, 'b': -0.1, 'c': 0.2, 'd': 2})
    # print(osi.coeff('dx', 'x'))
    # print(osi.coeff('dx', 'z'))
