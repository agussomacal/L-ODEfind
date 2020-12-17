import copy
import itertools
from functools import partial
from typing import Callable, List, Dict

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation

import src.scripts.config as config
from src.lib.operators import DataSplit, PolyD, Poly, D
from src.lib.pdefind import DataManager
from src.lib.regressors import TimeReg, Trigonometric
from src.lib.variables import Variable, Domain, Field
from src.scripts.spatial.experiments_utils import ExperimentSetting
from src.scripts.utils.utils import check_create_path, savefig2
from src.scripts.utils.utils import timeit


def gif(data, t, x, path, fps=30):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    # Query the figure's on-screen size and DPI. Note that when saving the figure to
    # a file, we need to provide a DPI for that separately.
    print('fig size: {0} DPI, size in inches {1}'.format(
        fig.get_dpi(), fig.get_size_inches()))

    # Plot a scatter that persists (isn't redrawn) and the initial line.
    ax.plot(x, data[0, :], 'r-', linewidth=2)
    line, = ax.plot(x, data[0, :], 'b-', linewidth=2)
    ax.set_ylim((data.min(), data.max()))

    def update(i):
        title = 'Time {:.2f}'.format(t[i])
        ax.set_title(title)

        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        line.set_ydata(data[i, :])
        return line, ax

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    dt = t[1] - t[0]  # in seconds
    with timeit('Animation: '):
        anim = FuncAnimation(fig, update, frames=np.arange(0, len(t), 1.0 / fps / dt, dtype=int),
                             interval=1.0 / fps * 1000)
        anim.save('{}/StressedString.gif'.format(path), dpi=80, writer='imagemagick')


def get_x_operator_func(derivative_order, polynomial_order, rational=False):
    def x_operator(field, regressors):
        new_field = copy.deepcopy(field)
        new_field = PolyD(derivative_order) * new_field
        new_field.append(regressors)
        # if rational:
        #     new_field.append(new_field.__rtruediv__(1.0))
        new_field = (Poly(polynomial_order) * new_field)
        new_field = Field(
            [var for var in new_field.data if not np.allclose(var.data, 1) or '1.000' in var.get_full_name()])
        return new_field

    return x_operator


def get_y_operator_func(derivative_order):
    def y_operator(field):
        return D(derivative_order['t'] + 1, "t") * field

    return y_operator


class StressedStringExperimentSetting(ExperimentSetting):
    """
    https://people.sc.fsu.edu/~jburkardt/cpp_src/fd1d_wave/fd1d_wave.html
    """

    def __init__(self, experiment_name, subfolders=()):
        ExperimentSetting.__init__(self, experiment_name=experiment_name, subfolders=subfolders)
        print('============================')
        print(self.experiment_name)

        self.t = []
        self.data = []
        self.dt = None
        self.t0 = None
        self.T = None
        self.dx = None
        self.x0 = None
        self.L = None

        self.t = None
        self.x = None

        self.m = None
        self.k = None

        self.stress_mode = None

        self.initial_perturbation_function = None

    def set_underlying_model(self, dt, time_steps, space_steps, initial_perturbation_function: Callable = 'sin',
                             L=1, mu=1, k=0.5, m=1, A=0, g=10, stress_mode='stress_linear',
                             boundary_conditions=(0, 0), dirischlet=True, dogif=False):
        """
            # ---------- simulating data ----------
            :param dt:
            :param time_steps:
            :param odespy_method:
            :return:
            """
        self.stress_mode = stress_mode

        self.dt = dt
        self.t0 = 0
        self.T = time_steps * self.dt
        self.dx = float(L) / space_steps
        self.x0 = 0
        self.L = L

        self.t = np.linspace(self.t0, self.T, time_steps)
        self.x = np.linspace(self.x0, self.L, space_steps)

        w = 2 * np.pi * np.sqrt(k / m)
        self.m = m
        self.k = k

        # mu=100, k=0.6, m=1, A=0, g=100
        if self.stress_mode == 'stress_linear':
            def stress(t):
                return (m * g + k * A) / mu * (1 - t / (self.T - self.t0) / 2)
        elif self.stress_mode == 'stress_trigonometric':
            def stress(t):
                return (m * g + k * A) / mu * np.cos(w * t) / 2
        elif self.stress_mode == 'stress_constant':
            def stress(t):
                return (m * g + k * A) / mu  # np.cos(w * t)
        else:
            raise Exception('Not implemented')
        if initial_perturbation_function == 'sin':
            self.initial_perturbation_function = lambda x: np.sin(np.pi / self.L * x)  # exactly half wave
        elif isinstance(initial_perturbation_function, Callable):
            self.initial_perturbation_function = initial_perturbation_function
        else:
            raise Exception('function not implemented')

        self.data = np.zeros((time_steps, space_steps))
        if dirischlet:
            self.data[:, 0] = boundary_conditions[0]
            self.data[:, -1] = boundary_conditions[1]
        else:
            raise Exception('not Dirichlet conditions not implemented')
        self.data[0, :] = self.initial_perturbation_function(self.x)
        dfrac = self.dt ** 2 / self.dx ** 2
        c2 = stress(self.t[1])
        cval = c2 * dfrac
        self.data[1, 1:-1] = 0.5 * cval * self.data[0, 2:] + \
                             (1 - cval) * self.data[0, 1:-1] + \
                             0.5 * cval * self.data[0, :-2]
        # + self.dt * 0 initially velocity is zero because it begins in the maximum amplitude.

        for ti in range(1, time_steps - 1):
            c2 = stress(self.t[ti])
            cval = c2 * dfrac
            self.data[ti + 1, 1:-1] = cval * self.data[ti, 2:] + \
                                      2 * (1 - cval) * self.data[ti, 1:-1] + \
                                      cval * self.data[ti, :-2] + \
                                      - self.data[ti - 1, 1:-1]

        if dogif:
            gif(self.data, self.t, self.x, path=self.plots_path)

    def plot_initial_condition(self, ax, color):
        # Plot a scatter that persists (isn't redrawn) and the initial line.
        ax.plot(self.x, self.data[0, :], '-', c=color, linewidth=2, label='init_condition')
        ax.set_ylim((np.min(self.data), np.max(self.data)))

    def get_domain(self):
        return Domain(
            lower_limits_dict={"t": np.min(self.t), "x": self.x0},
            upper_limits_dict={"t": np.max(self.t), "x": self.L},
            step_width_dict={"t": self.dt, "x": self.dx})

    def get_variables(self):
        domain = self.get_domain()
        return [Variable(self.data, domain, domain2axis={"t": 0, "x": 1}, variable_name="u")]

    # ========================= explore polynomial and derivatives =========================
    def explore_eqdiff_fitting(self, derivatives2explore, poly2explore):

        there_are_regs = len(self.regressors_builders) > 0
        rsquares = {k: [] for k in ['r2', 'poly_degree'] + list(derivatives2explore.keys()) +
                    (['regressor'] if there_are_regs else [])}

        for regressors_on in [False] + ([True] if there_are_regs else []):
            for poly_degree in poly2explore:
                print("\n---------------------")
                print("Polynomial degree: {}".format(poly_degree))
                print("Derivative order:", end='')

                for derivatives in itertools.product(
                        *[list(range(der if ax == 't' else der + 1)) for ax, der in derivatives2explore.items()]):
                    derivative_depth = {k: v for k, v in zip(derivatives2explore.keys(), derivatives)}
                    print(" {}".format(derivative_depth), end='')
                    data_manager = DataManager()
                    data_manager.add_variables(self.get_variables())

                    if there_are_regs:
                        rsquares['regressor'].append(regressors_on)
                    if regressors_on:
                        data_manager.add_regressors(self.get_regressors())
                    data_manager.set_domain()
                    data_manager.set_X_operator(
                        get_x_operator_func(derivative_depth, poly_degree, data_manager.regressors))
                    data_manager.set_y_operator(get_y_operator_func(derivative_depth))

                    # try:
                    pde_finder = self.fit_eqdifff(data_manager)

                    for k, v in derivative_depth.items():
                        rsquares[k].append(v)
                    rsquares['poly_degree'].append(poly_degree)
                    rsquares['r2'].append(np.mean(self.get_rsquare_of_eqdiff_fit(pde_finder, data_manager).values))

                    # ========== plots coefficients ==========
                    subname = '_y{}_der_x{}_pol{}_reg{}'.format(
                        derivative_depth['t'] + 1,
                        '_'.join([k + str(v) for k, v in derivative_depth.items()]),
                        poly_degree,
                        regressors_on
                    )

                    # ----- plot coefficients -----
                    path = check_create_path(self.plots_path, 'PDEFinder_coefs')
                    with savefig2('{}/PDEFinder_coefs_{}.png'.format(path, subname)):
                        self.plot_coefficients(pdefind=pde_finder)

                    # ----- plot fit versus real -----
                    path = check_create_path(self.plots_path, 'fit_vs_real')
                    with savefig2('{}/fit_vs_real_{}.png'.format(path, subname)):
                        self.plot_fitted_vs_real(pde_finder, data_manager)

                    df_rsquares = pd.DataFrame.from_dict(rsquares)
                    df_rsquares = df_rsquares.sort_values(by='r2', ascending=False)
                    df_rsquares.to_csv('{}/rsquares_eqfit.csv'.format(self.plots_path))

    def plot_heatmaps(self):
        # ---------- plot heatmap of rsquares ----------
        there_are_regs = len(self.regressors_builders) > 0
        df_rsquares = pd.read_csv('{}/rsquares_eqfit.csv'.format(self.plots_path), index_col=0)
        with savefig2('{}/rsquares_eqfit_{}.eps'.format(self.plots_path, self.experiment_name), close=False):
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, df_rsquares.shape[0] / 2),
                                   gridspec_kw={'width_ratios': [1, 3]})
            # df_rsquares = pd.DataFrame.from_dict(rsquares)
            df_rsquares = df_rsquares.sort_values(by='r2', ascending=False)
            df_rsquares_temp = df_rsquares.copy()
            df_rsquares_temp['t'] += 1
            df_rsquares_temp.rename(columns={'poly_degree': 'Poly.\n degree',
                                             'x': 'Spat.\n Deriv.',
                                             't': 'Temp.\n Deriv.'}, inplace=True)
            columns = ['Poly.\n degree', 'Spat.\n Deriv.', 'Temp.\n Deriv.'] + (['regressor'] if there_are_regs else [])
            sns.heatmap(df_rsquares_temp[columns].astype(int), annot=True, ax=ax[0], cbar=False, cmap='viridis')
            df_rsquares_temp[df_rsquares_temp < 0] = 0
            df_rsquares_temp.sort_values(by='r2', ascending=True).plot(y='r2', kind='barh', ax=ax[1], colormap='summer')
            for i in range(2):
                ax[i].tick_params(axis='both', which='major', labelsize=12)
                ax[i].tick_params(axis='both', which='minor', labelsize=9)
                ax[i].set(yticklabels=[])
                ax[i].tick_params(left=False)
            plt.title('R²', fontsize=18)
            plt.tight_layout()
        plt.show()
        plt.close()


def cuartic(x):
    return -20 * x * (x - 0.25) * (x - 1) * (x - 1.01)


def trigonometric(x, L=1):
    return np.sin(np.pi / L * x)


def spatial_experiment(experiment_name: str, type_of_stress: str, init_cond: str,
                       derivatives2explore: Dict[str, int], poly2explore: List[int],
                       regressors2use: List[str] = ()):
    assert type_of_stress in ['trigonometric', 'linear', 'constant'], "type_of_stress should be linear or constant"
    assert all([polyd >= 1 for polyd in poly2explore]), 'poly2explore must be 1 or more.'

    L = 1
    color = 'b'
    subfolders = [experiment_name]
    stress_mode = 'stress_' + type_of_stress

    # ---------- initial conditions ----------
    if init_cond == 'trigonometric':
        initial_perturbation_function = partial(trigonometric, L=L)
    elif init_cond == 'cuartic':
        initial_perturbation_function = cuartic
    else:
        raise Exception('Initial perturbation {} not implemented'.format(init_cond))

    # ---------- define spatial experiment ----------
    experiment = StressedStringExperimentSetting(
        experiment_name='String_{}_{}'.format(stress_mode, init_cond),
        subfolders=subfolders)
    experiment.set_underlying_model(dt=0.01, time_steps=1000, space_steps=100,
                                    initial_perturbation_function=initial_perturbation_function,
                                    L=L, mu=100, k=0.6, m=1, A=0, g=100, stress_mode=stress_mode,
                                    boundary_conditions=(0, 0), dirischlet=True, dogif=False)
    experiment.set_pdefind_params(with_mean=True, with_std=True, alphas=100, max_iter=1000000, cv=20,
                                  max_train_cases=np.inf)
    experiment.set_data_split(trainSplit=DataSplit({"t": 0.7}), testSplit=DataSplit({"t": 0.3}, {"t": 0.7}))

    # ---------- plot init condition ----------
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    experiment.plot_initial_condition(ax, color=color)
    ax.legend()
    plt.show()
    fig.savefig('{}/initial_conditions.eps'.format(check_create_path(*([config.plots_dir] + subfolders))))
    plt.close()

    # ---------- regressors ----------
    regressors = []
    if 'trigonometric' in regressors2use:
        regressors.append(Trigonometric(period=np.sqrt(experiment.m / experiment.k),
                                        domain_axes_name='t',
                                        units='s'))
    elif 'linear' in regressors2use:
        regressors.append(TimeReg(domain_axes_name='t', units='s'))
    experiment.set_regressor_builders(regressors_builders=regressors)

    # # ---------- eqdiff experiments ----------
    experiment.explore_eqdiff_fitting(derivatives2explore=derivatives2explore,
                                      poly2explore=poly2explore)
    experiment.plot_heatmaps()


if __name__ == "__main__":
    spatial_experiment(
        experiment_name='SpatialExperiments',
        type_of_stress='constant',
        init_cond='trigonometric',
        derivatives2explore={'t': 2, 'x': 1},
        poly2explore=[1, 2],
        regressors2use=[])
