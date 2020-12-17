#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import src.scripts.utils
import src.scripts.utils.metrics as evaluator

import src.scripts.utils.utils
from src.lib.operators import PolyD, D, Poly, DataSplit, MultipleDelay
from src.lib.pdefind import PDEFinder, DataManager
from src.lib.variables import Variable
from src.scripts.utils.utils import save, load, check_create_path
from src.scripts import config


# ---------- dictionary of functions and target ----------
def get_x_operator_func(derivative_order, polynomial_order, rational=False):
    def x_operator(field, regressors):
        new_field = copy.deepcopy(field)
        if derivative_order > 0:
            new_field = PolyD({'t': derivative_order}) * new_field
        new_field.append(regressors)
        if rational:
            new_field.append(new_field.__rtruediv__(1.0))
        new_field = Poly(polynomial_order) * new_field
        return new_field

    return x_operator


def get_y_operator_func(derivative_order_y):
    def y_operator(field):
        new_field = D(derivative_order_y, "t") * field
        return new_field

    return y_operator


def get_x_operator_func_delay(delay_order, polynomial_order, rational=False):
    def x_operator(field):
        new_field = copy.deepcopy(field)
        if delay_order > 0:
            new_field = MultipleDelay({'t': list(range(1, delay_order + 1))}) * new_field
        new_field1 = (Poly(polynomial_order) * new_field)
        if rational:
            new_field2 = Poly(polynomial_order) * new_field.__rtruediv__(1.0)
        else:
            new_field2 = []
        new_field1.append(new_field2)
        return new_field1

    return x_operator


class ExperimentSetting:
    def __init__(self, experiment_name, subfolders=['']):
        # ---------------- paths --------------------
        self.experiment_name = experiment_name
        self.plots_path = check_create_path(*([config.plots_dir] + subfolders + [experiment_name]))

        # ----- pdfind params -----
        self.alphas = None
        self.alpha = None
        self.max_iter = None
        self.cv = None
        self.with_mean = None
        self.with_std = None
        self.max_num_params = np.inf
        self.max_train_cases = np.inf

        # ----- regressors_builders -----
        self.regressors_builders = []

        # ----- split data params -----
        self.trainSplit = None
        self.testSplit = None

        # ----- experiments record -----
        self.experiments = []

    # ========================= setters =========================
    def set_pdefind_params(self, with_mean=True, with_std=True, alphas=100, max_iter=10000, cv=20, alpha=None,
                           max_num_params=np.inf, max_train_cases=np.inf, use_lasso=True):
        self.alphas = alphas
        self.alpha = alpha
        self.max_iter = max_iter
        self.cv = cv
        self.with_mean = with_mean
        self.with_std = with_std
        self.max_num_params = max_num_params
        self.max_train_cases = max_train_cases
        self.use_lasso = use_lasso

    def set_data_split(self, trainSplit=DataSplit({"t": 0.7}), testSplit=DataSplit({"t": 0.3}, {"t": 0.7})):
        self.trainSplit = trainSplit
        self.testSplit = testSplit

    def set_regressor_builders(self, regressors_builders):
        self.regressors_builders = regressors_builders

    # ========================= get data manager =========================
    def get_domain(self):
        pass

    def get_variables(self):
        return []

    def get_regressors(self):
        # TODO: only works with time variable regressor
        reggressors = []
        domain = self.get_domain()
        variables = self.get_variables()
        for reg_builder in self.regressors_builders:
            for variable in variables:
                reg_builder.fit(self.trainSplit * variable.domain, self.trainSplit * variable)
                serie = reg_builder.transform(
                    domain.get_range(axis_names=[reg_builder.domain_axes_name])[reg_builder.domain_axes_name])
                reggressors.append(Variable(serie, domain, domain2axis={reg_builder.domain_axes_name: 0},
                                            variable_name='{}_{}'.format(variable.get_name(), reg_builder.name)))
        return reggressors

    def get_data_manager(self):
        data_manager = DataManager()
        data_manager.add_variables(self.get_variables())
        data_manager.add_regressors(self.get_regressors())
        data_manager.set_domain()
        return data_manager

    # ========================= fitting model =========================

    def fit_eqdifff(self, data_manager):
        with src.scripts.utils.utils.timeit('pdefind fitting'):
            pde_finder = PDEFinder(with_mean=self.with_mean, with_std=self.with_std, use_lasso=self.use_lasso)
            pde_finder.set_fitting_parameters(cv=self.cv, n_alphas=self.alphas, max_iter=self.max_iter,
                                              alphas=self.alpha)
            X = data_manager.get_X_dframe(self.trainSplit)
            Y = data_manager.get_y_dframe(self.trainSplit)

            if X.shape[0] > self.max_train_cases:
                sample = np.random.choice(X.shape[0], size=self.max_train_cases)
                X = X.iloc[sample, :]
                Y = Y.iloc[sample, :]
            # if X.shape[1] > self.max_num_params:
            #     raise Exception('More params than allowed: params of X={} and max value of params is {}'.format(
            #         X.shape[1], self.max_num_params))
            pde_finder.fit(X, Y)
        return pde_finder

    def load_fitsave_eqdifff(self, data_manager, filename=None, subfolders=None):
        pde_finder = load(filename + '.pickle', self.experiment_name, subfolders)
        if not pde_finder:
            save(self.fit_eqdifff(data_manager), filename + '.pickle', self.experiment_name, subfolders)
        return pde_finder

    # ========================= auxiliar functions =========================
    def get_test_time(self, data_manager, type='Variable'):
        t = Variable(data_manager.domain.get_range('t')['t'], data_manager.domain, domain2axis={'t': 0},
                     variable_name='t')
        t = self.testSplit * t
        if type == 'Variable':
            return t
        elif type == 'numpy':
            return np.array(t.data)
        else:
            raise Exception('Not implemented return type: only Variable and numpy')

    def get_test_y_yhat(self, pde_finder, data_manager):
        y = data_manager.get_y_dframe(self.testSplit)
        yhat = pd.DataFrame(pde_finder.transform(data_manager.get_X_dframe(self.testSplit)), columns=y.columns)
        return y, yhat

    def get_rsquare_of_eqdiff_fit(self, pde_finder, data_manager):
        y, yhat = self.get_test_y_yhat(pde_finder, data_manager)
        return evaluator.rsquare(yhat=yhat, y=y)
        # return evaluator.mse(yhat=yhat, y=y)

    @staticmethod
    def get_derivative_in_y(derivative_in_y, derivatives_in_x):
        # -1 means the 1 more from the depth
        if derivative_in_y == -1:
            der_y = derivatives_in_x + 1
        else:
            der_y = derivative_in_y
        return der_y

    # ========================= plots =========================
    def plot_fitted_vs_real(self, pde_finder, data_manager, col="blue"):
        y, yhat = self.get_test_y_yhat(pde_finder, data_manager)

        plt.close('all')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        ax.plot(y, yhat, '.', c=col, alpha=0.7)
        ax.set_xlabel("real data points")
        ax.set_ylabel("fitted data points")
        ax.set_title("Fitted vs Real")
        return ax

    def plot_fitted_and_real(self, pde_finder, data_manager, col="blue", subinit=None, sublen=None):
        y, yhat = self.get_test_y_yhat(pde_finder, data_manager)
        t = self.get_test_time(data_manager, type='numpy')

        if subinit is not None:
            y = y[subinit:]
            yhat = yhat[subinit:]
            t = t[subinit:]
        if sublen is not None:
            y = y[:sublen]
            yhat = yhat[:sublen]
            t = t[:sublen]

        plt.close('all')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        ax.plot(t, yhat, '.-', c=col, label="fitted", alpha=0.45)
        ax.plot(t, y, 'k-', label="real", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("function")
        ax.set_title("Time series: real and fitted")
        ax.legend()

    def plot_feature_importance(self, pdefind):
        pdefind.feature_importance.T.plot.barh(rot=45, legend=False)
        plt.tight_layout()

    def plot_coefficients(self, pdefind):
        dfc = pdefind.coefs_.T
        dfc.index = [
            c.replace('1.0', '').replace('0', '').replace('Derivative', 'D').replace('**', '^').replace('*', ' ') for c
            in dfc.index]
        dfc.plot.barh(legend=True)
        plt.tight_layout()
