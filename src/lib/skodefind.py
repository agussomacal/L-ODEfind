import copy
import numpy as np
import pandas as pd

from src.lib.operators import D, Poly, PolyD
from src.lib.pdefind import DataManager, PDEFinder
from src.lib.regressors import Trigonometric, TimeReg
from src.lib.variables import Variable, Domain


class SkODEFind:
    def __init__(self, target_derivative_order, max_derivative_order, max_polynomial_order, rational=False,
                 with_mean=True, with_std=True, alphas=150, max_iter=10000, cv=20, use_lasso=True,
                 regressor_builders=()):
        self.data_manager = DataManager()
        self.coefs_ = None
        self.rational = rational
        self.use_lasso = use_lasso
        self.cv = cv
        self.max_iter = max_iter
        self.alphas = alphas
        self.with_std = with_std
        self.with_mean = with_mean
        self.max_polynomial_order = max_polynomial_order
        self.max_derivative_order = max_derivative_order
        self.target_derivative_order = target_derivative_order
        self.var_names = None

        self.pde_finder = PDEFinder(with_mean=self.with_mean, with_std=self.with_std, use_lasso=self.use_lasso)
        self.pde_finder.set_fitting_parameters(cv=self.cv, n_alphas=self.alphas, max_iter=self.max_iter)

        self.regressor_builders = regressor_builders

    # ---------- dictionary of functions and target ----------
    def get_x_operator_func(self):
        def x_operator(field, regressors):
            new_field = copy.deepcopy(field)
            if self.max_derivative_order > 0:
                new_field = PolyD({'t': self.max_derivative_order}) * new_field
            new_field.append(regressors)
            if self.rational:
                new_field.append(new_field.__rtruediv__(1.0))
            new_field = Poly(self.max_polynomial_order) * new_field
            return new_field

        return x_operator

    def get_y_operator_func(self):
        def y_operator(field):
            new_field = D(self.target_derivative_order, "t") * field
            return new_field

        return y_operator

    def get_regressors(self, domain, variables):
        # TODO: only works with time variable regressor
        reggressors = []
        for reg_builder in self.regressor_builders:
            for variable in variables:
                reg_builder.fit(variable.domain, variable)
                serie = reg_builder.transform(
                    domain.get_range(axis_names=[reg_builder.domain_axes_name])[reg_builder.domain_axes_name])
                reggressors.append(Variable(serie, domain, domain2axis={reg_builder.domain_axes_name: 0},
                                            variable_name='{}_{}'.format(variable.get_name(), reg_builder.name)))
        return reggressors

    def prepare_data(self, X):
        # ---------- Prepare data ----------
        # no time is defined in input so invents dt=1 and starting from 0
        domain = Domain(lower_limits_dict={"t": X.index.min()},
                        upper_limits_dict={"t": X.index.max()},
                        step_width_dict={"t": np.diff(X.index)[0]})

        # define variables
        X = pd.DataFrame(X)
        variables = [Variable(X[series_name].values.ravel(), domain, domain2axis={"t": 0},
                              variable_name=series_name)
                     for i, series_name in enumerate(X.columns)]

        self.data_manager.add_variables(variables)
        self.data_manager.add_regressors(self.get_regressors(domain, variables))
        self.data_manager.set_domain()

        self.data_manager.set_X_operator(self.get_x_operator_func())
        self.data_manager.set_y_operator(self.get_y_operator_func())
        self.var_names = [var.get_full_name() for var in self.data_manager.field.data]

    def fit(self, X: (pd.DataFrame, pd.Series), y=None):
        """
        In principle the target is not needed because it uses the X time series to fit the differential equation.
        :param X: rows are series; columns index time.
        :param y:
        :return:
        """
        self.prepare_data(X)

        # ---------- fit data ----------
        self.pde_finder = PDEFinder(with_mean=self.with_mean, with_std=self.with_std, use_lasso=self.use_lasso)
        self.pde_finder.set_fitting_parameters(cv=self.cv, n_alphas=self.alphas, max_iter=self.max_iter)

        self.pde_finder.fit(self.data_manager.get_X_dframe(), self.data_manager.get_y_dframe())
        self.coefs_ = self.pde_finder.coefs_

    def predict(self, forecast_horizon):
        assert self.coefs_ is not None, 'coeffs was not defined, use set_coefs'
        times = forecast_horizon * self.data_manager.domain.step_width["t"]
        return pd.DataFrame(
            self.pde_finder.integrate(
                t=times +
                  self.data_manager.domain.upper_limits["t"] - self.data_manager.domain.step_width["t"],
                data_manager=self.data_manager,
                dery=self.target_derivative_order),
            index=times + self.data_manager.domain.upper_limits["t"]
        )[0]

    def __str__(self):
        return 'skodefind_target{}_maxd{}_maxpoly{}'.format(self.target_derivative_order, self.max_derivative_order,
                                                            self.max_polynomial_order)


class SKIntegrate(SkODEFind):
    def __init__(self, target_derivative_order, max_derivative_order, max_polynomial_order):
        super().__init__(target_derivative_order, max_derivative_order, max_polynomial_order)

    def set_coefs(self, coefs):
        self.coefs_ = coefs
        self.pde_finder.coefs_ = coefs

    def fit(self, X: (pd.DataFrame, pd.Series), y=None):
        self.prepare_data(X)
