import copy
import itertools

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import odeint
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from src.lib.algorithms import get_func_for_ode
from src.lib.operators import Identity, PolyD
from src.lib.variables import Variable, Domain, SymVariable, Field, SymField
from src.scripts.utils.metrics import error


##########################################################
#                       DataManager
##########################################################
class DataManager:
    def __init__(self):
        self.X_operator = lambda Field, regressors: Field
        self.y_operator = lambda Field: Field

        self.domain = Domain()

        self.field = Field()
        self.regressors = Field()
        self.sym_regressors = SymField()
        self.sym_field = SymField()

    def set_domain(self, domain_info=None):
        if domain_info is None:
            # in case none is passed then it will get the maximum domain from the variables of the field.
            self.field.set_domain()
            self.domain = self.field.domain
        elif isinstance(domain_info, dict):
            if ("lower_limits_dict" in domain_info.keys()) and ("upper_limits_dict" in domain_info.keys()) and \
                    ("step_width_dict" in domain_info.keys()):
                lower_limits_dict = domain_info["lower_limits_dict"]
                upper_limits_dict = domain_info["upper_limits_dict"]
                step_width_dict = domain_info["step_width_dict"]
            elif all([isinstance(element, np.ndarray) for element in domain_info.values()]):
                lower_limits_dict = {}
                upper_limits_dict = {}
                step_width_dict = {}
                for axis_name, range_vals in domain_info.items():
                    lower_limits_dict[axis_name] = range_vals[0]
                    upper_limits_dict[axis_name] = range_vals[-1]
                    step_width_dict[axis_name] = range_vals[1] - range_vals[0]
            else:
                Exception("imput should be or dict of ranges or dict of limits and steps")
            self.domain = Domain(lower_limits_dict, upper_limits_dict, step_width_dict)
        elif isinstance(domain_info, Domain):
            self.domain = copy.deepcopy(domain_info)

    def add_variables(self, variables):  # type:((list| Variable)) -> None
        self.field.append(variables)
        self.sym_field.append(variables)

    def add_field(self, field):
        self.field.append(field)
        self.sym_field.append(field)

    def add_regressors(self, regressors):  # : (Variable, Field)
        self.regressors.append(regressors)
        self.sym_regressors.append(regressors)

    def set_X_operator(self, operator):
        self.X_operator = operator

    def set_y_operator(self, operator):
        self.y_operator = operator

    @staticmethod
    def filter_yvars_in_Xvars(y_field, x_field):  # type: (Field, Field) -> Field
        y_var_names = [var.get_full_name() for var in y_field.data]
        return Field([var for var in x_field.data if var.get_full_name() not in y_var_names])

    @staticmethod
    def filter_ysymvars_in_Xsymvars(y_field, x_field):  # type: (SymField, SymField) -> SymField
        y_sym_expressions = [str(sym_var) for sym_var in y_field.data]
        return SymField([symvar for symvar in x_field.data if str(symvar) not in y_sym_expressions])

    def get_X_sym(self, split_operator=Identity()):
        """
        gets the simbolic expression of X
        :return:
        """
        sym_X = self.X_operator(split_operator * self.sym_field,
                                split_operator * self.sym_regressors if self.sym_regressors != [] else self.sym_regressors)

        return self.filter_ysymvars_in_Xsymvars(self.get_y_sym(split_operator), x_field=sym_X)

    def get_y_sym(self, split_operator=Identity()):
        """
        gets the simbolic expression of y
        :return:
        """
        return self.y_operator(split_operator * self.sym_field)

    def get_X(self, split_operator=Identity()):
        X = self.X_operator(split_operator * self.field,
                            split_operator * self.regressors if self.sym_regressors != [] else self.regressors)
        return self.filter_yvars_in_Xvars(self.get_y(split_operator), x_field=X)

    def get_y(self, split_operator=Identity()):
        return self.y_operator(split_operator * self.field)

    def get_X_dframe(self, split_operator=Identity()):
        return self.get_X(split_operator).to_pandas()

    def get_y_dframe(self, split_operator=Identity()):
        return self.get_y(split_operator).to_pandas()

    def get_Xy_eq(self):
        X = self.get_X()
        X = SymField([SymVariable(x.name, SymVariable.get_init_info_from_variable(x)[1], x.domain) for x in X.data])
        Y = self.get_y()
        Y = SymField([SymVariable(y.name, SymVariable.get_init_info_from_variable(y)[1], y.domain) for y in Y.data])
        return X, Y


##########################################################
#               StandardScalerForPDE
##########################################################
class StandardScalerForPDE(StandardScaler):
    def sym_var_transform(self, X, y='deprecated', copy=None):
        if isinstance(X, SymVariable):
            X = SymField(X)
        # if isinstance(X, Variable):
        #     X = Field(X)
        if isinstance(X, SymField):
            new_field = X * 1
            if self.with_mean:
                new_field = new_field - self.mean_
            if self.with_std:
                new_field = new_field / self.scale_
            return new_field

    def sym_var_inverse_transform(self, X, copy=None):
        if isinstance(X, SymVariable):
            X = SymField(X)
        if isinstance(X, Variable):
            X = Field(X)
        if isinstance(X, (SymField, Field)):
            new_field = X * 1
            if self.with_std:
                new_field = new_field * self.scale_
            if self.with_mean:
                new_field = new_field + self.mean_
            return new_field


##########################################################
#                       PDEFinder
##########################################################
class PDEFinder:
    def __init__(self, with_mean=True, with_std=True, use_lasso=True):
        self.lasso_cv = None
        self.X_scaler = StandardScalerForPDE(with_mean=with_mean, with_std=with_std)
        self.y_scaler = StandardScalerForPDE(with_mean=with_mean, with_std=with_std)
        self.coefs_ = pd.DataFrame()
        self.coef_threshold = 0
        self.feature_importance = pd.DataFrame()
        self.use_lasso = use_lasso

    def prepare_for_fitting(self, X_train, y_train):
        """
        Creates transformations functions to make fitting more stable. (Standarize)
        :param X_train:
        :param y_train:
        :return:
        """
        self.X_scaler.fit(X_train)
        self.y_scaler.fit(y_train)

        if self.X_scaler.with_mean is False:
            self.X_scaler.mean_ = 0
        if self.y_scaler.with_mean is False:
            self.y_scaler.mean_ = 0

        if self.X_scaler.with_std is False:
            self.X_scaler.scale_ = 1
        if self.y_scaler.with_std is False:
            self.y_scaler.scale_ = 1

    def set_fitting_parameters(self, cv=10, n_alphas=100, alphas=None, max_iter=10000):
        # self.lasso_cv = MultiTaskLassoCV(eps=0.0001,
        #                                  n_alphas=n_alphas,
        #                                  alphas=alphas,
        #                                  fit_intercept=False,
        #                                  normalize=False,
        #                                  # precompute='auto',
        #                                  max_iter=max_iter,
        #                                  tol=0.0001,
        #                                  copy_X=True,
        #                                  cv=cv,
        #                                  verbose=False,
        #                                  n_jobs=-1,
        #                                  # positive=False,
        #                                  random_state=None,
        #                                  selection='cyclic')
        # self.lasso_cv = ElasticNetCV(alphas=alphas,
        #                              copy_X=True,
        #                              cv=cv,
        #                              eps=0.00001,
        #                              fit_intercept=False,
        #                              l1_ratio=0.2,
        #                              max_iter=max_iter,
        #                              n_alphas=n_alphas,
        #                              n_jobs=-1,
        #                              normalize=False,
        #                              positive=False,
        #                              precompute='auto',
        #                              random_state=None,
        #                              selection='random',
        #                              tol=0.00001,
        #                              verbose=0)
        if self.use_lasso:
            self.lasso_cv = LassoCV(eps=0.00001,
                                    n_alphas=n_alphas,
                                    alphas=alphas,
                                    fit_intercept=False,
                                    normalize=False,
                                    precompute='auto',
                                    max_iter=max_iter,
                                    tol=0.000001,
                                    copy_X=True,
                                    cv=cv,
                                    verbose=False,
                                    n_jobs=-1,
                                    positive=False,
                                    random_state=42,
                                    selection='random')
        else:
            self.lasso_cv = LinearRegression(fit_intercept=False,
                                             normalize=False,
                                             n_jobs=-1,
                                             copy_X=True)
        # TODO: add variable to decide the fitting algorithm (MultiTaskLassoCV, ElasticNetCV, LassoCV or linear)

    def fit(self, X_train, y_train, verbose=False):
        """
        Given the derivatives or polynomial couplings between the interest variables it finds the differential equation.
        :type X_train: pd.DataFrame
        :type y_train: pd.DataFrame
        :return:
        """
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X_train.isna().any().any() or y_train.isna().any().any():
            where_nas = X_train.isna().any(axis=1) | y_train.isna().any(axis=1)
            X_train = X_train.loc[~where_nas, :]
            y_train = y_train.loc[~where_nas, :]
            if verbose:
                print('Warning, there where {} rows in trainig data with NaN or Infs'.format(np.sum(where_nas)))

        self.prepare_for_fitting(X_train, y_train)  # preprocess, scaling via z-score.
        for name, yt in zip(y_train.columns, self.y_scaler.transform(y_train).T):
            X_train_temp = self.X_scaler.transform(X_train.values)
            # transform first, then fit
            coefs_temp = np.zeros(X_train_temp.shape[1])
            mask = np.repeat(True, X_train_temp.shape[1])
            for i in range(10):  # maximum depth
                lasso_cv = copy.deepcopy(self.lasso_cv)
                lasso_cv.fit(X_train_temp[:, mask], yt)
                coefs_temp[mask] = lasso_cv.coef_

                if np.all(lasso_cv.coef_ != 0) or np.all(
                        lasso_cv.coef_ == 0):  # all the coefs are not zero then is the minimum expression possible
                    break
                mask[mask] = lasso_cv.coef_ != 0

            # transform coefs to forget the scaling and have the real coeficients.
            self.coefs_ = pd.concat([self.coefs_, pd.DataFrame(coefs_temp.reshape(1, -1),
                                                               columns=X_train.columns,
                                                               index=[name])],
                                    axis=0)

        self.coefs_ = self._get_coefs(self.coefs_)
        self.determine_feature_importance(X_train, y_train)

    def determine_feature_importance(self, X, y):
        min_error = error(self.inner_transform(X, self.coefs_), y).values
        max_error = error(self.inner_transform(X, self.coefs_ * 0), y).values

        self.feature_importance = pd.DataFrame(0, columns=self.coefs_.columns, index=self.coefs_.index)
        if np.any(max_error < min_error):
            self.feature_importance = np.nan * self.feature_importance
        else:
            for target_i, j in itertools.product(*list(map(lambda x: list(range(x)), self.coefs_.shape))):
                coefs_ = self.coefs_.copy()
                coefs_.iloc[target_i, j] = 0
                self.feature_importance.iloc[target_i, j] = (error(self.inner_transform(X, coefs_.iloc[target_i, :]),
                                                                   y.iloc[:, target_i]) - min_error[target_i]) / \
                                                            (max_error - min_error)[target_i]
            self.feature_importance = self.feature_importance.divide(self.feature_importance.sum(axis=1), axis=0)

    def _get_coefs(self, coefs):
        coefs_ = coefs / self.X_scaler.scale_
        coefs_ = coefs_.apply(lambda c: c * self.y_scaler.scale_)
        # TODO: if no 1.00000... (the constant) is present this fails. Because it always should have to balance zscoreing
        if any(["1.00" in c for c in coefs_.columns]):
            coefs_.loc[:, [True if "1.00" in c else False for c in coefs_.columns]] += (
                    self.y_scaler.mean_ - np.dot(coefs_, self.X_scaler.mean_)).reshape((-1, 1))

        coefs_[coefs_.abs() < self.coef_threshold] = 0
        return coefs_

    def inner_transform(self, X, coefs):
        return np.matmul(X.values, coefs.values.T)

    def transform(self, X):
        return self.inner_transform(X, self.coefs_)

    def get_equation(self, sym_x, sym_y):
        """

        :param sym_x: SymField
        :param sym_y: SymField
        :return:
        :rtype: (SymVariable, SymField)
        """
        sym_field_res = sym_x.matmul(self.coefs_.T)
        sym_field_res = sym_field_res - sym_y
        sym_field_res.simplify()
        return sym_field_res

    @staticmethod
    def get_v0(data_manager, dery):
        ax_name = data_manager.domain.axis_names[0]
        starting_point = {ax_name: -1}  # make predictions to the future.
        init_point = starting_point.copy()
        # get derivatives up to the unknown
        v0 = []
        for sym_var, var in zip(data_manager.sym_field.data, data_manager.field.data):
            terms = [var.name.diff(ax_name, i) for i in range(dery)]
            v0_temp = (PolyD(derivative_order_dict={ax_name: dery - 1}) * var).evaluate_ix(init_point)
            v0_temp = [v0_temp[str(f).replace(' ', '')] if i == 0 else v0_temp['1.0*' + str(f).replace(' ', '')] for
                       i, f in enumerate(terms)]
            v0 += v0_temp

        last_time = data_manager.domain.upper_limits[ax_name] - data_manager.domain.step_width[ax_name] * (dery - 1)

        return v0, last_time

    def integrate(self, t, data_manager, dery):
        """

        :param split_data_operator_list:
        :type dm: DataManager
        :param starting_point: when more than one variable it is used to define the other domain points.
        :type starting_point: dict
        :type domain_variable2predict: str
        :type horizon: int
        :return:
        """
        assert len(np.shape(t)) == 1, "horizons must be a 1dimensional list or array"
        assert len(data_manager.domain) == 1, "only works with 1d variables."
        var_names = [var.get_full_name() for var in data_manager.field.data]

        eq_x_sym_expression, eq_y_sym_expression = data_manager.get_Xy_eq()
        ode_func = get_func_for_ode(eq_x_sym_expression.matmul(self.coefs_.T),
                                    eq_y_sym_expression, data_manager.regressors)
        v0, last_time = self.get_v0(data_manager, dery)
        t = np.append(last_time, t)
        v = scipy.integrate.odeint(func=ode_func, y0=v0, t=t)

        if len(np.shape(v)) == 1:
            v = np.reshape(v, (-1, 1))

        res = np.array(v[-len(t):, np.linspace(0, v.shape[1], len(var_names) + 1, dtype=int)[:-1]], dtype=float)
        res = res[1:, :]  # eliminate the v0
        return res
