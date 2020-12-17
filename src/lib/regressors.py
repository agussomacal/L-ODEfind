import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler


class RegressorBuilders:
    def __init__(self, name, domain_axes_name):
        self.name = str(name)
        self.domain_axes_name = domain_axes_name
        self.fit_coefs = None

    @staticmethod
    def reg_func(*args):
        pass

    def fit(self, domain_train, values_train):
        pass

    def transform(self, t):
        return self.reg_func(t, *self.fit_coefs)


class Trigonometric(RegressorBuilders):
    def __init__(self, period, domain_axes_name, units):
        RegressorBuilders.__init__(self, 'Trigonometric_peirod{}{}'.format(period, units), domain_axes_name)
        self.period = period
        self.bounds = ([self.period * 0.9, -np.pi / 2], [self.period * 1.1, np.pi / 2])
        self.p0 = [self.period, 0]

    @staticmethod
    def reg_func(t, T, fi):
        return np.cos(2 * np.pi * t / T + fi)

    def fit(self, domain_train, values_train):
        values = values_train.data.mean(axis=tuple([ix for axname, ix in values_train.domain2axis.items()
                                                    if axname != self.domain_axes_name]))
        popt, pcov = curve_fit(self.reg_func,
                               domain_train.get_range(self.domain_axes_name)[self.domain_axes_name],
                               StandardScaler().fit_transform(values.reshape((-1, 1))).ravel(),
                               bounds=self.bounds,
                               p0=self.p0)
        self.fit_coefs = popt


class TimeReg(RegressorBuilders):
    def __init__(self, domain_axes_name, units):
        RegressorBuilders.__init__(self, 'Timereg{}'.format(units), domain_axes_name)
        self.fit_coefs = {}

    @staticmethod
    def reg_func(t, **kwargs):
        return t