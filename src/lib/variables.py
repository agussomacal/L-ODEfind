import numpy as np
import pandas as pd
import copy
import sympy
from sympy.core.function import UndefinedFunction
from collections import OrderedDict

from src.lib.algorithms import get_symfunc_from_name_domain
from src.lib.algorithms import merge_dicts


##########################################################
#                       Domain
##########################################################
class Domain:
    """
        TODO: a domain var could be a network or a range
        """

    def __init__(self, lower_limits_dict={}, upper_limits_dict={}, step_width_dict={}):
        if not (set(lower_limits_dict.keys()) == set(upper_limits_dict.keys()) == set(step_width_dict.keys())):
            kint = set(lower_limits_dict.keys()).intersection(set(upper_limits_dict.keys())).intersection(
                set(step_width_dict.keys()))
            lower_limits_dict = {k: lower_limits_dict[k] for k in kint}
            upper_limits_dict = {k: upper_limits_dict[k] for k in kint}
            step_width_dict = {k: step_width_dict[k] for k in kint}
        self.axis_names = []
        self.shape = dict()

        self.lower_limits = dict()
        self.upper_limits = dict()
        self.step_width = dict()
        self.add_axis(lower_limits_dict, upper_limits_dict, step_width_dict)

    def get_aproximate_index(self, ax_name, ax_value):
        assert ax_value <= self.upper_limits[ax_name], "should be"
        assert ax_value >= self.lower_limits[ax_name]
        return int((ax_value - self.lower_limits[ax_name]) / self.step_width[ax_name])

    def get_value_from_index(self, ax_name, ax_index):
        assert ax_index <= self.shape[ax_name], "should be less than the shape of the axis"
        assert ax_index >= -self.shape[ax_name], "should be greater than - the shape of the axis"
        if ax_index >= 0:
            return self.lower_limits[ax_name] + self.step_width[ax_name] * ax_index
        elif ax_index < 0:
            return self.upper_limits[ax_name] + self.step_width[ax_name] * ax_index

    def get_shape(self, axis_names=None):
        if axis_names is None:
            axis_names = self.axis_names
        elif isinstance(axis_names, str):
            axis_names = [axis_names]

        return {ax_name: self.get_aproximate_index(ax_name, self.upper_limits[ax_name]) for ax_name in axis_names}

    def get_range(self, axis_names=None):
        if axis_names is None:
            axis_names = self.axis_names
        elif isinstance(axis_names, str):
            axis_names = [axis_names]

        return {ax_name: np.arange(self.lower_limits[ax_name],
                                   self.upper_limits[ax_name] + self.step_width[ax_name],  # include the las element.
                                   self.step_width[ax_name])
                for ax_name in axis_names}

    def is_index_in_range(self, value_dict):
        for ax_name, domain_index in value_dict.items():
            if 0 > domain_index or self.shape[ax_name] <= domain_index:
                return False
        return True

    def get_subdomain_from_index(self, axis_names_ixes):
        """
        Returns a new domain with less axis and possibly less range.
        :param axis_names: if a dict it should contain axis names and a list of 2 elements denoting lower and upper limits
        :return:
        """
        sub_domain = Domain()
        for ax_name, limits in axis_names_ixes.items():
            sub_domain.add_axis(
                {ax_name: max(self.get_value_from_index(ax_name, limits[0]), self.lower_limits[ax_name])},
                # to avoid going outside range.
                {ax_name: min(self.get_value_from_index(ax_name, limits[1]), self.upper_limits[ax_name])},
                {ax_name: self.step_width[ax_name]})
        return sub_domain

    def get_subdomain(self, axis_names=None):
        """
        Returns a new domain with less axis and possibly less range.
        :param axis_names: if a dict it should contain axis names and a list of 2 elements denoting lower and upper limits
        :return:
        """
        if axis_names is None:
            axis_names = {ax_name: [self.lower_limits[ax_name], self.upper_limits[ax_name]] for ax_name in
                          self.axis_names}
        elif isinstance(axis_names, str):
            axis_names = {axis_names: [self.lower_limits[axis_names], self.upper_limits[axis_names]]}

        sub_domain = Domain()
        for ax_name, limits in axis_names.items():
            sub_domain.add_axis({ax_name: max(limits[0], self.lower_limits[ax_name])},  # to avoid going outside range.
                                {ax_name: min(limits[1], self.upper_limits[ax_name])},
                                {ax_name: self.step_width[ax_name]})
        return sub_domain

    def add_axis(self, lower_limits_dict, upper_limits_dict, step_width_dict):
        """
        Add axis if it is not yet in the domain.
        :param domain2values:
        :return:
        """
        assert set(lower_limits_dict.keys()) == set(upper_limits_dict.keys()) == set(step_width_dict.keys())
        axis_names = step_width_dict.keys()
        for axis_name in axis_names:
            if axis_name not in self.axis_names:
                self.lower_limits[axis_name] = lower_limits_dict[axis_name]
                self.upper_limits[axis_name] = upper_limits_dict[axis_name]
                self.step_width[axis_name] = step_width_dict[axis_name]
                self.axis_names.append(axis_name)
        self.shape = self.get_shape()

    def __mul__(self, other):
        """
        cartesian product: the result is the union of the domains. But it must have equal ranges.
        :param other:
        :return:
        """
        if isinstance(other, Domain):
            lower_limits = merge_dicts(self.lower_limits, other.lower_limits, max)
            upper_limits = merge_dicts(self.upper_limits, other.upper_limits, min)
            for ax_name in set(other.axis_names).intersection(set(self.axis_names)):
                assert self.step_width[ax_name] == other.step_width[ax_name]

            newdict = self.step_width.copy()
            newdict.update(other.step_width)
            return Domain(lower_limits, upper_limits, newdict)

    def __len__(self):
        return len(self.axis_names)


##########################################################
#                       Variable
##########################################################
class Variable:
    def __init__(self, data, domain, domain2axis, variable_name):
        self.data = np.array(data)
        self.domain = copy.deepcopy(domain)  # dict {"coord_name" : range of values}
        self.domain2axis = domain2axis
        self.shape = self.get_shape()

        self.name = variable_name
        if isinstance(variable_name, str):
            self.name = get_symfunc_from_name_domain(variable_name, self.domain)
        if isinstance(self.name, UndefinedFunction):
            self.name = sympy.sympify(float(str(self.name.__name__)))

    def get_subset_from_index_limits(self, domain_index_sub_range_dict):
        """
        Get a new variable out of a subset of this variable given the indexes of the domain.

        :param domain_index_sub_range_dict: dict {axis name: [lower index, upper index] ...}
        :return: new Variable with the domain reduced.
        """
        domain_sub_range_dict = {ax_name: [self.domain.get_value_from_index(ax_name, ix_limits[0]),
                                           self.domain.get_value_from_index(ax_name, ix_limits[1])]
                                 for ax_name, ix_limits in domain_index_sub_range_dict.items()
                                 if ax_name in self.domain.axis_names}

        data = self.data * 1
        for axis_name, limits in domain_index_sub_range_dict.items():
            if axis_name in self.domain2axis.keys():
                data = np.take(data, np.arange(limits[0], limits[1]), axis=self.domain2axis[axis_name])
        return Variable(data,
                        domain=self.domain.get_subdomain(domain_sub_range_dict),
                        domain2axis=self.domain2axis,
                        variable_name=self.name)

    def get_name(self):
        return str(self.name.func)

    def get_full_name(self):
        return ''.join([s for s in str(self.name) if s != ' '])

    def get_sym_var_from_name(self):
        return SymVariable(self.name, {}, self.domain)

    def get_axis(self, axis_name):
        return self.domain2axis[axis_name]

    def eval(self, domain_values):
        assert isinstance(domain_values, dict), "Should be a dictionary"
        indexes = [np.nan] * len(domain_values)
        for ax_name, ax_value in domain_values.items():
            indexes[self.domain2axis[ax_name]] = self.domain.get_aproximate_index(ax_name, ax_value)
        return self.data[tuple(indexes)]

    def index_eval(self, domain_index_values):
        """
        Finds the value of the variable given the ndarray indexes of the domain.
        :param domain_index_values:
        :return:
        """
        assert isinstance(domain_index_values, dict), "Should be a dictionary"
        indexes = [np.nan] * len(domain_index_values)
        for ax_name, ax_index in domain_index_values.items():
            indexes[self.domain2axis[ax_name]] = ax_index
        return self.data[tuple(indexes)]

    def get_shape(self):
        return self.data.shape

    def add_axis(self, domain):
        for axis_name in domain.axis_names:
            if axis_name not in self.domain2axis.keys():
                self.domain.add_axis({axis_name: domain.lower_limits[axis_name]},
                                     {axis_name: domain.upper_limits[axis_name]},
                                     {axis_name: domain.step_width[axis_name]})
                self.domain2axis[axis_name] = len(self.domain2axis)
                self.data = np.repeat(np.expand_dims(self.data, axis=self.domain2axis[axis_name]),
                                      self.domain.shape[axis_name],
                                      axis=self.domain2axis[axis_name])
        self.shape = self.get_shape()

    def reorder_axis(self, domainaxisorder):
        """
        when combining 2 variables their axis names should be in the same order to allow valid operations.
        :param axis_names:
        :return:
        """
        if isinstance(domainaxisorder, Domain):
            domain2axis = {ax_name: i for i, ax_name in enumerate(domainaxisorder.axis_names)}
            assert set(domain2axis.keys()) == set(self.domain2axis.keys())
        elif isinstance(domainaxisorder, dict):
            domain2axis = domainaxisorder
            assert set(domain2axis.keys()) == set(self.domain2axis.keys())
        else:
            Exception("Incorrect type")

        permutation = [np.nan] * len(domain2axis)
        for axis_name, axis_dim in domain2axis.items():
            permutation[axis_dim] = self.domain2axis[axis_name]

        self.data = np.transpose(self.data, permutation)  # np.transpose(self.data, permutation)
        self.domain2axis = domain2axis
        self.shape = self.get_shape()

    @staticmethod
    def prepare_data_for_operation(var1, var2):

        new_domain = var1.domain * var2.domain
        domain2axis = {ax_name: i for i, ax_name in enumerate(new_domain.axis_names)}

        new_variable_1 = copy.deepcopy(var1)
        new_variable_1.add_axis(new_domain)
        new_variable_1.reorder_axis(domain2axis)

        new_variable_2 = copy.deepcopy(var2)
        new_variable_2.add_axis(new_domain)
        new_variable_2.reorder_axis(domain2axis)
        return new_variable_1, new_variable_2

    def __mul__(self, other):
        if isinstance(other, Variable):
            mul_variable_1, mul_variable_2 = self.prepare_data_for_operation(self, other)

            mul_variable_1.data = mul_variable_1.data * mul_variable_2.data
            mul_variable_1.name = other.name * self.name
            return mul_variable_1
        elif isinstance(other, (float, int)):
            new_var = copy.deepcopy(self)
            new_var.data = new_var.data * other
            new_var.name = self.name * other
            return new_var

    def __pow__(self, power, modulo=None):
        if isinstance(power, Variable):
            mul_variable_1, mul_variable_2 = self.prepare_data_for_operation(self, power)

            mul_variable_1.data = mul_variable_1.data ** mul_variable_2.data
            mul_variable_1.name = self.name ** power.name
            return mul_variable_1
        elif isinstance(power, (float, int)):
            new_var = copy.deepcopy(self)
            new_var.data = new_var.data ** power
            new_var.name = self.name ** power
            return new_var

    def __rpow__(self, other):
        if isinstance(other, Variable):
            return Variable.__pow__(other, self)
        elif isinstance(other, (float, int)):
            new_var = copy.deepcopy(self)
            new_var.data = other ** new_var.data
            new_var.name = other ** self.name
            return new_var

    def __add__(self, other):
        if isinstance(other, Variable):
            mul_variable_1, mul_variable_2 = self.prepare_data_for_operation(self, other)

            mul_variable_1.data = mul_variable_1.data + mul_variable_2.data
            mul_variable_1.name = other.name + self.name
            return mul_variable_1
        elif isinstance(other, (float, int)):
            new_var = copy.deepcopy(self)
            new_var.data = new_var.data + other
            new_var.name = self.name + other
            return new_var

    def __radd__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Variable):
            new_other = copy.deepcopy(other)
            new_other.data = 1 / new_other.data
            new_other.name = 1 / new_other.name
            return self * new_other
        elif isinstance(other, (int, float)):
            return self * (1.0 / other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            new_self = copy.deepcopy(self)
            new_self.data = other / new_self.data
            new_self.name = other / new_self.name
            return new_self

    def __eq__(self, other):
        if isinstance(other, Variable):
            if other.shape == self.shape:
                new_other = copy.deepcopy(other)
                new_other.reorder_axis(self.domain2axis)
                return np.all(new_other.data == self.data)
        return False

    def __len__(self):
        return 1

    def __str__(self):
        return self.get_full_name()


##########################################################
#                    SymVariable
##########################################################
class SymVariable:
    def __init__(self, sym_expression, evaluation_dict, domain):
        self.domain = domain
        self.sym_expression = sympy.simplify(sym_expression)
        self.evaluation_dict = evaluation_dict

    @staticmethod
    def get_init_info_from_variable(variable):
        sym_expression = get_symfunc_from_name_domain(variable.get_name(), variable.domain)
        # str(self.sym_v.sym_expression).split(str(self.sym_v.sym_expression._sorted_args))[0]
        evaluation_dict = {variable.get_name(): variable.index_eval}
        return sym_expression, evaluation_dict, copy.deepcopy(variable.domain)

    def __str__(self):
        return str(self.sym_expression).replace(' ', '')

    def simplify(self):
        self.sym_expression = sympy.simplify(self.sym_expression)

    def evaluate(self, value_dict):
        """
        We need to evaluate over the domain because it could be shifted.
        TODO: when asking for t+k in the prediction situation this won't work.
        TODO: functions can't be nested f(f(x,y),y)
        :param value_dict:
        :return:
        """
        function_replace_dict = dict()
        for f in self.sym_expression.atoms(sympy.Function):
            f_domain = f.free_symbols  # get the symbols (variable domain) in the function
            # substitutes the values dict in the function and given the hidden shifts it may have gets the real domain
            # points where to evaluate.
            domain_order = OrderedDict([(str(dom_var), str(f).find(str(dom_var))) for dom_var in f_domain])
            domain_order = [k for k, v in sorted(domain_order.items(), key=lambda dp: dp[1])]
            domain_values = list(map(int, str(f.subs(value_dict)).split("(")[1].split(")")[0].split(",")))
            new_value_dict = {domain_var_name: value for domain_var_name, value in zip(domain_order, domain_values)}
            # checks that this values are in the range and substitutes the function.
            if self.domain.is_index_in_range(new_value_dict):
                function_replace_dict[str(f)] = self.evaluation_dict[str(f.func)](new_value_dict)
        return sympy.simplify(self.sym_expression.subs(sympy.sympify(function_replace_dict)).subs(value_dict))

    def get_subset_from_index_limits(self, domain_index_sub_range_dict):
        """
        Get a variable with a reduced domain
        :return:
        """
        domain_sub_range_dict = {ax_name: [self.domain.get_value_from_index(ax_name, ix_limits[0]),
                                           self.domain.get_value_from_index(ax_name, ix_limits[1])]
                                 for ax_name, ix_limits in domain_index_sub_range_dict.items()}

        return SymVariable(sym_expression=self.sym_expression,
                           evaluation_dict=self.evaluation_dict,
                           domain=self.domain.get_subdomain(domain_sub_range_dict))

    @staticmethod
    def shift(sym_variable, axis_shifts):
        """
        Shift over some axis. ej: (x -> x+2, y -> y-3)
        :param axis_shifts:
        :return:
        """
        sym_expression = sym_variable.sym_expression.subs({ax_name: sympy.Symbol(ax_name) + shift
                                                           for ax_name, shift in axis_shifts.items()})
        return SymVariable(sym_expression=sym_expression,
                           evaluation_dict=sym_variable.evaluation_dict,
                           domain=copy.deepcopy(sym_variable.domain))

    @staticmethod
    def prepare_data_for_operation(var1, var2):
        new_var1 = copy.deepcopy(var1)
        new_var2 = copy.deepcopy(var2)
        new_domain = new_var1.sym_domain * new_var2.sym_domain

        new_var1.add_axis(new_domain)
        new_var2.add_axis(new_domain)
        return new_var1, new_var2

    def __add__(self, other):
        if isinstance(other, SymVariable):
            newdict = self.evaluation_dict.copy()
            newdict.update(other.evaluation_dict)
            return SymVariable(sym_expression=self.sym_expression + other.sym_expression,
                               evaluation_dict=newdict,
                               domain=self.domain * other.domain)
        if isinstance(other, (int, float)):
            return SymVariable(sym_expression=self.sym_expression + other,
                               evaluation_dict=self.evaluation_dict,
                               domain=copy.deepcopy(self.domain))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __mul__(self, other):
        if isinstance(other, SymVariable):
            newdict = self.evaluation_dict.copy()
            newdict.update(other.evaluation_dict)
            return SymVariable(sym_expression=self.sym_expression * other.sym_expression,
                               evaluation_dict=newdict,
                               domain=self.domain * other.domain)
        if isinstance(other, (int, float)):
            return SymVariable(sym_expression=self.sym_expression * other,
                               evaluation_dict=self.evaluation_dict,
                               domain=copy.deepcopy(self.domain))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):
        if isinstance(power, SymVariable):
            newdict = self.evaluation_dict.copy()
            newdict.update(power.evaluation_dict)
            return SymVariable(sym_expression=self.sym_expression ** power.sym_expression,
                               evaluation_dict=newdict,
                               domain=self.domain * power.domain)
        if isinstance(power, (int, float)):
            return SymVariable(sym_expression=self.sym_expression ** power,
                               evaluation_dict=self.evaluation_dict,
                               domain=copy.deepcopy(self.domain))

    def __rpow__(self, other):
        if isinstance(other, SymVariable):
            return SymVariable.__pow__(other, self)
        elif isinstance(other, (float, int)):
            return SymVariable(sym_expression=other ** self.sym_expression,
                               evaluation_dict=self.evaluation_dict,
                               domain=copy.deepcopy(self.domain))

    def __truediv__(self, other):
        if isinstance(other, SymVariable):
            newdict = self.evaluation_dict.copy()
            newdict.update(other.evaluation_dict)
            return SymVariable(sym_expression=self.sym_expression / other.sym_expression,
                               evaluation_dict=newdict,
                               domain=self.domain * other.domain)
        if isinstance(other, (int, float)):
            return SymVariable(sym_expression=self.sym_expression / other,
                               evaluation_dict=self.evaluation_dict,
                               domain=copy.deepcopy(self.domain))

    def __rtruediv__(self, other):
        if isinstance(other, SymVariable):
            return SymVariable.__truediv__(other, self)
        if isinstance(other, (int, float)):
            return SymVariable(sym_expression=other / self.sym_expression,
                               evaluation_dict=self.evaluation_dict,
                               domain=copy.deepcopy(self.domain))

    def __len__(self):
        return 1

    def __eq__(self, other):
        if isinstance(other, SymVariable):
            return other.sym_expression == self.sym_expression
        else:
            return False


##########################################################
#                       Field
##########################################################
class Field:
    def __init__(self, vars=[]):
        self.data = []
        self.domain = Domain()
        self.append(vars)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "[" + ", ".join([str(var) for var in self.data]) + "]"

    def get_sym_field_from_name(self):
        return SymField([var.get_sym_var_from_name() for var in self.data])

    def append(self, new_vars):
        if isinstance(new_vars, list):
            if all([isinstance(var, (Field, Variable, int, float)) for var in new_vars]):
                for var in new_vars:
                    if isinstance(var, (int, float)):
                        var = Variable(np.array(var), Domain(), domain2axis={},
                                       variable_name=str(var))  # new constant variable
                    self.append(var)
        elif isinstance(new_vars, Variable):
            self.data.append(new_vars)
        elif isinstance(new_vars, Field):
            for var in new_vars.data:
                self.data.append(var)
        else:
            raise Exception("data should be a list of Variables or Fields, Variable or Field")

        self.set_domain()
        assert all([isinstance(var, (Variable, int, float)) for var in self.data]), "all data should be Variables"

    def evaluate(self, value_dict):
        return OrderedDict([(var.get_full_name(), var.eval(value_dict)) for var in self.data])

    def evaluate_ix(self, value_dict):
        return OrderedDict([(var.get_full_name(), var.index_eval(value_dict)) for var in self.data])

    def set_domain(self):
        for var in self.data:
            self.domain = self.domain * var.domain

    def to_pandas(self, domain=None):
        if domain is None:
            domain = self.domain

        df = pd.DataFrame()
        for var in copy.deepcopy(self.data):
            var.add_axis(domain)
            var.reorder_axis(domain)
            df = df.append(pd.Series(var.data.ravel(), name=var.get_full_name()))
        return df.T

    def get_subset_from_index_limits(self, domain_index_sub_range_dict):
        return Field([var.get_subset_from_index_limits(domain_index_sub_range_dict) for var in self.data])

    def dot(self, other):
        new_field = self * other
        new_field.set_domain()
        return sum(new_field.data)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_field = copy.deepcopy(self)
            new_field.data = [other * var for var in self.data]
            return new_field
        elif isinstance(other, Field) and (len(other) == len(self)):
            new_field = copy.deepcopy(self)
            new_field.data = [svar * ovar for svar, ovar in zip(self.data, other.data)]
            return new_field

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):
        if isinstance(power, (int, float)):
            new_field = copy.deepcopy(self)
            new_field.data = [var ** power for var in self.data]
            return new_field
        elif isinstance(power, Field) and (len(power) == len(self)):
            new_field = copy.deepcopy(self)
            new_field.data = [svar ** pow for svar, pow in zip(self.data, power.data)]
            return new_field

    def __rpow__(self, other):
        if isinstance(other, Field):
            return Field.__pow__(other, self)
        elif isinstance(other, (int, float)):
            new_field = copy.deepcopy(self)
            new_field.data = [other ** var for var in self.data]
            return new_field

    def __add__(self, other):
        new_field = copy.deepcopy(self)
        if isinstance(other, (int, float)):
            new_field.data = [var + other for var in self.data]  # for some reason putting other + var breaks it all
            return new_field
        elif isinstance(other, Field) and (len(other) == len(self)):
            new_field.data = [svar + ovar for svar, ovar in zip(self.data, other.data)]
            return new_field
        elif isinstance(other, (list, np.ndarray)) and len(other) == len(self):
            new_field.data = [svar + o for svar, o in zip(self.data, other)]
            return new_field

    def __sub__(self, other):
        return self + (-1 * other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        else:
            new_field = copy.deepcopy(self)
            if isinstance(other, Field) and (len(other) == len(self)):
                new_field.data = [svar / ovar for svar, ovar in zip(self.data, other.data)]
                return new_field
            elif isinstance(other, (list, np.ndarray)) and len(other) == len(self):
                new_field.data = [svar / o for svar, o in zip(self.data, other)]
                return new_field

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            new_field = copy.deepcopy(self)
            new_field.data = [svar.__rtruediv__(other) for svar in self.data]
            return new_field
        else:
            return Field.__truediv__(other, self)


class SymField:
    def __init__(self, vars=[]):
        self.data = []
        self.domain = Domain()
        self.append(vars)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "[" + ", ".join([str(var) for var in self.data]) + "]"

    def set_domain(self):
        for var in self.data:
            self.domain = self.domain * var.domain

    def evaluate(self, value_dict):
        return [sym_var.evaluate(value_dict) for sym_var in self.data]

    def simplify(self):
        for sym_var in self.data:
            sym_var.simplify()

    def append(self, new_vars):
        if isinstance(new_vars, SymVariable):
            self.data.append(new_vars)
        elif isinstance(new_vars, (int, float)):
            self.append(SymVariable(sym_expression=sympy.sympify(new_vars), evaluation_dict={}, domain=Domain()))
        elif isinstance(new_vars, Variable):
            self.append(SymVariable(*SymVariable.get_init_info_from_variable(new_vars)))
        elif isinstance(new_vars, (Field, SymField)):
            for var in new_vars.data:
                self.append(var)
        elif isinstance(new_vars, list):
            for var in new_vars:
                self.append(var)
        else:
            raise Exception("F should be a list of SymVariables, SymVariable or SymField")

        self.set_domain()
        assert all([isinstance(var, (SymVariable, int, float)) for var in self.data]), "all data should be SymVariables"

    def dot(self, other):
        new_field = self * other
        new_field.set_domain()
        return sum(new_field.data)

    def matmul(self, other):
        if isinstance(other, pd.DataFrame):
            other = other.loc[[str(d) for d in self.data]].values
        if isinstance(other, np.ndarray) and other.shape[0] == len(self):
            new_field = SymField()
            for column_vector in other.T:
                new_field.append(self.dot(column_vector))
            return new_field

    def __mul__(self, other):
        new_field = copy.deepcopy(self)
        if isinstance(other, (int, float)):
            new_field.data = [var * other for var in self.data]
            return new_field
        elif isinstance(other, SymField) and (len(other) == len(self)):
            new_field.data = [svar * ovar for svar, ovar in zip(self.data, other.data)]
            return new_field
        elif isinstance(other, (list, np.ndarray)) and len(other) == len(self):
            new_field.data = [svar * float(o) for svar, o in zip(self.data, other)]
            return new_field

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        new_field = copy.deepcopy(self)
        if isinstance(other, (int, float)):
            new_field.data = [other + var for var in self.data]
            return new_field
        elif isinstance(other, SymField) and (len(other) == len(self)):
            new_field.data = [svar + ovar for svar, ovar in zip(self.data, other.data)]
            return new_field
        elif isinstance(other, (list, np.ndarray)) and len(other) == len(self):
            new_field.data = [svar + o for svar, o in zip(self.data, other)]
            return new_field

    def __sub__(self, other):
        return self + (-1 * other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        else:
            new_field = copy.deepcopy(self)
            if isinstance(other, SymField) and (len(other) == len(self)):
                new_field.data = [svar / ovar for svar, ovar in zip(self.data, other.data)]
                return new_field
            elif isinstance(other, (list, np.ndarray)) and len(other) == len(self):
                new_field.data = [svar / o for svar, o in zip(self.data, other)]
                return new_field

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            new_field = copy.deepcopy(self)
            new_field.data = [svar.__rtruediv__(other) for svar in self.data]
            return new_field
        else:
            return SymField.__truediv__(other, self)

    def __pow__(self, power, modulo=None):
        new_field = copy.deepcopy(self)
        if isinstance(power, (int, float, SymVariable)):
            new_field.data = [var ** power for var in self.data]
            return new_field
        elif isinstance(power, SymField) and (len(power) == len(self)):
            new_field.data = [svar ** pow for svar, pow in zip(self.data, power.data)]
            return new_field
        elif isinstance(power, (list, np.ndarray)) and len(power) == len(self):
            new_field.data = [svar ** o for svar, o in zip(self.data, power)]
            return new_field

    def __rpow__(self, other):
        if isinstance(other, SymField):
            return SymField.__pow__(other, self)
        elif isinstance(other, (int, float, SymVariable)):
            new_field = copy.deepcopy(self)
            new_field.data = [other ** var for var in self.data]
            return new_field
