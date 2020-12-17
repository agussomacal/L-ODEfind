import numpy as np
import copy
import sympy
from itertools import chain

from src.lib.variables import Variable, SymVariable, Field, SymField, Domain
from src.lib.algorithms import polynomial_order_cartprod, dfs_polynomial_order


##########################################################
#                       Operator
##########################################################
class Operator:
    def __init__(self, backward_lag=0, forward_lag=0):
        self.forward_lag = forward_lag
        self.backward_lag = backward_lag

    def var_operator_func(self, var):
        """

        :type var: Variable
        :return:
        """
        raise Exception('var_operator_func not implemented')

    def sym_var_operator_func(self, sym_var):
        """

        :type sym_var: SymVariable
        """
        raise Exception('sym_var_operator_func not implemented')

    def domain_operator_func(self, domain):
        """

        :type sym_var: SymVariable
        """
        raise Exception('domain_operator_func not implemented')

    def __mul__(self, other):
        if isinstance(other, Variable):
            return self.var_operator_func(other)
        elif isinstance(other, SymVariable):
            return self.sym_var_operator_func(other)
        elif isinstance(other, Field):
            return Field([self * var for var in other.data])
        elif isinstance(other, SymField):
            return SymField([self * sym_var for sym_var in other.data])
        elif isinstance(other, Domain):
            return self.domain_operator_func(other)
        elif isinstance(other, Operator):
            o = Operator()
            o.var_operator_func = lambda var: self.var_operator_func(other.var_operator_func(var))
            o.sym_var_operator_func = lambda sym_var: self.sym_var_operator_func(other.sym_var_operator_func(sym_var))
            o.backward_lag = other.backward_lag + self.backward_lag  # TODO: not necessarily will always work, rethink
            o.forward_lag = other.forward_lag + self.forward_lag
            return o


##########################################################
#                       Operator
##########################################################
class Identity(Operator):
    def var_operator_func(self, var):
        """

        :type var: Variable
        :return:
        """
        return var

    def sym_var_operator_func(self, sym_var):
        """

        :type sym_var: SymVariable
        """
        return sym_var

    def domain_operator_func(self, domain):
        """

        :type sym_var: SymVariable
        """
        return domain

    def __init__(self):
        Operator.__init__(self)


##########################################################
#                       Delay
##########################################################
class Delay(Operator):
    def __init__(self, delay, axis_name):
        """
        :param difference_order:
        :param axis_name:
        """
        Operator.__init__(self, backward_lag=np.max((0, delay)), forward_lag=-np.min((0, delay)))
        self.axis_name = axis_name
        self.delay = delay

    def var_operator_func(self, var):
        """

        :type var: Variable
        """
        # TODO: roll should put nan or something in the border
        return Variable(data=np.roll(var.data, shift=self.delay, axis=var.get_axis(self.axis_name)),
                        domain=var.domain,
                        domain2axis=var.domain2axis,
                        variable_name=var.name.subs(sympy.Symbol(self.axis_name),
                                                    sympy.Symbol(self.axis_name) - self.delay))

    def sym_var_operator_func(self, sym_var):
        return SymVariable.shift(sym_var, {self.axis_name: -self.delay})

    def __mul__(self, other):
        if isinstance(other, Delay) and other.axis_name == self.axis_name:
            return Delay(self.delay + other.delay, self.axis_name)
        else:
            return Operator.__mul__(self, other)


##########################################################
#                       MultipleDelay
##########################################################
class MultipleDelay(Operator):
    """
    Multiple differentiations returned as a Field where each coord represents one derivative of the derivative tree
    """

    def __init__(self, delays_dict):
        if isinstance(delays_dict, dict):
            self.delays_dict = delays_dict
        Operator.__init__(self, backward_lag=np.max([0] + list(delays_dict.values())),
                          forward_lag=-np.min([0] + list(delays_dict.values())))

    def successive_multiple_delays(self, other):
        if isinstance(other, (Variable)):
            temp_other = Field([other])
        elif isinstance(other, SymVariable):
            temp_other = SymField([other])
        else:
            temp_other = other

        delayed_vars = []
        for ax_name, delays_list in self.delays_dict.items():
            for var in temp_other.data:
                for delay in delays_list:
                    delayed_vars.append(Delay(axis_name=ax_name, delay=delay) * var)

        return delayed_vars

    def var_operator_func(self, var):
        return Field(self.successive_multiple_delays(var))

    def sym_var_operator_func(self, sym_var):
        return SymField(self.successive_multiple_delays(sym_var))


##########################################################
#                       Diff
##########################################################
class Diff(Operator):
    def __init__(self, difference_order, axis_name):
        """
        TODO: derivatives order 2 and 1 differently so it doesent grows 2 steps on each derivative. f(t+1) - f(t-1)
        :param difference_order:
        :param axis_name:
        """
        Operator.__init__(self, backward_lag=difference_order, forward_lag=difference_order)
        self.axis_name = axis_name
        self.difference_order = difference_order

    def difference(self, var):
        """

        :type var: Variable
        :param var:
        :return:
        """
        # axis = var.get_axis(self.axis_name)
        # np.take(np.diff(var.data, axis=axis), np.arange(var.data.shape[axis] + 1), axis=axis, mode='clip')
        # diff = np.diff(var.data, axis=axis)
        # diff = np.concatenate([diff, np.take(diff, [-1], axis=axis)], axis=axis)
        # return Variable(data=diff,
        #                 domain=var.domain,
        #                 domain2axis=var.domain2axis,
        #                 variable_name=var.name.diff(sympy.Symbol(self.axis_name), 1)) # TODO: it is not the derivative in the name, but for now is easier this way
        #
        return Variable(data=np.gradient(var.data, axis=var.get_axis(self.axis_name)),
                        domain=var.domain,
                        domain2axis=var.domain2axis,
                        variable_name=var.name.diff(sympy.Symbol(self.axis_name), 1) *
                                      (var.domain.step_width[self.axis_name]))
        # TODO: it is not the derivative in the name, but for now is easier this way

    def sym_difference(self, sym_var):
        """

        :type sym_var: SymVariable
        """
        # return SymVariable.shift(sym_var, {self.axis_name: 1}) - sym_var
        return (SymVariable.shift(sym_var, {self.axis_name: 1}) - SymVariable.shift(sym_var,
                                                                                    {self.axis_name: -1})).__truediv__(
            2.0)

    def var_operator_func(self, var):
        """

        :type var: Variable
        """
        temp_var = copy.deepcopy(var)
        for i in range(self.difference_order):
            temp_var = self.difference(temp_var)
        return temp_var

    def sym_var_operator_func(self, sym_var):
        temp_sym_var = copy.deepcopy(sym_var)
        for i in range(self.difference_order):
            temp_sym_var = self.sym_difference(temp_sym_var)
        return temp_sym_var

    def __mul__(self, other):
        if isinstance(other, Diff) and other.axis_name == self.axis_name:
            return Diff(self.difference_order + other.difference_order, self.axis_name)
        else:
            return Operator.__mul__(self, other)


##########################################################
#                       D
##########################################################
class D(Operator):
    def __init__(self, derivative_order, axis_name):
        Operator.__init__(self, backward_lag=derivative_order, forward_lag=derivative_order)
        self.axis_name = axis_name
        self.derivative_order = derivative_order

    def var_operator_func(self, var):
        # return Diff(self.derivative_order, self.axis_name).var_operator_func(var).__truediv__(
        #    float((var.domain.step_width[self.axis_name]) ** self.derivative_order))
        return Diff(self.derivative_order, self.axis_name).var_operator_func(var).__truediv__(
            float((var.domain.step_width[self.axis_name]) ** self.derivative_order))

    def sym_var_operator_func(self, sym_var):
        # return Diff(self.derivative_order, self.axis_name).sym_var_operator_func(sym_var).__truediv__(
        #       float((sym_var.domain.step_width[self.axis_name]) ** self.derivative_order))
        return Diff(self.derivative_order, self.axis_name).sym_var_operator_func(sym_var).__truediv__(
            float((sym_var.domain.step_width[self.axis_name]) ** self.derivative_order))

    def __mul__(self, other):
        if isinstance(other, D) and other.axis_name == self.axis_name:
            return D(self.derivative_order + other.derivative_order, self.axis_name)
        else:
            return Operator.__mul__(self, other)


##########################################################
#                       PolyD
##########################################################
class PolyD(Operator):
    """
    Multiple differentiations returned as a Field where each coord represents one derivative of the derivative tree
    """

    def __init__(self, derivative_order_dict):
        if isinstance(derivative_order_dict, dict):
            self.derivative_order_dict = derivative_order_dict
        lag = np.max([0] + list(derivative_order_dict.values()))
        Operator.__init__(self, backward_lag=lag, forward_lag=lag)

    def successive_multiple_derivatives(self, other):
        cart_prod_dict = polynomial_order_cartprod(self.derivative_order_dict)  # ordered dict from root to border

        root_name = tuple([0] * len(self.derivative_order_dict))
        derivatives = {node_name: None for node_name in cart_prod_dict.keys()}
        derivatives[root_name] = other  # no derivatives involved.
        for node_name, neighbours in cart_prod_dict.items():
            for neighbour_name, axis_name in neighbours.items():
                if derivatives[neighbour_name] is None:
                    derivatives[neighbour_name] = D(derivative_order=1, axis_name=axis_name) * derivatives[node_name]

        # TODO: shall we use Field for Variables and SymVariables?
        return list(derivatives.values())

    def var_operator_func(self, var):
        return Field(self.successive_multiple_derivatives(var))

    def sym_var_operator_func(self, sym_var):
        return SymField(self.successive_multiple_derivatives(sym_var))

    def __mul__(self, other):
        if isinstance(other, PolyD):
            # to get the maximum order for each axis
            derivative_order_dict = dict(sorted(chain(self.derivative_order_dict.items(),
                                                      other.derivative_order_dict.items()), key=lambda t: t[1]))
            return PolyD(derivative_order_dict)
        else:
            return Operator.__mul__(self, other)


##########################################################
#                       Poly
##########################################################
class Poly(Operator):
    """
    Given a Field, returns multiple multiplications over each coordinate [f, g] -> [1, f, f**2, f*g, g, g**2]
    """

    def __init__(self, polynomial_order):
        Operator.__init__(self)
        self.polynomial_order = polynomial_order

    def successive_multiplications(self, other):
        assert isinstance(other, (tuple, list))
        polynomial_vars_dict = {tuple([0] * len(other)): 1}
        for father_node, index_2_multiply, son_node in dfs_polynomial_order(len(other), self.polynomial_order):
            polynomial_vars_dict[son_node] = polynomial_vars_dict[father_node] * other[index_2_multiply]

        return list(polynomial_vars_dict.values())

    def var_operator_func(self, var):
        return Field(self.successive_multiplications([var]))

    def sym_var_operator_func(self, sym_var):
        return SymField(self.successive_multiplications([sym_var]))

    def __mul__(self, other):
        if isinstance(other, Poly):
            return Poly(other.polynomial_order + self.polynomial_order)

        # in this case the operation over fields is not element by element; it meshes all so it should change.
        elif isinstance(other, Field):
            return Field(self.successive_multiplications(other.data))
        elif isinstance(other, SymField):
            return SymField(self.successive_multiplications(other.data))
        else:
            return Operator.__mul__(self, other)


##########################################################
#                     DataSplit
##########################################################
class DataSplit(Operator):
    """
    Given a Field, or a Variable, returns the splited domain
    """

    def __init__(self, axis_percentage_dict, axis_init_percentage_dict={}):
        """

        :param axis_percentage_dict: percentage of contiguous data to take for each axis.
        :param axis_init_percentage_dict: percentage of the data where to init the sequence spliting
        """
        Operator.__init__(self)
        self.axis_percentage_dict = axis_percentage_dict
        self.axis_init_percentage_dict = axis_init_percentage_dict

    def get_subdomain_ranges(self, domain):
        axis_init_percentage_dict = self.axis_init_percentage_dict.copy()
        domain_index_sub_range_dict = {}
        for ax_name, shape in domain.shape.items():
            if ax_name not in axis_init_percentage_dict.keys():
                axis_init_percentage_dict[ax_name] = 0
            # if nothing is said about this dimension then all is used
            if ax_name not in self.axis_percentage_dict.keys():
                axis_percentage_dict = 1
            else:
                axis_percentage_dict = self.axis_percentage_dict[ax_name]
            domain_index_sub_range_dict[ax_name] = [int(axis_init_percentage_dict[ax_name] * shape),
                                                    int((axis_init_percentage_dict[ax_name] + axis_percentage_dict) * shape)]
        return domain_index_sub_range_dict

    def var_operator_func(self, var):
        return var.get_subset_from_index_limits(self.get_subdomain_ranges(var.domain))

    def sym_var_operator_func(self, sym_var):
        return sym_var.get_subset_from_index_limits(self.get_subdomain_ranges(sym_var.domain))

    def domain_operator_func(self, domain):
        return domain.get_subdomain_from_index(self.get_subdomain_ranges(domain))


##########################################################
#                 DataSplitOnIndex
##########################################################
class DataSplitOnIndex(Operator):
    """
    Given a Field, or a Variable, returns the splited domain
    """

    def __init__(self, axis_index_dict):
        """

        :param axis_index_dict: index of last domain point to take for each axis from data.
        """
        Operator.__init__(self)
        self.axis_index_dict = axis_index_dict

    def get_subdomain_ranges(self, var):
        return {ax_name: [0, self.axis_index_dict[ax_name]] for ax_name, shape in var.domain.shape.items()}

    def var_operator_func(self, var):
        return var.get_subset_from_index_limits(self.get_subdomain_ranges(var))

    def sym_var_operator_func(self, sym_var):
        return sym_var.get_subset_from_index_limits(self.get_subdomain_ranges(sym_var))


##########################################################
#                 DataSplitOnIndex
##########################################################
class DataSplitIndexClip(Operator):
    """
    Given a Field, or a Variable, returns the splited domain
    """

    def __init__(self, axis_start_dict=None, axis_end_dict=None, axis_len_dict=None):
        """

        :param axis_index_dict: index of last domain point to take for each axis from data.
        """
        Operator.__init__(self)
        assert sum([axis_start_dict is None, axis_end_dict is None, axis_len_dict is None]) == 1, \
            "axis_start_dict=None, axis_end_dict=None, axis_len_dict=None 2 should be not None"
        if axis_len_dict is not None:
            assert all([v >= 0 for v in axis_len_dict.values()]), 'len should be positive'
            if axis_start_dict is None:
                axis_start_dict = {ax_name: axis_end_dict[ax_name] - axis_len_dict[ax_name]
                                   for ax_name in axis_end_dict.keys()}
            if axis_end_dict is None:
                axis_end_dict = {ax_name: axis_start_dict[ax_name] + axis_len_dict[ax_name]
                                 for ax_name in axis_start_dict.keys()}
            self.axis_len_dict = axis_len_dict
        else:
            self.axis_len_dict = {ax_name: axis_end_dict[ax_name] - axis_start_dict[ax_name] for ax_name in
                                  axis_start_dict.keys()}

        self.axis_start_dict = axis_start_dict
        self.axis_end_dict = axis_end_dict

    def get_subdomain_ranges(self, var):
        subdomain_ranges = {}
        for ax_name, shape in var.domain.shape.items():
            if self.axis_end_dict[ax_name] < 0:
                e = shape + self.axis_end_dict[ax_name] + 1
                s = e - self.axis_len_dict[ax_name]
            elif self.axis_end_dict[ax_name] > shape:
                e = shape
                s = e - self.axis_len_dict[ax_name]
            elif self.axis_start_dict[ax_name] < 0:
                s = 0
                e = s + self.axis_len_dict[ax_name]
            else:
                s = self.axis_start_dict[ax_name]
                e = self.axis_end_dict[ax_name]
            subdomain_ranges[ax_name] = [s, e]

        return subdomain_ranges

    def var_operator_func(self, var):
        return var.get_subset_from_index_limits(self.get_subdomain_ranges(var))

    def sym_var_operator_func(self, sym_var):
        return sym_var.get_subset_from_index_limits(self.get_subdomain_ranges(sym_var))
