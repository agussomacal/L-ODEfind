from itertools import product
from collections import OrderedDict
import sympy
import numpy as np
import types


def get_symfunc_from_name_domain(func_name, domain):
    if len(domain) == 0:
        return sympy.Function(func_name)
    else:
        domain_symbols = sympy.symbols(" ".join(sorted(domain.axis_names)))
        if type(domain_symbols) != tuple:
            domain_symbols = [domain_symbols]
        return sympy.Function(func_name)(*domain_symbols)


def polynomial_order_cartprod(domain_depth_dict):
    """
    TODO: is not necessary the cart prod, we can go through the graph given de root and addin 1 in a DFS.
    Given a derivative depth for each domain variable it calculates the graph of derivations that should be done
    in order to get all the partial derivatives.
    :param domain_depth_dict: ej: {"t": 2, "x":2} -> means up to order 2 in derivatives of t and x
    :return:
    """
    axis_names = list(domain_depth_dict.keys())
    cart_prod_dict = {cart_prod_i: {} for cart_prod_i in product(*[list(range(depth + 1))
                                                                   for depth in domain_depth_dict.values()])}
    cart_prod_dict = OrderedDict(cart_prod_dict)
    for i, cart_prod_i in enumerate(cart_prod_dict.keys()):
        for j in range(i + 1, len(cart_prod_dict)):
            cart_prod_j = list(cart_prod_dict.keys())[j]
            difference = np.array(cart_prod_j) - np.array(cart_prod_i)
            if np.sum(np.abs(difference)) == 1:
                if np.sum(difference) == 1:  # means that j is 1 derivative ahead of i
                    axis_name = axis_names[np.where(difference == 1)[0][0]]
                    cart_prod_dict[cart_prod_i][cart_prod_j] = axis_name  # False because is not calculated
                elif np.sum(difference) == -1:  # means that i is 1 derivative ahead of j
                    axis_name = axis_names[np.where(difference == -1)[0][0]]
                    cart_prod_dict[cart_prod_j][cart_prod_i] = axis_name

    # order from less number of variables to more
    cart_prod_dict = OrderedDict(sorted(cart_prod_dict.items(), key=lambda cp_element: np.sum(cp_element[0])))
    return cart_prod_dict


def dfs_polynomial_order(num_variables, polynomial_order):
    """
    TODO: is not necessary the cart prod, we can go through the graph given de root and addin 1 in a DFS.
    Given a derivative depth for each domain variable it calculates the graph of derivations that should be done
    in order to get all the partial derivatives.
    :param domain_depth_dict: ej: {"t": 2, "x":2} -> means up to order 2 in derivatives of t and x
    :return:
    """
    root_name = [0] * num_variables
    queue = [root_name]

    for element in queue:
        if sum(element) < polynomial_order:
            for i in range(num_variables):
                new_element = element[:]
                new_element[i] += 1
                if new_element not in queue:
                    queue.append(new_element)
                    yield (tuple(element), i, tuple(new_element))


def merge_dicts(dict1, dict2, merge_func):
    """
    joins 2 dicts using merge_func to decide when the keys exists in both
    :return:
    """
    # return {k: merge_func(i for i in (dict1.get(k), dict2.get(k)) if i is not None) for k in dict1.keys() | dict2}
    # return {k: merge_func(i for i in (dict1.get(k), dict2.get(k)) if i is not None) for k in dict1.keys() | dict2.keys()}
    return {k: merge_func(i for i in (dict1.get(k), dict2.get(k)) if i is not None) for k in
            set(dict1.keys()).union(set(dict2.keys()))}


def copy_func(f, name=None):
    return types.FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)


def get_atoms(sym_expression):
    func_atoms_set = sym_expression.atoms(sympy.Function)  # gets the functions present in sym_expression
    # gets all the derivatives of the function present in the expression
    diff_func_atoms_set = set().union(*[sym_expression.atoms(f.diff()) for f in func_atoms_set])
    # sorts the atoms from
    der_atoms = list(reversed(sorted([a for a in diff_func_atoms_set], key=lambda x: len(x.variables))))
    func_atoms = list(func_atoms_set)
    return der_atoms, func_atoms


def get_func_for_ode(sym_x_expression_list, sym_y_expression_list, regressors):
    symbols = list(set().union(
        *[sym_var.sym_expression.free_symbols for sym_var in sym_x_expression_list.data + sym_y_expression_list.data]))
    # symbols += list(set().union(*[sym_var.sym_expression.free_symbols for sym_var in sym_y_expression_list.data]))
    assert len(symbols) == 1, "Only available for 1 dimensional domains: symbols={}".format(symbols)
    symbol = symbols[0]

    # associate the in-derivatives with the out-derivatives shifting 1 place.
    # derivatives_list = [get_sorted_derivative_atoms(sym_exp) for sym_exp in sym_x_expression_list]
    f2xindex_dict = OrderedDict()
    # for i, f in enumerate(derivatives_list):
    #     f2xindex_dict[f] = i

    for symy in sym_y_expression_list.data:
        f = symy.sym_expression.atoms(sympy.Function).pop()
        f2xindex_dict[f] = len(f2xindex_dict)

        der_atoms, func_atoms = get_atoms(symy.sym_expression)
        max_derivative = max([len(it.variables) for it in der_atoms])
        for i in range(1, max_derivative):
            f2xindex_dict[f.diff(symbol, i)] = len(f2xindex_dict)
    # sort from derivatives to X(t) otherwise the substitution breaks.
    f2xindex_dict = OrderedDict(reversed(f2xindex_dict.items()))

    def func(x, t):
        x = np.array(x).ravel()
        res = [np.nan] * len(x)

        # replace the already known derivatives by shifting
        for f, i in f2xindex_dict.items():
            der_f = f.diff()
            if der_f in f2xindex_dict.keys():
                res[i] = x[f2xindex_dict[der_f]]

        # replace the last derivative with the equation
        for ix, sym_exp in zip(np.where(np.isnan(res))[0], sym_x_expression_list.data):
            res[ix] = sym_exp.sym_expression
            res[ix] = res[ix].subs({reg.name: reg.eval({str(symbol): t}) for reg in regressors.data})

            for f, i in f2xindex_dict.items():
                res[ix] = res[ix].subs({f: x[i]})
            # TODO: assumes that domain values in t are in the same order that the free symbols
            # TODO: change symbol -> s (symbol name)
            res[ix] = res[ix].subs({symbol: t_val for t_val, symbol in zip([t], symbols)})
            # res[ix] = np.nan
            res[ix] = float(res[ix])
        return res

    return func

# ------------------------------------------------------------
# Attempt to generalize the prediction for PDE
# ------------------------------------------------------------

def get_lag_from_sym_expression(sym_expression):
    """
    Aux function (Attempt to generalize the prediction for PDE).

    """
    value_dict = {symbol.name: 0 for symbol in sym_expression.free_symbols}
    forward_lag = {ax_name: 0 for ax_name in value_dict.keys()}
    backward_lag = {ax_name: 0 for ax_name in value_dict.keys()}

    # sym_expression.subs(value_dict)
    for f in sym_expression.atoms(sympy.Function):
        f_domain = f.free_symbols
        # substitutes the values dict in the function and given the hidden shifts it may have gets the real domain
        # points where to evaluate.
        domain_order = OrderedDict([(str(dom_var), str(f).find(str(dom_var))) for dom_var in f_domain])
        domain_order = [k for k, v in sorted(domain_order.items(), key=lambda dp: dp[1])]
        domain_values = list(map(int, str(f.subs(value_dict)).split("(")[1].split(")")[0].split(",")))
        for domain_var_name, value in zip(domain_order, domain_values):
            if forward_lag[domain_var_name] < value:
                forward_lag[domain_var_name] = value
            if backward_lag[domain_var_name] > value:
                backward_lag[domain_var_name] = value

    return backward_lag, forward_lag


def get_func_for_pde(integration_variable_name, sym_x_expression_list, sym_y_expression_list, regressors):
    """
    Attempt to generalize the prediction for PDE.
    Not working.
    """
    eq = sym_y_expression_list - sym_x_expression_list
    symbols = list(set().union(*[sym_var.sym_expression.free_symbols for sym_var in eq.data]))
    lag = max([get_lag_from_sym_expression(sym_var)[integration_variable_name] for sym_var in eq])

    # associate the in-derivatives with the out-derivatives shifting 1 place.
    # derivatives_list = [get_sorted_derivative_atoms(sym_exp) for sym_exp in sym_x_expression_list]
    f2xindex_dict = OrderedDict()
    # for i, f in enumerate(derivatives_list):
    #     f2xindex_dict[f] = i

    for symy in sym_y_expression_list.data:
        f = symy.sym_expression.atoms(sympy.Function).pop()
        f2xindex_dict[f] = len(f2xindex_dict)

        der_atoms, func_atoms = get_atoms(symy.sym_expression)
        max_derivative = max([len(it.variables) for it in der_atoms])
        for i in range(1, max_derivative):
            f2xindex_dict[f.diff(symbol, i)] = len(f2xindex_dict)
    # sort from derivatives to X(t) otherwise the substitution breaks.
    f2xindex_dict = OrderedDict(reversed(f2xindex_dict.items()))

    def func(data0, boundary, integration_steps):  # data0: Variable
        data_shape = data0.shape
        data_shape[data0.domain2axis[integration_variable_name]] = data0.shape + integration_steps
        data = np.zeros(data_shape)

        data[np.indices(data0.shape)] = data0.data
        # for i in range(integration_steps):

        return func

