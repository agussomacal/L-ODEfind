#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import sys
import time
from contextlib import contextmanager

import matplotlib.pylab as plt
import pandas as pd

# from collections import ChainMap
from src.scripts.config import get_filename

src_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(src_path)


def check_create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_create_path(*args):
    """
    Given a list of strings it will check or/and create the full path assuming each name names a folder in the
    given order.

    :param args:
    :return:
    """
    path = ''
    for i, arg in enumerate(args):
        if i > 0:
            path = os.path.join(path, arg)
        else:
            path = arg
        check_create_dir(dir_name=path)
    return path


def varname2latex(var_name, derivative=0):
    new_var_name = var_name
    if derivative > 0:
        if derivative == 1:
            new_var_name = '\\frac{\partial ' + new_var_name + '}{\partial t}'
        else:
            new_var_name = '\\frac{\partial^' + str(derivative) + new_var_name + '}{\partial t^' + str(derivative) + '}'
    return r'{}'.format('$' + new_var_name + '$')


def convert_type(series):
    try:
        return series.astype(float)
    except:
        pass
    try:
        return series.astype(str)
    except:
        pass


# figure saver
@contextmanager
def savefig(figname, experiment, subfolders=[], verbose=False, format='eps', close=True):
    yield
    filename = get_filename(figname, experiment, subfolders)
    if verbose:
        print('Saving in: {}'.format(filename))
    plt.savefig(filename, dpi=500, format=format)
    if close:
        plt.close()


@contextmanager
def savefig2(filename, verbose=False, close=True):
    yield
    if verbose:
        print('Saving in: {}'.format(filename))
    plt.savefig(filename, dpi=500)
    if close:
        plt.close()

def load(path, experiment, subfolders=[]):
    filename = get_filename(path, experiment, subfolders)
    if os.path.exists(filename):
        print('loading ', filename)
        if filename.split['.'][-1] == 'csv':
            data = pd.read_csv(filename)
        elif filename.split['.'][-1] == 'pickle':
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            raise Exception(filename, 'is not pickle or csv')
        return data
    else:
        return False


def save(data, path, experiment, subfolders=[]):
    yield
    filename = get_filename(path, experiment, subfolders)
    print('Saving in: {}'.format(filename))

    if filename.split['.'][-1] == 'csv':
        data.to_csv(filename)
    elif filename.split['.'][-1] == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


@contextmanager
def timeit(msg):
    t0 = time.time()
    yield
    print('Duracion {}: {}'.format(msg, time.time()-t0))