import numpy as np


def error(yhat, y, axis=0):
    return np.sqrt(((yhat - y) ** 2).sum(axis))


def rsquare(yhat, y, axis=0):
    try:
        return 1 - ((yhat - y) ** 2).sum(axis) / ((y - y.mean(axis)) ** 2).sum(axis)
    except:
        return np.nan


def smape_loss(p, t):
    return np.mean(np.abs(p - t) / (np.abs(t) + np.abs(p))) * 2