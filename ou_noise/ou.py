""" OU (Ornstein-Uhlenbeck) process

    dX = -A(X-alpha)dt + v dB
    """
import math

import scipy.constants
import scipy.stats
import scipy.optimize
import numpy
from . import quadratic_variation


def mle(t, x, start=None):
    """Maximum-likelihood estimator"""

    if start is None:
        v = est_v_quadratic_variation(t, x)
        start = (0.5, numpy.mean(x), v)

    def error_fuc(theta):
        return -loglik(t, x, theta[0], theta[1], theta[2])

    start = numpy.array(start)
    result = scipy.optimize.minimize(error_fuc, start, method='L-BFGS-B',
                                     bounds=[(1e-6, None), (None, None), (1e-8, None)],
                                     options={'maxiter': 500, 'disp': False})
    return result.x


def path(x0, t, mean_rev_speed, mean_rev_level, vola):
    """ Simulates a sample path"""
    assert len(t) > 1
    x = scipy.stats.norm.rvs(size=len(t))
    x[0] = x0
    dt = numpy.diff(t)
    scale = std(dt, mean_rev_speed, vola)
    x[1:] = x[1:] * scale
    for i in range(1, len(x)):
        x[i] += mean(x[i - 1], dt[i - 1], mean_rev_speed, mean_rev_level)
    return x


def pdf(x0, t, mean_rev_speed, mean_rev_level, vola, x):
    mu = mean(x0, t, mean_rev_speed, mean_rev_level)
    sigma = std(t, mean_rev_speed, vola)
    return scipy.stats.norm.pdf(x=x, loc=mu, scale=sigma)


def logpdf(x0, t, mean_rev_speed, mean_rev_level, vola, x):
    mu = mean(x0, t, mean_rev_speed, mean_rev_level)
    sigma = std(t, mean_rev_speed, vola)
    return scipy.stats.norm.logpdf(x=x, loc=mu, scale=sigma)


def loglik(t, x, mean_rev_speed, mean_rev_level, vola):
    """Calculates log likelihood of a path"""
    dt = numpy.diff(t)
    mu = mean(x[:-1], dt, mean_rev_speed, mean_rev_level)
    sigma = std(dt, mean_rev_speed, vola)
    return numpy.sum(scipy.stats.norm.logpdf(x[1:], loc=mu, scale=sigma))


def est_v_quadratic_variation(t, x, weights=None):
    """ Estimate v using quadratic variation"""
    assert len(t) == len(x)
    q = quadratic_variation(x, weights)
    return math.sqrt(q/(t[-1] - t[0]))


def mean(x0, t, mean_rev_speed, mean_rev_level):
    assert mean_rev_speed >= 0
    return x0 * numpy.exp(-mean_rev_speed * t) + (1.0 - numpy.exp(- mean_rev_speed * t)) * mean_rev_level


def ou0_mean(x0, t, mean_rev_speed):
    """ Mean function of an OU process defined as dX = -AXdt + vdB"""
    assert mean_rev_speed >= 0
    return x0 * numpy.exp(-mean_rev_speed * t)


def variance(t, mean_rev_speed, vola):
    assert mean_rev_speed >= 0
    assert vola >= 0
    return vola * vola * (1.0 - numpy.exp(- 2.0 * mean_rev_speed * t)) / (2 * mean_rev_speed)


def std(t, mean_rev_speed, vola):
    return numpy.sqrt(variance(t, mean_rev_speed, vola))
