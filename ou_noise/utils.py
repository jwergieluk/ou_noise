import math
import numpy
import scipy.stats


def quadratic_variation(x, weights=None):
    """ Realized quadratic variation of a path. The weights must sum up to one. """
    assert len(x) > 1
    dx = numpy.diff(x)
    if weights is None:
        return numpy.sum(dx*dx)
    return len(x)*numpy.sum(dx * dx * weights)


def gaussian_path(x0, t, loc_fun, scale_fun, params):
    assert len(t) > 1
    x = scipy.stats.norm.rvs(size=len(t))
    x[0] = x0
    dt = numpy.diff(t)
    scale = scale_fun(dt, params)
    x[1:] = x[1:] * scale
    for i in range(1, len(x)):
        x[i] += loc_fun(x[i - 1], dt[i - 1], params)
    return x


def rel_error(a, b, abs_diff_thd=1e-8):
    if math.fabs(b) < abs_diff_thd:
        return math.fabs(a - b)
    return math.fabs((a-b)/b)

