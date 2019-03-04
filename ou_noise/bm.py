""" Standard Brownian motion """

import numpy
import scipy.stats


def path(t, x0=0.0):
    """ Simulates a sample path"""
    assert len(t) > 1
    x = scipy.stats.norm.rvs(size=len(t))
    x[0] = x0
    dt = numpy.diff(t)
    x[1:] = x[1:] * numpy.sqrt(dt)
    return numpy.cumsum(x)


