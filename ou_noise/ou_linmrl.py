""" OU with linear mean reversion level

    dX = -a(X-(b0 + b1*t))dt + v dB
    """
import math
from typing import Sequence, Union

import numpy
import scipy.optimize
import scipy.stats
from . import ou


FloatVec = Union[float, Sequence[float]]


def path(x0: float, t: FloatVec, a: float, b0: float, b1: float, v: float):
    """ Simulates a path"""
    assert len(t) > 1
    x = scipy.stats.norm.rvs(size=len(t))
    x[0] = x0
    dt = numpy.diff(t)
    scale = ou.std(dt, a, v)
    x[1:] = x[1:] * scale
    for i in range(1, len(x)):
        x[i] += mean(x[i - 1], t[i - 1], t[i], a, b0, b1)
    return x


def mle(t: FloatVec, x: FloatVec):
    """ Maximum-likelihood estimator """
    v = ou.est_v_quadratic_variation(t, x)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t, x)
    dt = numpy.diff(t)

    def error_fun1(theta_0):
        mu = mean(x[:-1], t[:-1], t[1:], theta_0, intercept, slope)
        sigma = ou.std(dt, theta_0, v)
        return -numpy.sum(scipy.stats.norm.logpdf(x[1:], loc=mu, scale=sigma))

    result = scipy.optimize.minimize(error_fun1, numpy.array([0.5, ]), method='L-BFGS-B', bounds=[(1e-8, None), ],
                                     options={'maxiter': 150, 'disp': False})
    if not result.success:
        return None
    a = result.x[0]

    def error_fun(theta):
        mu = mean(x[:-1], t[:-1], t[1:], theta[0], theta[1], theta[2])
        sigma = ou.std(dt, theta[0], theta[3])
        return -numpy.sum(scipy.stats.norm.logpdf(x[1:], loc=mu, scale=sigma))

    angle = math.atan(slope)
    bounds = [(0.5*a, 2.0*a), (None, None), (math.tan(angle - 0.1), math.tan(angle + 0.1)), (0.5 * v, v * 2.0)]
    start = numpy.array([a, intercept, slope, v])

    result = scipy.optimize.minimize(error_fun, start, method='L-BFGS-B', bounds=bounds,
                                     options={'maxiter': 500, 'disp': False})
    if not result.success:
        return None
    return result.x


def est_mle_qv(t: FloatVec, x: FloatVec):
    dt = numpy.diff(t)
    v = ou.est_v_quadratic_variation(t, x)

    def error_fuc(theta):
        mu = mean(x[:-1], t[:-1], t[1:], theta[0], theta[1], theta[2])
        sigma = ou.std(dt, theta[0], v)
        return numpy.sum(scipy.stats.norm.logpdf(x[1:], loc=mu, scale=sigma))

    result = scipy.optimize.minimize(error_fuc, numpy.array([0.5, 0.0, 0.5]), method='L-BFGS-B',
                                     bounds=[(1e-08, None), (None, None), (None, None)],
                                     options={'maxiter': 500, 'disp': False})
    return result.x


def drift_loglik_grad(t, x, weights, v, s0: float = 1.0):
    dt = numpy.diff(t)

    def df(theta: Sequence[float]) -> Sequence[float]:
        a, b0, b1 = theta[:3]
        mu = mean(x[:-1], t[:-1], t[1:], a, s0 * b0, math.tan(b1))
        sigma = ou.std(dt, a, v)

        f = x[1:] - mu
        fda1 = -dt * x[:-1] * numpy.exp(-a * dt)
        fda2 = (1.0 - numpy.exp(-a * dt)) * 2.0 * math.tan(b1) / (a*a)
        fda3 = (s0*b0 - numpy.tan(b1)/a)*dt*numpy.exp(-a*dt)
        fda4 = dt*t[:-1]*numpy.exp(-a*dt)*math.tan(b1)
        fda = fda1 + fda2 + fda3 + fda4
        fdb0 = (1.0 - numpy.exp(-a*dt))*s0
        db1 = 1.0/(numpy.cos(b1)*numpy.cos(b1))
        fdb1 = -db1*(1.0 - numpy.exp(-a*dt))/a + db1*(t[1:] - t[:-1]*numpy.exp(-a*dt))

        nu = v * v * (1.0 - numpy.exp(-2.0 * a * dt)) / (2.0 * a)
        nuda = (v*v/(2.0*a*a))*(2.0*a*dt*numpy.exp(-2.0*a*dt) - (1.0 - numpy.exp(-2.0*a*dt)))

        gda = (fda*sigma - f*nuda)/(sigma*sigma)
        gdb0 = fdb0/sigma
        gdb1 = fdb1/sigma

        grad1 = -sum(weights * gda * (x[1:] - mu) / sigma)
        grad2 = -sum(weights * gdb0 * (x[1:] - mu) / sigma)
        grad3 = -sum(weights * gdb1 * (x[1:] - mu) / sigma)

        return [grad1, grad2, grad3]
    return df


def loglik(t: FloatVec, x: FloatVec, a: float, b0: float, b1: float, v: float):
    dt = numpy.diff(t)
    mu = mean(x[:-1], t[:-1], t[1:], a, b0, b1)
    sigma = ou.std(dt, a, v)
    return numpy.sum(scipy.stats.norm.logpdf(x[1:], loc=mu, scale=sigma))


def mean(xs: FloatVec, s: FloatVec, t: FloatVec, a: float, b0: float, b1: float):
    assert a >= 0
    p1 = xs * numpy.exp(-a * (t - s))
    p2 = (b0 - b1 / a) * (1.0 - numpy.exp(-a * (t - s))) + b1 * (t - numpy.exp(-a * (t - s)) * s)
    return p1 + p2


def std(dt: FloatVec, a: float, v: float):
    return ou.std(dt, a, v)
