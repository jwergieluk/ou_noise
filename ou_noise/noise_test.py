import unittest
import math
import random
from typing import Sequence

import numpy
from . import ou, rel_error, ou_linmrl, bm, quadratic_variation
import scipy.stats


def assert_nearly_equal(test_cls: unittest.TestCase, a: float, b: float, error_bound: float, msg: str=''):
    if rel_error(a, b) < error_bound:
        return
    test_cls.fail(f'dist({a}, {b}) > {error_bound} {msg}')


class TestQuadraticVariation(unittest.TestCase):
    def test_quadratic_variation1(self):
        x = bm.path(numpy.arange(0.0, 50.0, 0.01))
        self.assertTrue(rel_error(quadratic_variation(x), 50.0) < 0.05)

    def test_quadratic_variation2(self):
        t = numpy.arange(0, 100, 0.01)
        ref_params = [0.01, 1.0, 1.0]
        x = ou.path(0.0, t, *ref_params)
        self.assertTrue(rel_error(quadratic_variation(x), 100.0) < 0.05)


class TestOu(unittest.TestCase):
    def test_ou_moments(self):
        mean_rev_speed, mean_rev_level, t, x0, vola = 1, 1, 1, 0, 1
        self.assertAlmostEqual(ou.mean(x0, t, mean_rev_speed, mean_rev_level), 1.0 - math.exp(-t))
        self.assertAlmostEqual(ou.variance(t, mean_rev_speed, vola), (1.0 - math.exp(-2)) / (2 * mean_rev_speed))
        self.assertAlmostEqual(math.log(ou.pdf(x0, t, mean_rev_speed, mean_rev_level, vola, 0.1)),
                               ou.logpdf(x0, t, mean_rev_speed, mean_rev_level, vola, 0.1))

    def test_ou_mle1(self):
        t = numpy.arange(0, 100, 0.01)
        ref_params = [2.0, 0.5, 0.05]
        x = ou.path(5.0, t, *ref_params)
        params = ou.mle(t, x)
        self.assertIsNotNone(params)
        self.assertEqual(len(params), 3)
        for i in range(len(params)):
            self.assertTrue(rel_error(ref_params[i], params[i]) < 0.05)

    def test_est_v(self):
        t = numpy.arange(0, 100, 0.01)
        ref_params = [0.1, 0.5, 0.05]
        x = ou.path(5.0, t, *ref_params)
        v = ou.est_v_quadratic_variation(t, x)
        self.assertTrue(rel_error(v, 0.05) < 0.05)


class TestOuLinMrl(unittest.TestCase):
    def test_moments1(self):
        mean_rev_speed, mean_rev_level, t, x0, vola = 1, 1, 1, 0, 1
        # xs, s, t, a: float, b0: float, b1: float
        self.assertAlmostEqual(ou_linmrl.mean(x0, 0.0, t, mean_rev_speed, mean_rev_level, 0.0), 1.0 - math.exp(-t))

    def test_moments2(self):
        for i in range(100):
            mean_rev_speed = random.expovariate(1.0)
            mean_rev_level = random.normalvariate(0.0, 1.0)
            t = random.expovariate(1.0)
            x0 = random.normalvariate(0.0, 1.0)
            # xs, s, t, a: float, b0: float, b1: float
            m1 = ou_linmrl.mean(x0, 0.0, t, mean_rev_speed, mean_rev_level, 0.0)
            m2 = ou.mean(x0, t, mean_rev_speed, mean_rev_level)
            self.assertAlmostEqual(m1, m2)

    def test_ou_linmrl_mle1(self):
        t = numpy.arange(0, 100, 0.01)
        ref_params = (0.2, 15.0, 0.2, 0.5)
        x = ou_linmrl.path(12.0, t, *ref_params)
        # plt.plot(t, x)
        # plt.savefig('/tmp/ou_path.pdf')
        # plt.close()
        params = ou_linmrl.mle(t, x)
        self.assertIsNotNone(params)
        self.assertEqual(len(params), 4)

        # this is crazy
        assert_nearly_equal(self, ref_params[0], params[0], 1.0)
        assert_nearly_equal(self, ref_params[1], params[1], 1.0)
        assert_nearly_equal(self, ref_params[2], params[2], 0.25)
        assert_nearly_equal(self, ref_params[3], params[3], 0.05)

    @unittest.skip
    def test_ou_linmrl_mle_qv1(self):
        t = numpy.arange(0, 1000, 0.01)
        ref_params = (2.0, 0.5, 0.2, 5.0)
        x = ou_linmrl.path(0.5, t, *ref_params)
        params = ou_linmrl.mle(t, x)
        self.assertIsNotNone(params)
        self.assertEqual(len(params), 4)

        self.assertTrue(rel_error(ref_params[0], params[0]) < 0.1)
        # self.assertTrue(rel_diff(ref_params[1], params[1]) < 0.2)
        self.assertTrue(rel_error(ref_params[2], params[2]) < 0.1)
        self.assertTrue(rel_error(ref_params[3], params[3]) < 0.01)

    def test_drift_loglik_grad1(self):
        t = numpy.arange(0, 5.0, 0.01)
        dt = numpy.diff(t)
        ref_params = numpy.array([0.2, 15.0, 0.2, 0.5])
        x = ou_linmrl.path(12.0, t, *ref_params)
        weights = numpy.repeat(1.0/(len(t)-1), len(t)-1)

        def f(theta: Sequence[float]) -> float:
            mu = ou_linmrl.mean(x[:-1], t[:-1], t[1:], theta[0], theta[1], math.tan(theta[2]))
            sigma = ou.std(dt, theta[0], theta[3])
            return float(-numpy.sum(weights * scipy.stats.norm.logpdf(x[1:], loc=mu, scale=sigma)))

        df = ou_linmrl.drift_loglik_grad(t, x, weights, ref_params[3], 1.0)
        g = df(ref_params)
        eps = 1e-8
        e0, e1, e2 = [numpy.array([eps, 0.0, 0.0, 0.0]),
                      numpy.array([0.0, eps, 0.0, 0.0]),
                      numpy.array([0.0, 0.0, eps, 0.0])]
        g_approx = [(f(ref_params + e0) - f(ref_params)) / eps,
                    (f(ref_params + e1) - f(ref_params)) / eps,
                    (f(ref_params + e2) - f(ref_params)) / eps]

        # assert_nearly_equal(self, g[0], g_approx[0], eps)
        # assert_nearly_equal(self, g[1], g_approx[1], eps)
        # assert_nearly_equal(self, g[2], g_approx[2], eps)

