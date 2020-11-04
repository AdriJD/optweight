import unittest
import numpy as np

from pixell import sharp
from enlib import cg

from optweight import solvers
from optweight import alm_utils
from optweight import map_utils

class TestCGWiener(unittest.TestCase):
    
    def test_CGWiener_init(self):
        
        alm_data = np.ones(6)
        icov_signal = lambda alm: alm
        icov_noise = lambda alm: alm

        solver = solvers.CGWiener(alm_data, icov_signal, icov_noise)
        self.assertTrue(hasattr(solver, 'alm_data'))
        self.assertTrue(hasattr(solver, 'icov_signal'))
        self.assertTrue(hasattr(solver, 'icov_noise'))
        self.assertTrue(hasattr(solver, 'beam'))
        self.assertTrue(issubclass(solvers.CGWiener, cg.CG))
        self.assertIs(solver.dot, alm_utils.contract_almxblm)

    def test_CGWiener_init_kwargs(self):
        
        alm_data = np.ones(6)
        icov_signal = lambda alm: alm
        icov_noise = lambda alm: alm

        x0 = np.zeros_like(alm_data)
        M = lambda alm: alm
        dot = lambda a, b: np.sum(a * b)

        solver = solvers.CGWiener(
            alm_data, icov_signal, icov_noise, x0=x0, M=M, dot=dot)
        self.assertIs(solver.dot, dot)
        self.assertIs(solver.M, M)
        np.testing.assert_array_equal(solver.x, x0)

    def test_CGWiener_run(self):
        
        alm_data = np.asarray([1, 2, 3, 4j, 5j, 6])
        icov_signal = lambda alm: 2 * alm
        icov_noise = lambda alm: 3 * alm

        solver = solvers.CGWiener(alm_data, icov_signal, icov_noise)
        b_in = solver.b.copy()
        while solver.i < 1:
            solver.step()

        np.testing.assert_array_almost_equal(solver.x, alm_data * (3 / 5))
        np.testing.assert_array_almost_equal(solver.A(solver.x), b_in)

    def test_CGWiener_run_beam(self):
        
        alm_data = np.asarray([1, 2, 3, 4j, 5j, 6])
        icov_signal = lambda alm: 2 * alm
        icov_noise = lambda alm: 3 * alm
        beam = lambda alm: 0.1 * alm

        solver = solvers.CGWiener(alm_data, icov_signal, icov_noise, beam=beam)
        b_in = solver.b.copy()
        while solver.i < 1:
            solver.step()

        x_exp = alm_data * (0.1 * 3 / (2 + 0.01 * 3))
        np.testing.assert_array_almost_equal(solver.x, x_exp) 
        np.testing.assert_array_almost_equal(solver.A(solver.x), b_in)
    
    def test_CGWiener_constrained_realisation_init(self):
        
        alm_data = np.ones(6)
        icov_signal = lambda alm: alm
        icov_noise = lambda alm: alm
        rand_isignal = np.random.randn(6) 
        rand_inoise = np.random.randn(6)

        solver = solvers.CGWiener(
            alm_data, icov_signal, icov_noise, 
            rand_isignal=rand_isignal, rand_inoise=rand_inoise)

        self.assertTrue(hasattr(solver, 'alm_data'))
        self.assertTrue(hasattr(solver, 'icov_signal'))
        self.assertTrue(hasattr(solver, 'icov_noise'))
        self.assertTrue(hasattr(solver, 'rand_isignal'))
        self.assertTrue(hasattr(solver, 'rand_inoise'))
        self.assertTrue(hasattr(solver, 'beam'))
        self.assertTrue(issubclass(solvers.CGWiener, cg.CG))

    def test_CGWiener_constrained_realisation_run(self):
        
        alm_data = np.ones(6)
        icov_signal = lambda alm: 2 * alm
        icov_noise = lambda alm: 3 * alm
        rand_isignal = np.arange(6, dtype=float)
        rand_inoise = np.arange(6, dtype=float)[::-1]

        solver = solvers.CGWiener(
            alm_data, icov_signal, icov_noise, 
            rand_isignal=rand_isignal, rand_inoise=rand_inoise)

        b_in = solver.b.copy()
        while solver.i < 1:
            solver.step()

        x_exp = (3 * alm_data + rand_isignal + rand_inoise) / (3 + 2)
        np.testing.assert_array_almost_equal(solver.x, x_exp)
        np.testing.assert_array_almost_equal(solver.A(solver.x), b_in)

    def test_CGWiener_constrained_realisation_run_beam(self):
        
        alm_data = np.ones(6)
        icov_signal = lambda alm: 2 * alm
        icov_noise = lambda alm: 3 * alm
        rand_isignal = np.arange(6, dtype=float)
        rand_inoise = np.arange(6, dtype=float)[::-1]
        beam = lambda alm: 0.1 * alm

        solver = solvers.CGWiener(
            alm_data, icov_signal, icov_noise, beam=beam,
            rand_isignal=rand_isignal, rand_inoise=rand_inoise)

        b_in = solver.b.copy()
        while solver.i < 1:
            solver.step()

        x_exp = (0.1 * 3 * alm_data + rand_isignal + 0.1 * rand_inoise) / (0.01 * 3 + 2)
        np.testing.assert_array_almost_equal(solver.x, x_exp)
        np.testing.assert_array_almost_equal(solver.A(solver.x), b_in)

    def test_CGWiener_from_arrays(self):
        
        lmax = 2
        ainfo = sharp.alm_info(lmax=lmax)
        alm_data = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm_data = alm_data[np.newaxis,:]
        
        icov_ell = np.ones((1, lmax + 1)) * 2
        minfo = map_utils.get_gauss_minfo(2 * lmax)
        icov_pix = np.ones((1, minfo.npix)) * 3

        solver = solvers.CGWiener.from_arrays(
            alm_data, ainfo, icov_ell, icov_pix, minfo)
        b_in = solver.b.copy()
        while solver.i < 10:
            solver.step()

        np.testing.assert_array_almost_equal(solver.A(solver.x), b_in)

class TestCGWienerScaled(unittest.TestCase):

    def test_CGWienerScaled_init(self):
        
        alm_data = np.ones(6)
        icov_signal = lambda alm: alm
        sqrt_cov_signal = lambda alm: alm
        icov_noise = lambda alm: alm

        solver = solvers.CGWienerScaled(
            alm_data, icov_signal, icov_noise, sqrt_cov_signal)

        self.assertTrue(hasattr(solver, 'alm_data'))
        self.assertTrue(hasattr(solver, 'icov_signal'))
        self.assertTrue(hasattr(solver, 'icov_noise'))
        self.assertTrue(hasattr(solver, 'sqrt_cov_signal'))
        self.assertTrue(hasattr(solver, 'beam'))
        self.assertTrue(issubclass(solvers.CGWiener, cg.CG))
        self.assertIs(solver.dot, alm_utils.contract_almxblm)

    def test_CGWienerScaled_init_kwargs(self):
        
        alm_data = np.ones(6)
        icov_signal = lambda alm: alm
        icov_noise = lambda alm: alm
        sqrt_cov_signal = lambda alm: alm

        beam = lambda alm: alm
        x0 = np.zeros_like(alm_data)
        M = lambda alm: alm
        dot = lambda a, b: np.sum(a * b)

        solver = solvers.CGWienerScaled(
            alm_data, icov_signal, icov_noise, sqrt_cov_signal,
            beam=beam, x0=x0, M=M, dot=dot)

        self.assertIs(solver.dot, dot)
        self.assertIs(solver.beam, beam)
        self.assertIs(solver.M, M)
        np.testing.assert_array_equal(solver.x, x0)

    def test_CGWienerScaled_from_arrays(self):
        
        lmax = 2
        ainfo = sharp.alm_info(lmax=lmax)
        alm_data = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm_data = alm_data[np.newaxis,:]
        
        icov_ell = np.ones((1, lmax + 1)) * 2
        minfo = map_utils.get_gauss_minfo(2 * lmax)
        icov_pix = np.ones((1, minfo.npix)) * 3

        solver = solvers.CGWienerScaled.from_arrays(
            alm_data, ainfo, icov_ell, icov_pix, minfo)

        b_in = solver.b.copy()
        while solver.i < 10:
            solver.step()

        np.testing.assert_array_almost_equal(solver.A(solver.x), b_in)
