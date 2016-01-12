import numpy as np
from numpy.testing import assert_allclose

from scipy import linalg
from ..lpproj import eigh_robust

def sym_matrix(N, rand):
    A = rand.rand(N, N)
    return np.dot(A, A.T)

def assert_allclose_evecs(evecs1, evecs2):
    """Check that eigenvectors are equal up to a sign in each column"""
    sign = np.sign((evecs1 * evecs2).sum(0))
    assert_allclose(evecs1, evecs2 * sign)


def test_eigh_robust_basic():
    rand = np.random.RandomState(42)

    A = sym_matrix(10, rand)
    B = sym_matrix(10, rand)

    # Eigenvalues and vectors
    evals1, evecs1 = linalg.eigh(A, B)
    evals2, evecs2 = eigh_robust(A, B)
    assert_allclose_evecs(evecs1, evecs2)
    assert_allclose(evals1, evals2)

    # Eigenvectors only
    evals2 = eigh_robust(A, B, eigvals_only=True)
    assert_allclose(evals1, evals2)


def test_eigh_robust_overwrite():
    rand = np.random.RandomState(42)

    def check_results(rand, overwrite_a, overwrite_b):
        A = sym_matrix(10, rand)
        B = sym_matrix(10, rand)
        kwargs = dict(overwrite_a=overwrite_a,
                      overwrite_b=overwrite_b)
        evals1, evecs1 = linalg.eigh(A.copy(), B.copy(), **kwargs)
        evals2, evecs2 = eigh_robust(A.copy(), B.copy(), **kwargs)
        assert_allclose_evecs(evecs1, evecs2)
        assert_allclose(evals1, evals2)

    for overwrite_a in [True, False]:
        for overwrite_b in [True, False]:
            yield check_results, rand, overwrite_a, overwrite_b


def test_eigh_robust_eigval_selection():
    rand = np.random.RandomState(42)
    A = sym_matrix(10, rand)
    B = sym_matrix(10, rand)

    def check_results(eigvals):
        evals1, evecs1 = linalg.eigh(A, B, eigvals=eigvals)
        evals2, evecs2 = eigh_robust(A, B, eigvals=eigvals)
        assert_allclose_evecs(evecs1, evecs2)
        assert_allclose(evals1, evals2)

    for eigvals in [(0, 9), (0, 2), (8, 9), (4, 6)]:
        yield check_results, eigvals


def test_eigh_robust_turbo():
    rand = np.random.RandomState(42)
    A = sym_matrix(10, rand)
    B = sym_matrix(10, rand)

    def check_results(turbo):
        evals1, evecs1 = linalg.eigh(A, B, turbo=turbo)
        evals2, evecs2 = eigh_robust(A, B, turbo=turbo)
        assert_allclose_evecs(evecs1, evecs2)
        assert_allclose(evals1, evals2)

    for turbo in [True, False]:
        yield check_results, turbo

def test_non_generalized():
    rand = np.random.RandomState(42)
    A = sym_matrix(10, rand)

    evals1, evecs1 = eigh_robust(A)
    evals2, evecs2 = linalg.eigh(A)

    assert_allclose_evecs(evecs1, evecs2)
    assert_allclose(evals1, evals2)
