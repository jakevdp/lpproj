import numpy as np
from numpy.testing import assert_allclose

from scipy import linalg
from ..lpproj import eigh_robust

def test_eigh_robust():
    rand = np.random.RandomState(0)
    A = rand.rand(10, 10)
    A = np.dot(A, A.T)
    B = rand.rand(10, 10)
    B = np.dot(B, B.T)

    evals, evecs = linalg.eigh(A, B)
    evals_robust, evecs_robust = eigh_robust(A, B)

    # evecs equal up to a sign
    sign = np.sign((evecs * evecs_robust).sum(0))

    assert_allclose(evals, evals_robust)
    assert_allclose(evecs, evecs_robust * sign)
