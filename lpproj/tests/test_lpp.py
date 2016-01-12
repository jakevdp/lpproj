import numpy as np
from numpy.testing import assert_equal, assert_allclose
from sklearn.datasets import make_blobs

from .. import LocalityPreservingProjection

def test_lpp_transform():
    # just a smoketest
    X = np.random.rand(100, 2)

    lpp = LocalityPreservingProjection(n_components=1)
    lpp = lpp.fit(X)
    Xproj = lpp.transform(X)
    assert_equal(Xproj.shape, X[:, :1].shape)

    Xproj = lpp.fit_transform(X)
    assert_equal(Xproj.shape, X[:, :1].shape)


def test_line_plus_outlier():
    # test that a basic line plus outlier is projected as expected
    # cf. figure 1 in the LPP paper
    X = np.zeros((100, 2))
    X[:, 0] = np.arange(100)
    X[0, 1] = 30

    lpp = LocalityPreservingProjection(n_components=2)
    Xlpp = lpp.fit_transform(X)
    assert_allclose(Xlpp[1:, 1], 0, atol=1E-3)

    ratio = Xlpp[1:, 0] / X[1:, 0]
    assert_allclose(ratio, ratio[0])
