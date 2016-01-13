import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from sklearn.datasets import make_blobs

from .. import LocalityPreservingProjection


def test_estimator_checks():
    """Run scikit-learn's suite of basic estimator checks"""
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(LocalityPreservingProjection)


def test_bad_weights():
    lpp = LocalityPreservingProjection(weight='bad_argument')
    assert_raises(ValueError, lpp.fit, np.zeros((10, 2)))


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


def test_weights():
    X, y = make_blobs(100, n_features=3, random_state=42)

    # with large enough weights, results should be equivalent to adjacency
    lpp1 = LocalityPreservingProjection(n_components=2, weight='adjacency')
    lpp2 = LocalityPreservingProjection(n_components=2, weight='heat',
                                        weight_width=1E6)

    assert_allclose(lpp1.fit_transform(X),
                    lpp2.fit_transform(X))
