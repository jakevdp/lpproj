import numpy as np
from scipy import linalg

from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin


class LocalityPreservingProjection(BaseEstimator, TransformerMixin):
    """Locality Preserving Projection
    
    """
    def __init__(self, n_components=2, n_neighbors=5,
                 kernel='adjacency', kernel_width=1.0,
                 neighbors_algorithm='auto'):
        # TODO: allow radius neighbors
        # TODO: allow for precomputed kernels
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.kernel_width = kernel_width
        self.neighbors_algorithm = neighbors_algorithm

    def fit(self, X, y=None):
        X = check_array(X)
        W = self._compute_weights(X)
        self.projection_ = self._compute_projection(X, W)
        return self

    def transform(self, X):
        X = check_array(X)
        return np.dot(X, self.projection_)

    def _compute_projection(self, X, W):
        """Compute the LPP projection matrix

        Parameters
        ----------
        X : array_like, (n_samples, n_features)
            The input data
        W : array_like or sparse matrix, (n_samples, n_samples)
            The precomputed adjacency matrix

        Returns
        -------
        P : ndarray, (n_features, self.n_components)
            The matrix encoding the locality preserving projection
        """
        # TODO: check W input; handle sparse case
        X = check_array(X)

        D = np.diag(W.sum(1))
        L = D - W
        evals, evecs = eigh_robust(np.dot(X.T, np.dot(L, X)),
                                   np.dot(X.T, np.dot(D, X)),
                                   eigvals=(0, self.n_components - 1))
        return evecs

    def _compute_weights(self, X):
        X = check_array(X)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(X)

        if self.kernel == 'adjacency':
            W = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                 mode='connectivity', include_self=True)
        elif self.kernel == 'heat':
            W = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                 mode='distance')
            W.data = np.exp(-W.data ** 2 / self.kernel_width ** 2)
        else:
            raise ValueError("Unrecognized Kernel")
            
        # symmetrize the matrix
        # TODO: make this more efficient & keep sparse output
        W = W.toarray()
        W = np.maximum(W, W.T)
        return W


def eigh_robust(a, b=None, **kwargs):
    """Robustly solve the Hermitian generalized eigenvalue problem

    This function robustly solves the Hermetian generalized eigenvalue problem
    ``A v = lambda B v`` in the case that B is not strictly positive definite.
    When B is strictly positive-definite, the result is equivalent to
    scipy.linalg.eigh() within floating-point accuracy.

    Parameters
    ----------
    a : (M, M) array_like
        A complex Hermitian or real symmetric matrix whose eigenvalues and
        eigenvectors will be computed.
    b : (M, M) array_like, optional
        A complex Hermitian or real symmetric matrix.
        If omitted, identity matrix is assumed.
    **kwargs :
        Additional keywords are passed to ``scipy.linalg.eigh()``.

    Returns
    -------
    w : (N,) float ndarray
        The N (1<=N<=M) selected eigenvalues, in ascending order, each
        repeated according to its multiplicity.
    v : (M, N) complex ndarray
        (if eigvals_only == False)
    """
    if b is None:
        return linalg.eigh(a, b, **kwargs)
    if kwargs.get('type', 1) != 1:
        raise ValueError("Only type=1 generalized eigenvalue problems "
                         "are supported")

    # Compute eigendecomposition of b
    kwargs_b = dict(kwargs)
    kwargs_b['eigvals_only'] = False
    kwargs_b['eigvals'] = None
    kwargs_b['overwrite_a'] = kwargs.get('overwrite_b', False)
    S, U = linalg.eigh(b, **kwargs_b)

    # Combine a and b on left-hand-side
    S[S <= 0] = np.inf
    Sinv = 1. / np.sqrt(S)
    W = Sinv[:, None] * np.dot(U.T, np.dot(a, U)) * Sinv
    output = linalg.eigh(W, **kwargs)

    if not kwargs.get('eigvals_only'):
        evals, evecs = output
        output = (evals, np.dot(U, Sinv[:, None] * evecs))

    return output
