import numpy as np
from scipy import linalg

from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin


class LocalityPreservingProjection(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 neighbors_algorithm='auto', kernel_width=None, robust=True):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.neighbors_algorithm = neighbors_algorithm
        self.kernel_width = kernel_width
        self.robust = robust

    def fit(self, X, y=None):
        # TODO: make this more efficient
        X = check_array(X)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X

        if self.kernel_width is None:
            W = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                 mode='connectivity', include_self=True)
        else:
            W = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                 mode='distance')
            W.data = np.exp(-W.data ** 2 / self.kernel_width ** 2)
        W = W.toarray()
        W = np.maximum(W, W.T)
        
        D = np.diag(W.sum(1))
        L = D - W
        
        # Compute L' and D' and solve the
        # generalized eigenvalue problem L' a = v D' a
        Lp = np.dot(X.T, np.dot(L, X))
        Dp = np.dot(X.T, np.dot(D, X))
        eigvals = (0, self.n_components - 1)
        
        if not self.robust:
            evals, evecs = linalg.eigh(Lp, Dp, eigvals=eigvals)
        else:
            S, U = linalg.eigh(Dp)
            S[S <= 0] = np.inf
            Sm1_2 = 1. / np.sqrt(S)
            W = Sm1_2[:, None] * np.dot(U.T, np.dot(Lp, U)) * Sm1_2
            evals, evecs = linalg.eigh(W, eigvals=eigvals)
            evecs = np.dot(U, Sm1_2[:, None] * evecs)

        self.projection_ = evecs
        return self

    def transform(self, X):
        X = check_array(X)
        return np.dot(X, self.projection_)
