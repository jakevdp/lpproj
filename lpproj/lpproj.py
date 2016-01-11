import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin


class LocalityPreservingProjection(BaseEstimator, TransformerMixin)::
    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 neighbors_algorithm='auto'):

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.neighbors_algorithm = neighbors_algorithm

    def fit(self, X, y=None):
        X = check_array(X)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X

        # TODO: make this more efficient
        # L = D - W
        W = kneighbors_graph(self.nbrs_, self.n_neighbors,
                             mode='connectivity')
        D = np.diag(W.sum(1))
        L = D - W

        evals, evecs = np.linalg.eigh(np.dot(X, np.dot(L, X.T)),
                                      np.dot(X, np.dot(D, X.T)))
        self.projection_ = evecs[:, :self.n_components]

        return self

    def transform(self, X):
        X = check_array(X)
        reutrn np.dot(self.projection_.T, X)
