import numpy as np
from scipy import linalg

from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin


class LocalityPreservingProjection(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 neighbors_algorithm='auto', kernel_width=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.neighbors_algorithm = neighbors_algorithm
        self.kernel_width = kernel_width

    def fit(self, X, y=None):
        X = check_array(X)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X

        # TODO: make this more efficient
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
        
        # TODO: Make this computation more stable using
        # http://www.iipl.fudan.edu.cn/~zhangjp/literatures/MLF/TR-2002-09.pdf
        A = np.dot(X.T, np.dot(L, X))
        B = np.dot(X.T, np.dot(D, X))
        
        evals, evecs = linalg.eigh(A, B)            
        self.projection_ = evecs[:, :self.n_components]

        return self

    def transform(self, X):
        X = check_array(X)
        return np.dot(X, self.projection_)
