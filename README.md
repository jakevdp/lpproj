# Locality Preserving Projections in Python

This is a Python implementation of Locality Preserving Projections (LPP), compatible with scikit-learn.

[![version status](http://img.shields.io/pypi/v/lpproj.svg?style=flat)](https://pypi.python.org/pypi/lpproj)
[![downloads](http://img.shields.io/pypi/dm/lpproj.svg?style=flat)](https://pypi.python.org/pypi/lpproj)
[![build status](http://img.shields.io/travis/jakevdp/lpproj/master.svg?style=flat)](https://travis-ci.org/jakevdp/lpproj)
[![license](http://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/jakevdp/lpproj/blob/master/LICENSE)

## Installation and Use

This package is pure Python, and depends only on [numpy](http://numpy.org/), [scipy](http://scipy.org/), and [scikit-learn](http://scikit-learn.org/).
The released version can be installed via pip:

    $ pip install lpproj

A basic example of its use can be found in [Example.ipynb](Example.ipynb); the library follows scikit-learn's patterns for *fit()*, *transform()*, and *fit_transform()*. The simplest example for a feature matrix X:

'''python
from lpproj import LocalityPreservingProjection 
lpp = LocalityPreservingProjection(n_components=2)

X_2D = lpp.fit_transform(X)

'''

## More Information

For information on the LPP algorithm, see the paper, [Locality Preserving Projections (pdf)](http://papers.nips.cc/paper/2359-locality-preserving-projections.pdf).
