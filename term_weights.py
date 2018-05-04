# Term weighing schemes.
# We are using implementation of https://github.com/aysent/supervised-term-weighting/blob/master/stw.py
# We added UnsupervisedTfidfTransformer wrapper.
# This code is uder MIT license
"""
The MIT License (MIT)

Copyright (c) 2015 Aysen Tatarinov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer


class UnsupervisedTfidfTransformer(TfidfTransformer):
    def fit(self, X, y=None):
        return super().fit(X)


class SupervisedTermWeightingWTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, scheme='tfchi2'):

        self.scheme = scheme


    def fit(self, X, y):
        
        n_samples, n_features = X.shape

        # Masks for positive and negative samples
        pos_samples = sp.spdiags(y, 0, n_samples, n_samples)
        neg_samples = sp.spdiags(1-y, 0, n_samples, n_samples)

        # Extract positive and negative samples
        X_pos = pos_samples*X
        X_neg = neg_samples*X

        # tp: number of positive samples that contain given term
        # fp: number of positive samples that do not contain given term
        # fn: number of negative samples that contain given term
        # tn: number of negative samples that do not contain given term
        tp = np.bincount(X_pos.indices, minlength=n_features)
        fp = np.sum(y)-tp
        fn = np.bincount(X_neg.indices, minlength=n_features)
        tn = np.sum(1-y)-fn

        # Smooth document frequencies
        self._tp = tp + 1.0
        self._fp = fp + 1.0
        self._fn = fn + 1.0
        self._tn = tn + 1.0

        self._n_samples = n_samples
        self._n_features = n_features

        return self

    def transform(self, X):

        tp = self._tp
        fp = self._fp
        fn = self._fn
        tn = self._tn

        # Smooth document frequencies
        n = self._n_samples + 4

        f = self._n_features

        if self.scheme == 'tfchi2':

            k = n * (tp * tn - fp * fn)**2
            k /= (tp + fp) * (fn + tn) * (tp + fn) * (fp + tn)

        elif self.scheme == 'tfig':

            k = -((tp + fp) / n) * np.log((tp + fp) / n)
            k -= ((fn + tn) / n) * np.log((fn + tn) / n)
            k += (tp / n) * np.log(tp / (tp + fn))
            k += (fn / n) * np.log(fn / (tp + fn))
            k += (fp / n) * np.log(fp / (fp + tn))
            k += (tn / n) * np.log(tn / (fp + tn))

        elif self.scheme == 'tfgr':

            k = -((tp + fp) / n) * np.log((tp + fp) / n)
            k -= ((fn + tn) / n) * np.log((fn + tn) / n)
            k += (tp / n) * np.log(tp / (tp + fn))
            k += (fn / n) * np.log(fn / (tp + fn))
            k += (fp / n) * np.log(fp / (fp + tn))
            k += (tn / n) * np.log(tn / (fp + tn))
            
            d = -((tp + fp) / n) * np.log((tp + fp) / n)
            d -= ((fn + tn) / n) * np.log((fn + tn) / n)

            k /= d

        elif self.scheme == 'tfor':

            k = np.log( (tp * tn ) / (fp * fn) )

        elif self.scheme == 'tfrf':

            k = np.log(2 + tp / fn)
            
        X = X * sp.spdiags(k, 0, f, f)

        return normalize(X, 'l2')

