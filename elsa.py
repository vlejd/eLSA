"""Implementation of eLSA"""
import gensim
import numpy as np
from term_weights import SupervisedTermWeightingWTransformer, UnsupervisedTfidfTransformer

class ELSA(object):
    
    SCHEMES = ('None', 'tfidf', 'tfchi2', 'tfig', 'tfgr', 'tfor', 'tfrf')
    BASIC_SCHEMES = ('None', 'tfidf', 'tfchi2')

    def __init__(self, cls=None, weights='None', w=None, use_svd=True, svd_dim=None):
        self.cls = cls
        if self.cls is None:
            raise("No classifier")

        TRANSFORMERS = { 
            'tfidf': UnsupervisedTfidfTransformer(norm=None),
            'tfchi2': SupervisedTermWeightingWTransformer(scheme='tfchi2'),
            'tfig': SupervisedTermWeightingWTransformer(scheme='tfig'),
            'tfgr': SupervisedTermWeightingWTransformer(scheme='tfgr'),
            'tfor': SupervisedTermWeightingWTransformer(scheme='tfor'),
            'tfrf': SupervisedTermWeightingWTransformer(scheme='tfrf'),
            'None': None,
            None: None
        }
        self.weight_model = TRANSFORMERS[weights] 
        assert w is None or type(w) == dict
        if w is None:
            self.internal_w = w
        else:
            self.internal_w = w.copy()

        self.use_svd = use_svd
        self.svd_dim = svd_dim if svd_dim is not None else 200

    def get_matrix_w(self):
        if self.internal_w is None:
            return np.ones(self.num_terms)
        else:
            matrix_w = np.ones(self.num_terms)
            for x in self.internal_w:
                idd = self.dictionary.token2id[x]
                matrix_w[idd] = self.internal_w[x]

            return matrix_w

    def save_matrix_w(self, matrix_w):
        assert matrix_w.shape[0] == self.num_terms
        self.internal_w = {}
        for word in self.dictionary.token2id:
            word_idd = self.dictionary.token2id[word]
            self.internal_w[word] = matrix_w[word_idd]


    def fit(self, X, Y):
        self.dictionary = gensim.corpora.Dictionary(X)
        self.num_terms = len(self.dictionary.dfs)
        bow = list(map(self.dictionary.doc2bow, X))
        bow = gensim.matutils.corpus2csc(bow, num_terms=self.num_terms).T
        if self.weight_model is not None:
            self.weight_model.fit(bow, Y)
            bow = self.weight_model.transform(bow)

        matrix_w = self.get_matrix_w()
        bow = bow.multiply(matrix_w)
        if self.use_svd:
            bow = gensim.matutils.Sparse2Corpus(bow.T)
            self.lsi = gensim.models.LsiModel(bow, num_topics=self.svd_dim, id2word=self.dictionary, extra_samples=200, power_iters=3)
            self.corpus = gensim.matutils.corpus2dense(self.lsi[bow], self.lsi.num_topics).T
        else:
            self.corpus = bow.multiply(matrix_w)
            
        self.cls.fit(self.corpus, Y)
    
    def dw(self, X, Y):
        # Returns derivation with respect to weights w
        if not self.use_svd:
            raise('Derivation can be used only wth svd') 
        _ = self.predict(X)
        self.d_embedding = self.cls.dx(self.embedding, Y)
        uT = self.lsi.projection.u.T.copy()
        uT.resize((self.svd_dim, uT.shape[1])) # because gensim sometimes sucks
        res = self.d_embedding.dot(uT)
        dw = self.bow.T.multiply(res.T).sum(axis=1).T.A1
        return dw
        
    
    def predict(self, X):
        bow = list(map(self.dictionary.doc2bow, X))
        bow = gensim.matutils.corpus2csc(bow, num_terms=self.num_terms).T

        if self.weight_model is not None:
            bow = self.weight_model.transform(bow)
        self.bow = bow
        matrix_w = self.get_matrix_w()
        if self.use_svd:
            bow = bow.multiply(matrix_w)
            bow = gensim.matutils.Sparse2Corpus(bow.T)
            self.embedding = gensim.matutils.corpus2dense(self.lsi[bow], self.lsi.num_topics).T
        else:
            self.embedding = bow.multiply(matrix_w)            
        Yhat = self.cls.predict(self.embedding)
        return Yhat
    
    def score(self, X, Y):
        Yhat = self.predict(X)
        return 1-((Yhat-Y)**2).mean()
    
    def update(self, X, Y):
        pass