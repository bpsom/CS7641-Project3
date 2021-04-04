#!/usr/bin/env python3
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, chi2


class ULDimensionReduction(object):
    def __init__(self, uld_algo="PCA", n_clusters=2):
        self.trained_model = None
        self.n_clusters = n_clusters

        if uld_algo == "PCA":
            self.model = PCA(n_components=n_clusters)
        elif uld_algo == "ICA":
            self.model = FastICA(n_components=n_clusters)
        elif uld_algo == "RP":
            self.model = GaussianRandomProjection(n_components=n_clusters)
        elif uld_algo == "FA":
            self.model = FactorAnalysis(n_components=n_clusters)

    def add_evidence(self, X, y):
        """
        Add training data to learner
        """
        self.trained_model = self.model.fit(X, y)

    def transform_fit(self, X):
        return self.trained_model.fit_transform(X)

    def ret_model(self):
        return self.model

    # evaluate a given model using cross-validation
    def evaluate_model(self, X, y):
        rsfk = RepeatedStratifiedKFold(n_splits=self.n_clusters, n_repeats=2, random_state=5)
        scores = cross_val_score(PCA(n_components=self.n_clusters),
                                 X,
                                 y,
                                 scoring='accuracy',
                                 cv=rsfk,
                                 n_jobs=-1,
                                 error_score='raise')
        return scores


def pca_dim_reduction(sdata, n_clusters=10):
    x_train, x_test, y_train, y_test = sdata[0], sdata[1], sdata[2], sdata[3]
    pca = ULDimensionReduction("PCA", n_clusters)
    clf = pca.ret_model()
    new_train_x = clf.fit_transform(x_train)
    new_test_x = clf.fit_transform(x_test)
    return clf, new_train_x, new_test_x


def ica_dim_reduction(sdata, n_clusters=10):
    x_train, x_test, y_train, y_test = sdata[0], sdata[1], sdata[2], sdata[3]
    ica = ULDimensionReduction("ICA", n_clusters)
    clf = ica.ret_model()
    new_train_x = clf.fit_transform(x_train)
    new_test_x = clf.fit_transform(x_test)
    return clf, new_train_x, new_test_x


def rp_dim_reduction(sdata, n_clusters=10):
    x_train, x_test, y_train, y_test = sdata[0], sdata[1], sdata[2], sdata[3]
    rp = ULDimensionReduction("RP", n_clusters)
    clf = rp.ret_model()
    new_train_x = clf.fit_transform(x_train)
    new_test_x = clf.fit_transform(x_test)
    return clf, new_train_x, new_test_x


def fa_dim_reduction(sdata, n_clusters=10):
    x_train, x_test, y_train, y_test = sdata[0], sdata[1], sdata[2], sdata[3]
    fa = ULDimensionReduction("FA", n_clusters)
    clf = fa.ret_model()
    new_train_x = clf.fit_transform(x_train)
    new_test_x = clf.fit_transform(x_test)
    return clf, new_train_x, new_test_x


if __name__ == "__main__":
    print(" UL Dimension Reduction")

