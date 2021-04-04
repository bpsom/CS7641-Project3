#!/usr/bin/env python3
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class ULClustering(object):
    def __init__(self, ul_algo="EM", init="k-means++", n_clusters=2, initp='kmeans'):
        self.trained_model = None
        self.n_clusters = n_clusters

        if ul_algo == "Kmeans":
            self.model = KMeans(init=init,
                                n_clusters=n_clusters,
                                n_init=4,
                                random_state=0)
        else:
            self.model = GaussianMixture(n_components=n_clusters,
                                         covariance_type='full',
                                         tol=1e-3,
                                         max_iter=100,
                                         n_init=4,
                                         init_params=initp)

    def add_evidence(self, train_x):
        """
        Add training data to learner
        """
        self.trained_model = self.model.fit(train_x)

    def prediction(self, features):
        prediction = self.trained_model.predict(features)
        return prediction

    def transform_fit(self, features):
        return self.trained_model.fit_predict(features)

    def ret_accuracy_score(self, data_x, data_y):
        return self.trained_model.score(data_x, data_y)

    def ret_inertia(self):
        return self.trained_model.inertia_

if __name__ == "__main__":
    print(" UL Clustering")

