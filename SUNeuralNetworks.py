#!/usr/bin/env python3
import numpy as np
from sklearn import neural_network


class SUNeuralNetworks(object):
    """

    """
    def __init__(self, solver='adam', hidden_layer=(10,10,10), alpha=0.001, eps=0.1, episode=1000):
        self.learner_model = None
        self.model = None
        self.solver = solver
        self.hidden_layer = hidden_layer
        self.alpha = alpha
        self.eps = eps
        self.episode = episode
        self.model = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer,
                                                  solver=solver,
                                                  max_iter=episode,
                                                  epsilon=eps)

    def fit(self, train_features, train_targets):
        return self.model.fit(train_features, train_targets)

    def build_tree(self, data):
        """
        Train the model
        """
        train_features = data[:, 0:-1]
        train_targets = data[:, -1]
        self.learner_model = self.model.fit(train_features, train_targets)

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # slap on 1s column so linear regression finds a constant term
        new_data = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        new_data[:, 0: data_x.shape[1]] = data_x
        # build and save the model
        new_data[:, -1] = data_y
        self.build_tree(new_data)

    def model_prediction(self, test_features):
        prediction = self.learner_model.predict(test_features)
        return prediction

    def ret_accuracy_score(self, data_x, data_y):
        return self.learner_model.score(data_x, data_y)


if __name__ == "__main__":
    print(" ML Neural Networks")

