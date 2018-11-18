from logistic_regression import LogisticRegression

import numpy as np

class MulticlassLogisticRegression:
    def __init__(self, number_of_classes, number_of_iterations, learning_rate):
        """
        :param number_of_classes: number of classes to predict, must be greater then 2.
        :param number_of_iterations: number of iterations of each classifier.
        :param learning_rate: learning rate.
        """
        assert (number_of_classes > 2)
        self.nc = number_of_classes
        self.lrs = []
        for i in range(0, self.nc):
            self.lrs.append(LogisticRegression(number_of_iterations, learning_rate))

    def fit(self, X, y):
        """
        :param X: input.
        :param y: desired output.
        """
        for i in range(0, len(self.lrs)):
            self.lrs[i].fit(X, (y == i))

    def predict(self, X):
        """
        :param X: input.
        :return: class index with maximum probability.
        """
        y_prob = []
        for i in range(0, len(self.lrs)):
            y_prob.append(self.lrs[i].predict_prob(X))
        winner_class = np.argmax(y_prob)
        print("Predicted class is " + str(winner_class) + ", with probability " + str(y_prob[winner_class]))
        return winner_class
