import numpy as np


class Perceptron(object):
    def __init__(self, rate=0.01, n_iter=10):
        self.weights = []
        self.rate = rate
        self.n_iter = n_iter

        self.errors = []

    def fit(self, X, y):
        self.weights = [0 for i in range(len(X[0]) + 1)]

        for i in range(self.n_iter):
            iteration_error = 0

            for xi, target in zip(X, y):
                prediction = self.predict(xi)

                e = target - prediction
                iteration_error += abs(e)

                self.weights[0] = self.rate * e
                self.weights[1:] += (self.rate * e * xi)

            self.errors.append(iteration_error)

            if iteration_error == 0:
                break

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
