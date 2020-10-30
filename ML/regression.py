import numpy as np
from pandas import Series


class Regression(object):

    # y = m*x + b
    def __init__(self, x: Series, y: Series, count=10000):
        assert len(x) == len(y)
        self._n = len(x)
        self.m = 0
        self.b = 0
        self.count = count
        self.alpha = 0.0001
        self._x: np.ndarray = x.to_numpy().reshape(len(x), 1)
        self._y: np.ndarray = y.to_numpy().reshape(len(x), 1)

    def fit(self):
        for index in range(self.count):
            pred_y = self.m * self._x + self.b
            error = self._y - pred_y

            m_adjust = -(2 / self._n) * np.sum(self._x * error)
            b_adjust = -(2 / self._n) * np.sum(self._y - pred_y)

            self.b = self.b - self.alpha * b_adjust
            self.m = self.m - self.alpha * m_adjust

    def predict(self, value: float):
        return self.m * value + self.b
