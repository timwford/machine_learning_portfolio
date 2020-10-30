import numpy as np


class Stump(object):

    def __init__(self, data, labels, steps=8):
        self.data: np.ndarray = data
        self.labels = labels

        self.steps = steps
        self.split = 0
        self.dimension = 0
        self.ideal_split = None
        self.ideal_dimension = None
        self.minimum_error = None

    def fit(self):
        for i in range(self.data.shape[1]):
            self.dimension = i
            dimension_data = self.data[:, i]
            total_min = dimension_data.min()
            total_max = dimension_data.max()

            step_size = (total_max - total_min) / self.steps

            for j in range(self.steps):
                self.split = total_min + step_size * j
                predicted = self.predict(dimension_data)

                error = np.sum(abs(predicted.reshape(len(predicted), 1) - self.labels))

                if self.minimum_error is None or error < self.minimum_error:
                    self.minimum_error = error
                    self.ideal_split = self.split
                    self.ideal_dimension = self.dimension

    def __str__(self):
        if self.ideal_split is not None:
            return f"Decision stump: \n\t-split @ {self.ideal_split} \n\t-{self.steps} steps \n\t-{self.ideal_dimension} dimension"
        else:
            return f"Not fit yet"

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.where(values <= self.split, -1, 1)
