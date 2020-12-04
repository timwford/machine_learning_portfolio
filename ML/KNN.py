from typing import List

import numpy as np


class Point:
    def __init__(self, x, y, label=None):
        self.x = x
        self.y = y
        self.label = label
        self.distance = None

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __lt__(self, other):
        return self.distance < other.distance


class KNN:
    def __init__(self, k: int, data: List[Point], debug=False):
        self.k = k
        self.data = data
        self.debug = debug

    def distance(self, x: Point, y: Point):
        if x is None or y is None:
            return "Unable to do anything"

        if x.x is None or x.y is None or y.x is None or y.y is None:
            return "Unable to classify"

        return np.sqrt(np.square(x.x - y.x) + np.square(x.y - y.y))

    def predict(self, input_value: Point):
        unique_labels = []
        for p in self.data:
            p.distance = self.distance(p, input_value)
            if p.label not in unique_labels:
                unique_labels.append(p.label)

        self.data.sort()
        k_nearest = self.data[:self.k]

        dominant_class = {}
        for label in unique_labels:
            dominant_class[label] = 0

        for p in k_nearest:
            dominant_class[p.label] += 1

        if self.debug:
            print(dominant_class)

        return max(dominant_class, key=dominant_class.get)
