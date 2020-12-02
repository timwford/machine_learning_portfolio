import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generate_logistic_data():
    np.random.seed(12)
    num_observations = 2000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    features = np.vstack((x1, x2)).astype(np.float32)
    labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

    return features, labels

def plot_data(features, labels):
    plt.figure(figsize=(12, 8))
    plt.scatter(features[:, 0], features[:, 1],c=labels, alpha=.8)
    plt.show()

class LogisticRegression:

    def __init__(self, epoch: int, rate: float):
        self.epoch = epoch
        self.rate = rate
        self.intercept = 0
        self.w1 = 0
        self.w2 = 0

    def predict(self, x1_value, x2_value):
        return self.sigmoid(x1_value, x2_value)

    def sigmoid(self, x1_value, x2_value):
        return 1.0 / (1.0 + np.exp(- (self.intercept + self.w1 * x1_value + self.w2 * x2_value)))

    def update_weights(self, x1_value: float, x2_value: float, y_error, y_predict):
        self.intercept = self.intercept + self.rate * y_error * y_predict * (1.0 - y_predict)
        self.w1 = self.w1 + self.rate * y_error * y_predict * (1.0 - y_predict) * x1_value
        self.w2 = self.w2 + self.rate * y_error * y_predict * (1.0 - y_predict) * x2_value

    def fit(self, features, classes):
        for _ in range(0, self.epoch):
            for index in range(len(features)):
                x1_value = features[index][0]
                x2_value = features[index][1]

                y_predict = self.sigmoid(x1_value, x2_value)
                y_error = classes[index] - y_predict
                self.update_weights(x1_value, x2_value, y_error, y_predict)

    def __str__(self):
        return f"Logistic Regression intercept: {self.intercept} w1: {self.w1} w2: {self.w2}"


if __name__ == "__main__":
    model = LogisticRegression(2000, .0001)
    features, labels = generate_logistic_data()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

    model.fit(X_train, y_train)
    print(str(model))

    results = []
    high = []
    low = []
    for i in range(len(X_test)):
        result = np.round(model.predict(X_test[i][0], X_test[i][1]))
        if result != y_test[i]:
            result = 2

        if result > 0:
            high.append(result)
        else:
            low.append(result)

        results.append(result)

    plt.figure(figsize=(12, 8))

    plt.scatter(X_test[:, 0], X_test[:, 1], c=results, alpha=.8)
    plt.show()

    print(f"Accuracy score: {accuracy_score(y_test, results, normalize=True)}")
