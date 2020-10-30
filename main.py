# Machine Learning
# CSC4something
# Timothy Ford

import pandas as pd
import numpy as np
import typer

from ML.perceptron import Perceptron
from ML.plotter import scatterplot, plot_decision_regions
from ML.stump import Stump
from ML.regression import Regression

# Perceptron
iris_types = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
iris_values = ["sepal_l", "sepal_w", "petal_l", "petal_w"]
iris_type_help: str = "0 - Iris-virginica  1 - Iris-setosa 2 - Iris-versicolor"
iris_value_help: str = "0 - sepal_l  1 - sepal_w  2 - petal_l  3 - petal_w"

# Regression
flow_data_set = 'data/flow.csv'

# Stump
iris_data_set = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
stump_iris_types = ['Iris-setosa', 'Iris-versicolor']

app = typer.Typer()

@app.command()
def perceptron(flower1: int = typer.Option(1, help=iris_type_help),
                    flower2: int = typer.Option(2, help=iris_type_help),
                    col1: int = typer.Option(0, help=iris_value_help),
                    col2: int = typer.Option(2, help=iris_value_help),
                    rate: float = typer.Option(0.1, help="The Perceptrons' learning rate"),
                    max_iterations: int = typer.Option(100, help="The max number of iterations the perceptron will run")):

    df = pd.read_csv(iris_data_set, header=None, names=["sepal_l", "sepal_w", "petal_l", "petal_w", "breed"])
    print(df.head())

    flower1 = iris_types[flower1]
    flower2 = iris_types[flower2]
    flowers = [flower1, flower2]
    print(f"\nComparing {flower1} against {flower2}")

    y = df[df['breed'].isin(flowers)].iloc[:, 4].values
    y = np.where(y == flower1, -1, 1)

    value1 = col1
    value2 = col2

    print(f"\nUsing columns {iris_values[value1]} and {iris_values[value2]}")
    X = df[df['breed'].isin(flowers)].iloc[:, [value1, value2]].values

    percy = Perceptron(rate=rate, n_iter=max_iterations)

    percy.fit(X, y)

    print(f"Weights for fit: {percy.weights}")
    print(f"Errors for fit: {percy.errors}")

    plot_decision_regions(X, y, percy)

@app.command()
def regression():
    df = pd.read_csv(iris_data_set, header=None, names=["sepal_l", "sepal_w", "petal_l", "petal_w", "breed"])
    virginica_sepal_l = df[df['breed'].isin(['Iris-virginica'])]['sepal_l']
    virginica_sepal_w = df[df['breed'].isin(['Iris-virginica'])]['sepal_w']

    regression = Regression(virginica_sepal_w, virginica_sepal_l)
    regression.fit()

    scatterplot(virginica_sepal_w, virginica_sepal_l, regression.m, regression.b)

@app.command()
def stump():
    df = pd.read_csv(iris_data_set, header=None, names=["sepal_l", "sepal_w", "petal_l", "petal_w", "breed"])

    indexes = [2, 3]
    label_df = df[df['breed'].isin(stump_iris_types)]['breed']
    label_set = np.where(label_df == stump_iris_types[0], -1, 1).reshape(len(label_df), 1)
    data_set = df[df['breed'].isin(stump_iris_types)].iloc[:, indexes]

    stump = Stump(data_set.values, label_set, steps=100)
    stump.fit()

    print(stump)
    if stump.ideal_dimension == 1:
        scatterplot(data_set.iloc[:, 0], data_set.iloc[:, 1], 0, stump.ideal_split)
    else:
        scatterplot(data_set.iloc[:, 0], data_set.iloc[:, 1], 0, stump.ideal_split, vertical_line=True)


if __name__ == '__main__':
    app()
