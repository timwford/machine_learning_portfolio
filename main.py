# Machine Learning
# CSC4something
# Timothy Ford

import pandas as pd
import numpy as np
import typer

from ML.plotter import scatterplot
from ML.stump import Stump
from ML.regression import Regression

iris_data_set = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Stump
stump_iris_types = ['Iris-setosa', 'Iris-versicolor']


app = typer.Typer()

@app.command()
def regression():
    print("regression")

    df = pd.read_csv(iris_data_set, header=None, names=["sepal_l", "sepal_w", "petal_l", "petal_w", "breed"])
    # print(df.head())

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
