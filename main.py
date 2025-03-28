import pandas as pd
import numpy as np
from LinearRegression import LinearRegression

def data_standardize(data):
    x = data[:, 0].reshape(-1, 1)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / x_std
    N = data.shape[0]
    xbar = np.hstack((np.ones((N, 1)), x))
    y = data[:, 1].reshape(-1, 1)

    return x, xbar, y, N

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv("datasets/linear_reg/train.csv").dropna().values
    x, xbar, y, N = data_standardize(data)

    # Train the model
    linear_regressor = LinearRegression(
        epochs=1000,
        learning_rate=0.01,
        verbose=10,
    )

    w, costs = linear_regressor.train(x, xbar, y, N)

    # Show the costs graph and regression line
    linear_regressor.show_result()
    