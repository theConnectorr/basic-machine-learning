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

def linear():
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def h(w, x):
    return sigmoid(w.T @ x)

def compute_gradient(w, x, y):
    m = w.shape[0]
    gradient = (1 / m) * np.sum((h(w, x) - y) * x, axis=1)

    return gradient

def gender_numberize(gender):
    if gender == "Male":
        return 1
    return 0

if __name__ == "__main__":
    data = pd.read_csv("datasets/logistic_reg/social_network_ads.csv", names=["UserID", "Gender","Age", "EstimatedSalary", "Purchased"])
    data["Gender"] = np.where(data["Gender"] == "Male", 1, 0)
    data = data.values

    train_data = data[:3 * len(data) // 4]
    test_data = data[3 * len(data) // 4:]

    w = np.array([0.] * 4)
    x = train_data[:, :-1]
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x.astype(np.float64), axis=0)
    x = (x - x_mean) / x_std
    x = x.T
    y = train_data[:, -1]

    for iter in range(1000):
        w -= 0.01 * compute_gradient(w, x, y)

    print(w)