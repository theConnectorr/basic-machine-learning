import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def f(k, x): # calculates the value of the polynomial with coefficient of k at x
    result = 0 
    for i in range(len(k)):
        result += k[i] * x ** i
    return result

def cost(y_predict, y):
    result = 0
    for i in range(len(y)):
        result += (y_predict[i] - y[i]) ** 2

    return result

def forward_gradient(k, samples): # f
    result = [0] * len(k)
    for i in range(len(k)):
        speed = 0
        for sample in samples:
            print(sample)
            speed += (f(k, sample['x']) - sample['y']) * sample['x'] ** i    
        result[i] = 2 * speed

    return result

def update_coff(k, gradient, learning_rate):
    for i in range(len(k)):
        k[i] -= learning_rate * gradient[i]

    return k

def polynomial_regression(initial_k, samples):
    k = [x for x in initial_k]
    tries = 400
    for _ in range(tries):
        gradient = forward_gradient(k, samples)
        k = update_coff(k, gradient, 0.0000001)

    return k

if __name__ == "__main__":
    data_frame = pd.read_csv("points.csv", header=0)
    samples = [{"x": row.x, "y": row.x} for index, row in data_frame.iterrows()]
    samples_x = data_frame.x.tolist()
    samples_y = data_frame.y.tolist()
    plt.plot(samples_x, samples_y, "ro")

    k = [0] * 6
    k = polynomial_regression(k, samples)
    x = np.linspace(min(data_frame.x.tolist()), max(data_frame.x.tolist()))
    plt.plot(x, [f(k, x) for x in x])

    plt.show()
