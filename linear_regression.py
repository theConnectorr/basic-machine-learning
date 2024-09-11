import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_frame = pd.read_csv("points.csv", header=0) # read all the points' data from csv file
# convert the columns of the data_frame
# remember to treat these as matrices, not just vectors
X = np.array([data_frame.x.tolist()]).T
y = np.array([data_frame.y.tolist()]).T

# Xbar = [1, ...X]
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
B = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), B)

print(w)
x0 = np.linspace(-2, 4)
y0 = w[0][0] + w[1][0] * x0

print(x0)

plt.plot(X, y, 'ro')
plt.plot(x0, y0)
plt.axis()
plt.show()