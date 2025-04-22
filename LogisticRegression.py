import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def h(w, x):
    return sigmoid(w.T @ x)

class LogisticRegression():
    def __init__(self):
        pass
    
    def train():
        pass

    def show_result():
        pass