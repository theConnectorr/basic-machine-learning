import numpy as np
import matplotlib.pyplot as plot

class LinearRegression:
    def __init__(self, epochs: int, learning_rate, verbose):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

    def train(self, x, xbar, y, N):
        self.x = x
        self.xbar = xbar
        self.y = y
        self.N = N

        self.costs = np.zeros((self.epochs, 1))
        self.w = np.array([0., 0.]).reshape(-1, 1)

        for i in range(self.epochs):
            error = xbar @ self.w - y

            self.w[0] -= self.learning_rate * (np.sum(error) / N)
            self.w[1] -= self.learning_rate * (np.sum(error * x) / N)

            self.costs[i] = 0.5 * np.mean(error * error)
            if not (i % self.verbose): 
                print('Iteration ', i, ', cost: ', self.costs[i], end="\n")

        return self.w, self.costs
    
    def show_result(self):
        # Create subplots (2 rows, 1 column)
        _, axes = plot.subplots(2, 1, figsize=(6, 8))

        # First plot: Cost function over epochs
        axes[0].plot(np.arange(0, 1000), self.costs, 'y')
        axes[0].set_title("Cost Graph")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cost")

        # Second plot: Linear regression fit
        x_space = np.linspace(np.min(self.x), np.max(self.x), 100).reshape(-1, 1)
        y_space = np.hstack((np.ones((x_space.shape[0], 1)), x_space)) @ self.w

        axes[1].plot(self.x, self.y, "bo", label="Training Data")
        axes[1].plot(x_space, y_space, "r", label="Regression Line")
        axes[1].set_title("Regression line")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].legend()

        # Show the plots
        manager = plot.get_current_fig_manager()
        manager.resize(700, 700)
        plot.tight_layout()
        plot.show()