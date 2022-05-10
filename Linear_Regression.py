import numpy as np
from sklearn import datasets, linear_model, metrics

def linear_regression(X,y):
    # train: init
    n = len(X)

    W = np.random.rand(len(X[0]))
    b = np.random.rand()

    learning_rate = 0.8
    epochs = 1000000

    # train: gradient descent
    for i in range(epochs):
        if i % 100000 == 0:
            learning_rate = 0.75
        # calculate predictions
        prediction = np.array([np.dot(x, W) + b for x in X])

        # calculate error and cost (mean squared error)
        mean_squared_error = metrics.mean_squared_error(y, prediction)
        errors = (y - prediction)
        cost = np.sqrt(mean_squared_error)

        # calculate gradients

        Derivative_by_w = (2 / n) * np.dot(errors, X)
        Derivative_by_b = 2 * np.mean(errors)
        # TODO

        # update parameters
        W += learning_rate * Derivative_by_w
        b += learning_rate * Derivative_by_b

        if i % 5000 == 0:
            print("Epoch %d: %f" % (i, mean_squared_error))
        return W, b