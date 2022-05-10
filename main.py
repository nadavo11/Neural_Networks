import numpy as np
from sklearn import datasets, linear_model, metrics
from Linear_Regression import linear_regression
# Load diabetes dataset
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data  # matrix of dimensions 442x10

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Load diabetes dataset
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data  # matrix of dimensions 442x10

# with scikit learn:
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)

# The mean squared error
mean_squared_error = metrics.mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("Mean squared error: %.2f" % mean_squared_error)
print("=" * 80)

# Load diabetes dataset
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data  # matrix of dimensions 442x10

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# train
X = diabetes_X_train
y = diabetes_y_train

W, b = linear_regression(X, y)
# diagnostic output


# test
X = diabetes_X_test
y = diabetes_y_test

prediction = np.dot(X, W)+b
mean_squared_error = metrics.mean_squared_error(y,prediction)

print("tested mean square error: " ,( mean_squared_error))
