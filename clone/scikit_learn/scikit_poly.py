from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degree = 3

model = make_pipeline(PolynomialFeatures(degree), Ridge())

# Training data
X_train = [[0,0], [-1,-1], [2,2], [1,2]]

# Testing data
Y_train = [[0],[1],[0],[1]]

# Fit the data to the logistic model
model.fit(X_train, Y_train)

# Predict
X_test = [[1,2]]
result = model.predict(X_test)

print(result)
