from sklearn import linear_model

model = linear_model.LinearRegression()

# Training data
X_train = [[0,1], [-1,2], [2,0], [1,1]]

# Testing data
Y_train = [[0],[1],[2],[1]]

print(Y_train)

# Fit the data to the logistic model
model.fit(X_train, Y_train)

# Predict
X_test = [[1,1]]
result = model.predict(X_test)

print(result)


