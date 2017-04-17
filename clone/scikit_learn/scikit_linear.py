from sklearn import linear_model

model = linear_model.LinearRegression()

# Training data
X_train = [[0,1], [0,2], [5,10], [1,-1], [2,-4], [3,-8]]

# Testing data
Y_train = [[0],[0],[0],[1],[1],[1]]

print(Y_train)

# Fit the data to the logistic model
model.fit(X_train, Y_train)

# Predict
X_test = [[0,0]]
result = model.predict(X_test)

print(result)


