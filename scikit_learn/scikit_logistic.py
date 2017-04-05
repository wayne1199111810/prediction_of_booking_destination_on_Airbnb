from sklearn import linear_model

model = linear_model.LogisticRegression(C=1e5)

# Training data
X_train = [[0,0], [-1,-1], [2,2], [1,2]]

# Testing data
Y_train = ['apple','apple','orange','orange']

# Fit the data to the logistic model
model.fit(X_train, Y_train)

# Predict
X_test = [[1,0]]
result = model.predict(X_test)

print(result)
