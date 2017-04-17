from sklearn import svm

# Create SVC model
model = svm.SVC(probability=True)

# Training data
X_train = [[0,0], [-1,-1], [2,2], [1,2]]

# Testing data
Y_train = ['apple','apple','orange','orange']

# Fit the data to the SVC model
model.fit(X_train,Y_train)

# predict
X_test = [[1,0],[2,0]]
result = model.predict(X_test)
prob = model.predict_proba(X_test)

print(result)
print(prob)
print(model.classes_)
