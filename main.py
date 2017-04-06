# import matplotlib.pyplot as plt
import preProcessing as pp
import crossValidation as cv

if __name__ == "__main__":
	# pp.writeToFile()
	# instance, label = pp.readFromFile()
	k = 8
	data = cv.CV(k)
	X_train, Y_train, X_test, Y_test = data.iteration(0)



	print(Y_test)
	# print(len(X_train))
	# print(len(Y_train))