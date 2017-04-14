# import matplotlib.pyplot as plt
import preProcessing as pp
import crossValidation as cv

if __name__ == "__main__":
	# pp.writeToFile()
	instance, label = pp.readFromFile()

	# k = 8
	# data = cv.CV(k)
	# X_train, Y_train, X_test, Y_test = data.iteration(0)



	# print(Y_test)
	# print(len(X_train))
	# print(len(Y_train))
	test_ratio = 0.15
	testing_offset = int(label.shape[0] * test_ratio) + 1

	X_test = instance[0:testing_offset]
	Y_test = label[0:testing_offset]

	X_train = instance[testing_offset:]
	Y_train = label[testing_offset:]

	X_test.dump("Data/users_test.dat")
	Y_test.dump("Data/destination_test.dat")

	print('test: ' + str(X_test.shape))

	X_train.dump("Data/users_train.dat")
	Y_train.dump("Data/destination_trainpython .dat")

	print('train: ' + str(X_train.shape))