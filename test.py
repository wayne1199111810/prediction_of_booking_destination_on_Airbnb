# # import matplotlib.pyplot as plt
# import include.crossValidation as cv
# import include.binary.binaryClassifier as bc
# from include.utility import *
from include import crossValidation as cv
from include import preProcessing as pp
# from sklearn import linear_model
# from sklearn import svm
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.feature_selection import VarianceThreshold


if __name__ == "__main__":

	# size_of_training = 1000;
	# createNewTrainingFileWithSize(size_of_training)
	
	destination, users = pp.readRawData()

	pp.writeToFile(users, destination, 'Data/user_age.dat', 'Data/destination_age.dat', cvs = False)
