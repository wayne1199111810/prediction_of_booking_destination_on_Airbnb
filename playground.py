# import matplotlib.pyplot as plt
from include import preProcessing as pp
from include.utility import *
from include.multiclass import multiclass_trainer as mt
import include.crossValidation as cv
import include.utility
import math
# get data
# destination, users = pp.readRawData() 

# users, destination = pp.readFromFile('Data/users_origin.dat', 'Data/destination_origin.dat')
# pp.writeToFile(users[0:idx], destination[0:idx], 'Data/users_train.dat', 'Data/destination_train.dat', cvs = False)


#createNewTrainingFileWithSize(50000)


k = 3
data = cv.CV(k)
X_train, Y_train, X_valid, Y_valid = data.iteration(2)


trainer1 = mt.multiclass_trainer()
trainer1.train(X_train, Y_train, 3)
#result1 = trainer1.predict(X_valid)


trainer2 = mt.multiclass_trainer()
trainer2.train(X_train, Y_train, 3)
#result2 = trainer2.predict(X_valid)

trainers = [trainer1, trainer2]
x = X_valid






