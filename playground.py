# import matplotlib.pyplot as plt
from include import preProcessing as pp
from include.utility import *
from include.multiclass import multiclass_trainer as mt
import include.crossValidation as cv
import include.utility
# get data
# destination, users = pp.readRawData() 

# users, destination = pp.readFromFile('Data/users_origin.dat', 'Data/destination_origin.dat')
# pp.writeToFile(users[0:idx], destination[0:idx], 'Data/users_train.dat', 'Data/destination_train.dat', cvs = False)

k = 8 
data = cv.CV(k)
X_train, Y_train, X_valid, Y_valid = data.iteration(2)



trainer = mt.multiclass_trainer()

trainer.train(X_train, Y_train, 5)

result = trainer.predict(X_valid)
