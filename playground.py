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

createNewTrainingFileWithSize(3000)





