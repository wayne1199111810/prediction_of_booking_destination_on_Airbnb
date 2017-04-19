# import matplotlib.pyplot as plt
from include import preProcessing as pp
from include.utility import *

# get data
# destination, users = pp.readRawData() 

# users, destination = pp.readFromFile('Data/users_origin.dat', 'Data/destination_origin.dat')
# pp.writeToFile(users[0:idx], destination[0:idx], 'Data/users_train.dat', 'Data/destination_train.dat', cvs = False)

size_of_training = 5000;
createNewTrainingFileWithSize(size_of_training)