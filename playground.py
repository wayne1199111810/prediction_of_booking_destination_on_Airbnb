import crossValidation as cv
import binaryClassifier as bc
from utility import *

k = 3
data = cv.CV(k)
X_train, Y_train, X_valid, Y_valid = data.iteration(1)







