import numpy as np
import logisticRegression_multi as lg 

trainer = lg.logisticRegression_multi()

prob = [[0.89, 0.53, 0.14, 0.67, 0.15, 0.45, 0.97], [0.55, 0.14, 0.67, 0.85, 0.91, 0.20, 0.36]]
prob = np.array(prob)
classes = ['US', 'TW', 'JP', 'FR', 'IT', 'ER', 'GE']


result = trainer.getTopProb(prob, classes)


print(result)


