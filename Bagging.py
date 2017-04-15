import numpy as np

class Bagging:
	def subInstance(instance, label, ratio=1.0):
		mu, sigma, number_of_instance = 0.5, 0.15, instance.shape[0]
		bag_size = round((np.random.normal(mu, sigma, 1) * number_of_instance)[0])
		ran_idx = np.random.randint(number_of_instance, size = bag_size)
