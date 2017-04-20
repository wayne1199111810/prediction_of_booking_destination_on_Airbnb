from include.utility import *
from include.binary import binary_trainer as bt
from include.multiclass import multiclass_trainer as mt

class Bag:
	# return bag, whose size is a Gaussian distribution 
	# with mean = 0.6, covariance = 0.15 on original 
	# data set
	def subBag(instance, label, ratio=1.0):
		mu, sigma, number_of_instance = 0.5, 0.15, instance.shape[0]
		bag_size = int(round(0.2 + 0.8 * (np.random.normal(mu, sigma, 1) * number_of_instance)[0]))
		ran_idx = np.random.randint(number_of_instance, size = bag_size)
		return instance[ran_idx], label[ran_idx]

	# Binary Classifier
	def bagOfBinaryTrainers(X_train, Y_train, bag_size, k):
		trainers = []
		for j in range(bag_size):
			x, y = Bag.subBag(X_train, Y_train)
			trainer = bt.BinaryTrainer()
			trainer.train(x, y, k)
			result = trainer.predict(x)
			accuracy = binaryEvaluation(np.ravel(result), np.ravel(y))
			# print('accuracy:' + str(accuracy))
			if accuracy > 0.5:
				trainers.append(trainer)
		return trainers

	def predictFromBinaryTrainers(x, trainers):
		num_of_instance = x.shape[0]
		set_of_result = np.zeros((num_of_instance, len(trainers)))
		for i in range(len(trainers)):
			result_of_one_trainers = trainers[i].predict(x)
			set_of_result[:, i] = result_of_one_trainers.T
		result = Bag.voteFromBinaryTrainers(set_of_result)
		return result

	def voteFromBinaryTrainers(set_of_result):
		num_of_sample = set_of_result.shape[0]
		result = np.zeros((num_of_sample, 1))
		print(set_of_result.shape)
		for i in range(num_of_sample):
			if sum(set_of_result[i, :]) / set_of_result.shape[1] >= 0.5:
				result[i, 0] = 1
		return result

	# Multi-classifier
	def bagOfMultiTrainers(X_train, Y_train, bag_size, k):
		trainers = []
		for j in range(bag_size):
			x, y = Bag.subBag(X_train, Y_train)
			trainer = mt.multiclass_trainer()
			trainer.train(x, y, k)
			result = trainer.predict(x)
			accuracy = multiclassEvaluation(result, y)
			# print('accuracy:' + str(accuracy))
			if accuracy > 0.5:
				trainers.append(trainer)
		return trainers

	def predictFromMultiTrainers(x, trainers):
		num_of_instance = len(x)
		k = 5
		result = np.empty([num_of_instance,k], dtype=object)
		res_list = []
		for trainer in trainers:
			res_list.append(trainer.predict(x))

		for i in range(num_of_instance):
			country_score = {}
			for result in res_list:
				for j in range(k):
					if result[i][j] in country_score:
						country_score[result[i][j]] = country_score[result[i][j]] + 1/(math.log(j+2,2))
					else:
						country_score[result[i][j]] = 1/(math.log(j+2,2))
			for j in range(k):
				result[i][j] = max(country_score, key=country_score.get)
				country_score[result[i][j]] = 0
		return result


