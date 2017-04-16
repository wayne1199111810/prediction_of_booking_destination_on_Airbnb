from utility import *
import binary_trainer as bt

class Bag:
	def bagOfBinaryTrainers(X_train, Y_train, bag_size, k):
		trainers = []
		for j in range(bag_size):
			x, y = Bag.subBag(X_train, Y_train)
			trainer = bt.BinaryTrainer()
			trainer.train(x, y, k)
			result = trainer.predict(x)
			accuracy = binaryEvaluation(np.ravel(result), np.ravel(y))
			if accuracy > 0.5:
				trainers.append(trainer)
		return trainers

	def predictFromBinaryTrainers(x, trainers):
		num_of_instance = x.shape[0]
		set_of_result = np.zeros((num_of_instance, len(trainers)))
		for i in range(len(trainers)):
			# res_lg = trainers[i].lg_trainer.predict(x)
			# res_svm = trainers[i].svm_trainer.predict(x)
			# res_nb = trainers[i].nb_trainer.predict(x)
			# result_of_one_trainers = Bag.binaryVote(res_lg, res_svm, res_nb)
			result_of_one_trainers = trainers[i].predict(x)
			set_of_result[:, i] = result_of_one_trainers.T
		result = Bag.voteFromBinaryTrainers(set_of_result)
		return result

	def binaryVote(res1, res2, res3):
		num = len(res1)
		result = np.zeros((num,1))
		for i in range(num):
			if res1[i]+res2[i]+res3[i] >= 2:
				result[i , 0] = 1
		return result

	def voteFromBinaryTrainers(set_of_result):
		num_of_sample = set_of_result.shape[0]
		result = np.zeros((num_of_sample, 1))
		for i in range(num_of_sample):
			if sum(set_of_result[i, :]) > int(set_of_result.shape[1] / 2):
				result[i, 0] = 1
		return result

	def subBag(instance, label, ratio=1.0):
		mu, sigma, number_of_instance = 0.5, 0.15, instance.shape[0]
		bag_size = int(round(0.2 + 0.8 * (np.random.normal(mu, sigma, 1) * number_of_instance)[0]))
		ran_idx = np.random.randint(number_of_instance, size = bag_size)
		return instance[ran_idx], label[ran_idx]