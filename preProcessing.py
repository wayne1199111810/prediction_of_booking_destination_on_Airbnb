import User as U
import numpy as np
import csv

raw_data = 'Data/train_users_2.csv'

def read():
	def getUserInfo(row):
		u = U.User(row['id'], row['date_account_created'], row['timestamp_first_active'], row['date_first_booking'], row['gender'], row['age'], row['signup_method'], row['signup_flow'], row['language'], row['affiliate_channel'], row['affiliate_provider'], row['first_affiliate_tracked'], row['signup_app'], row['first_device_type'], row['first_browser'], row['country_destination'])
		if u.first_booking:
			return u.getData()
		else:
			return None, None
	with open(raw_data, 'r') as csvfile:
		nIters = 10000	
		reader = csv.DictReader(csvfile)		
		users = np.matrix([])
		destination = np.matrix([])
		i = 0
		for row in reader:
			label, features = getUserInfo(row)
			if label is None:
				i += 1
				continue
			if users.shape[1] == 0:
				users = features
				destination = label
			else:
				users = np.concatenate((users, features), axis=0)
				destination = np.concatenate((destination, label), axis=0)
			# i += 1
			# if i >= nIters:
			# 	break
		return destination, users

def writeToFile():
	destination, users = read()
	users.dump("Data/users.dat")
	destination.dump("Data/destination.dat")
	print('from write: ' + str(users.shape) + str(destination.shape))

def readFromFile(x_train, y_train):
	users = np.load(x_train)
	destination = np.load(y_train)
	print('from read: ' + str(users.shape) + str(destination.shape))
	return users, destination