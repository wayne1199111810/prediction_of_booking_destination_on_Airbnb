import User as U
import numpy as np
import csv

def read():
	def getUserInfo(row):
		u = U.User(row['id'], row['date_account_created'], row['timestamp_first_active'], row['date_first_booking'], row['gender'], row['age'], row['signup_method'], row['signup_flow'], row['language'], row['affiliate_channel'], row['affiliate_provider'], row['first_affiliate_tracked'], row['signup_app'], row['first_device_type'], row['first_browser'], row['country_destination'])
		if u.first_booking:
			return u.getData()
		else:
			return None, None
	with open('Data/train_users_2.csv', 'r') as csvfile:
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

		# 	if label == 0:
		# 		count_N += 1
		# 	elif label == 2:
		# 		count_U += 1
		# 	else:
		# 		count_other += 1
			# i += 1
			# if i >= 50:
			# 	break
		return destination, users