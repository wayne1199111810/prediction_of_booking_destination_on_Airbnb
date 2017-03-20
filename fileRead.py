import User as U
import matplotlib.pyplot as plt
import csv
def read():
	def getUserInfo(row):
		u = U.User(row['id'], row['date_account_created'], row['timestamp_first_active'], row['date_first_booking'], row['gender'], row['age'], row['signup_method'], row['signup_flow'], row['language'], row['affiliate_channel'], row['affiliate_provider'], row['first_affiliate_tracked'], row['signup_app'], row['first_device_type'], row['first_browser'], row['country_destination'])
		return u.getDate();
	with open('Data/train_users_2.csv', 'r') as csvfile:
		reader = csv.DictReader(csvfile)		
		users = []
		users = [U.User(row['id'], row['date_account_created'], row['timestamp_first_active'], row['date_first_booking'], row['gender'], row['age'], row['signup_method'], row['signup_flow'], row['language'], row['affiliate_channel'], row['affiliate_provider'], row['first_affiliate_tracked'], row['signup_app'], row['first_device_type'], row['first_browser'], row['country_destination']) for row in reader]
		return users