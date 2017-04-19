import numpy.matlib as matlab
from datetime import date
import numpy as np
import math

ageSection = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

class User:
	def __init__(self, idd, account_created, first_active, first_booking, gender, age,
	 sign_up_method, signup_flow, language, affiliate_channel, affiliate_provider,
	  first_affiliate_tracked, signup_app, first_device_type, first_browser, country_destination):

		def getFirstBooking(first_booking):
			if first_booking == '':
				return False
			else:
				return True

		def getTimestamp(data):
			segment = 6 + 6 + 6	# year, month, hour
			timestamp = matlab.zeros((1, segment)).astype(int)
			year = int(data[0:4])
			month = int(data[4:6])
			hour = int(data[8:10])
			if year == 2009:
				timestamp[0, 0] = 1
			elif year == 2010:
				timestamp[0, 1] = 1
			elif year == 2011:
				timestamp[0, 2] = 1
			elif year == 2012:
				timestamp[0, 3] = 1
			elif year == 2013:
				timestamp[0, 4] = 1
			elif year == 2014:
				timestamp[0, 5] = 1
			if month == 1 or month == 2:
				timestamp[0, 6] = 1
			elif month == 3 or month == 4:
				timestamp[0, 7] = 1
			elif month == 5 or month == 6:
				timestamp[0, 8] = 1
			elif month == 7 or month == 8:
				timestamp[0, 9] = 1
			elif month == 9 or month == 10:
				timestamp[0, 10] = 1
			elif month == 11 or month == 12:
				timestamp[0, 11] = 1
			if hour >= 0 or hour < 4:
				timestamp[0, 12] = 1
			elif hour >= 4 or hour < 8:
				timestamp[0, 13] = 1
			elif hour >= 8 or hour < 12:
				timestamp[0, 14] = 1
			elif hour >= 12 or hour < 16:
				timestamp[0, 15] = 1
			elif hour >= 16 or hour < 20:
				timestamp[0, 16] = 1
			elif hour >= 20 or hour <= 24:
				timestamp[0, 17] = 1
			return timestamp

		def getDiffBookAndActive(first_booking, active):
			segment = 2
			diff = matlab.zeros((1, segment)).astype(int)
			field = list(map(int, first_booking.split('-')))
			date_first_booking = date(field[0], field[1], field[2])			
			
			date_active = date(int(active[0:4]), int(active[4:6]), int(active[6:8]))
			date_diff = (date_first_booking - date_active).days
			if date_diff == 0:
				diff[0, 0] = 1
			else:
				diff[0, 1] = 1
			return diff

		def getDiffBookAndCreate(first_booking, create):
			segment = 3
			diff = matlab.zeros((1, segment)).astype(int)

			field = list(map(int, first_booking.split('-')))
			date_first_booking = date(field[0], field[1], field[2])
			field = list(map(int, create.split('-')))
			date_create = date(field[0], field[1], field[2])

			date_diff = (date_first_booking - date_create).days
			if date_diff == 0:
				diff[0, 0] = 1
			elif date_diff > 0:
				diff[0, 1] = 1
			else:
				diff[0, 2] = 1
			return diff

		def getGender(data):
			segment = 4
			gender = matlab.zeros((1, segment)).astype(int)
			if 'MALE' == data:
				gender[0, 0] = 1
			elif 'FEMALE' == data:
				gender[0, 1] = 1
			elif '-unknown-' == data:
				gender[0, 2] = 1
			else: # other
				gender[0, 3] = 1
			return gender

		def getAgeSection(data):
			global ageSection
			if data > 100 or data <= 4:
				return -1
			for i in range(len(ageSection)):
				if data < ageSection[i]:
					return i - 1

		def getAge(data):
			if data != '':
				data = getAgeSection(int(float(data)))
			else:
				data = -1
			global ageSection
			segment = len(ageSection)
			age = matlab.zeros((1, segment)).astype(int)
			if data == -1:
				age[0, segment - 1] = 1
			else:
				age[0, data] = 1
			return age

		def getSignUpMethod(data):
			segment = 3
			sign_up_method = matlab.zeros((1, segment)).astype(int)
			if data == 'facebook':
				sign_up_method[0, 0] = 1
			elif data == 'basic':
				sign_up_method[0, 1] = 1
			elif data == 'google':
				sign_up_method[0, 2] = 1
			else:
				print('sign up: ' + data)
			return sign_up_method

		def getFLow(signup_flow):
			return np.matrix([int(float(signup_flow))])

		def getLanguage(data):
			segment = 6
			language = matlab.zeros((1, segment)).astype(int)
			if data == 'en':
				language[0, 0] = 1
			elif data == 'zh' or data == 'ko' or data == 'ja' \
			or data == 'id' or data == 'th':
				language[0, 1] = 1
			elif data == 'fr' or data == 'es' or data == 'it' \
			or data == 'pt' or data == 'ca':
				language[0, 2] = 1
			elif data == 'ru' or data == 'pl' or data == 'fi' \
			or data == 'cs':
				language[0, 3] = 1
			elif data == 'de' or data == 'sv' or data == 'nl' \
			or data == 'da' or data == 'no' or data == 'is':
				language[0, 4] = 1	
			elif data == 'hu' or data == 'hr' or data == 'tr' \
			or data == 'el':
				language[0, 5] = 1
			return language

		def getChannel(data):
			segment = 8
			channel = matlab.zeros((1, segment)).astype(int)
			if data == 'direct':
				channel[0, 0] = 1
			elif data == 'seo':
				channel[0, 1] = 1
			elif data == 'sem-non-brand':
				channel[0, 2] = 1
			elif data == 'content':
				channel[0, 3] = 1
			elif data == 'other':
				channel[0, 4] = 1
			elif data == 'sem-brand':
				channel[0, 5] = 1
			elif data == 'remarketing':
				channel[0, 6] = 1
			elif data == 'api':
				channel[0, 7] = 1
			else:
				print('channel: ' + data)
			return channel

		def getProvider(data):
			segment = 15
			affiliate_provider = matlab.zeros((1, segment)).astype(int)
			if data == 'direct': # Airbnb
				affiliate_provider[0, 0] = 1
			elif data == 'google':
				affiliate_provider[0, 1] = 1
			elif data == 'craigslist':
				affiliate_provider[0, 2] = 1
			elif data == 'facebook' or data == 'facebook-open-graph':
				affiliate_provider[0, 3] = 1
			elif data == 'vast':
				affiliate_provider[0, 4] = 1
			elif data == 'bing':
				affiliate_provider[0, 5] = 1
			elif data == 'meetup':
				affiliate_provider[0, 6] = 1
			elif data == 'yahoo':
				affiliate_provider[0, 7] = 1
			elif data == 'email-marketing':
				affiliate_provider[0, 8] = 1
			elif data == 'padmapper':
				affiliate_provider[0, 9] = 1
			elif data == 'gsp':
				affiliate_provider[0, 10] = 1
			elif data == 'wayn':
				affiliate_provider[0, 11] = 1
			elif data == 'baidu':
				affiliate_provider[0, 12] = 1
			elif data == 'yandex':
				affiliate_provider[0, 13] = 1
			else: # data == 'naver'
				affiliate_provider[0, 14] = 1
			return affiliate_provider

		def getrack(data):
			segment = 8
			track = matlab.zeros((1, segment)).astype(int)
			if data == 'untracked':
				track[0, 0] = 1
			elif data == 'omg':
				track[0, 1] = 1
			elif data == 'linked':
				track[0, 2] = 1
			elif data == 'untracked':
				track[0, 3] = 1
			elif data == 'tracked-other':
				track[0, 4] = 1
			elif data == 'product':
				track[0, 5] = 1
			elif data == 'marketing':
				track[0, 6] = 1
			elif data == 'ops' or data == 'local ops':
				track[0, 6] = 1
			elif data == '':
				track[0, 7] = 1
			else:
				print('track:' + data)
				track[0, 7] = 1
			return track

		def getAppleLoyal(signup_app, first_device_type, first_browser):
			segment = 2
			loyal = matlab.zeros((1, segment)).astype(int)
			if signup_app == 'iOS' or first_device_type == 'Mac Desktop' or first_device_type == 'iPhone' or \
			first_device_type == 'iPad' or first_browser == 'Safari' or first_browser == 'Mobile Safari':
				loyal[0, 0] = 1
			else:
				loyal[0, 1] = 1
			return loyal

		def getMobileDevice(data):
			segment = 2
			mobiledevice = matlab.zeros((1, segment)).astype(int)
			if data == 'Android Phone' or data == 'Android Tablet' or data == 'iPhone' \
			or data == 'iPad':
				mobiledevice[0, 0] = 1
			else:
				mobiledevice[0, 1] = 1
			return mobiledevice

		def getCountry(country_destination):
			return country_destination
				
		self.first_booking = getFirstBooking(first_booking)
		if self.first_booking:
			self.timestamp = getTimestamp(first_active)
			self.diff_book_active = getDiffBookAndActive(first_booking, first_active)
			self.diff_book_create = getDiffBookAndCreate(first_booking, account_created)
			self.gender = getGender(gender)
			self.age = getAge(age)
			self.sign_up_method = getSignUpMethod(sign_up_method)
			self.signup_flow = getFLow(signup_flow)
			self.language = getLanguage(language)
			self.affiliate_channel = getChannel(affiliate_channel)
			self.affiliate_provider = getProvider(affiliate_provider)
			self.first_affiliate_tracked = getrack(first_affiliate_tracked)
			self.loyal = getAppleLoyal(signup_app, first_device_type, first_browser)
			self.mobile = getMobileDevice(first_device_type)
		self.country_destination = getCountry(country_destination)


	def getData(self):
		feature = self.timestamp
		feature = np.concatenate((feature, self.diff_book_active), axis=1)
		feature = np.concatenate((feature, self.diff_book_create), axis=1)
		feature = np.concatenate((feature, self.gender), axis=1)
		feature = np.concatenate((feature, self.age), axis=1)
		feature = np.concatenate((feature, self.sign_up_method), axis=1)
		feature = np.concatenate((feature, self.signup_flow), axis=1)
		feature = np.concatenate((feature, self.language), axis=1)
		feature = np.concatenate((feature, self.affiliate_channel), axis=1)
		feature = np.concatenate((feature, self.affiliate_provider), axis=1)
		feature = np.concatenate((feature, self.first_affiliate_tracked), axis=1)
		feature = np.concatenate((feature, self.loyal), axis=1)
		feature = np.concatenate((feature, self.mobile), axis=1)
		label =  np.matrix([self.country_destination])
		return label, feature

