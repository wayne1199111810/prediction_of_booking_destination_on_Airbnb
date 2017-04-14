import numpy as np
import numpy.matlib as matlab
import math

ageSection = [0, 15, 20, 23, 26, 29, 32, 35, 38, 41, 45, 50, 60, 70, 100]

class User:
	def __init__(self, idd, account_created, first_active, first_booking, gender, age,
	 sign_up_method, signup_flow, language, affiliate_channel, affiliate_provider,
	  first_affiliate_tracked, signup_app, first_device_type, first_browser, country_destination):

		def getFirstBooking(first_booking):
			if first_booking == '':
				return False
			else:
				return True

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
			if data > 100 or data <= 0:
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
			segment = 25
			language = matlab.zeros((1, segment)).astype(int)
			if data == 'en':
				language[0, 0] = 1
			elif data == 'zh':
				language[0, 1] = 1
			elif data == 'fr':
				language[0, 2] = 1
			elif data == 'es':
				language[0, 3] = 1
			elif data == 'de':
				language[0, 4] = 1
			elif data == 'ko':
				language[0, 5] = 1
			elif data == 'it':
				language[0, 6] = 1
			elif data == 'pt':
				language[0, 7] = 1
			elif data == 'ja':
				language[0, 8] = 1
			elif data == 'ru':
				language[0, 9] = 1
			elif data == 'pl':
				language[0, 10] = 1
			elif data == 'el':
				language[0, 11] = 1
			elif data == 'sv':
				language[0, 12] = 1
			elif data == 'nl':
				language[0, 13] = 1
			elif data == 'hu':
				language[0, 14] = 1
			elif data == 'da':
				language[0, 15] = 1
			elif data == 'id':
				language[0, 16] = 1
			elif data == 'fi':
				language[0, 17] = 1
			elif data == 'tr':
				language[0, 18] = 1
			elif data == 'th':
				language[0, 19] = 1
			elif data == 'cs':
				language[0, 20] = 1
			elif data == 'ca':
				language[0, 21] = 1
			elif data == 'no':
				language[0, 22] = 1
			elif data == 'hr':
				language[0, 23] = 1
			elif data == 'is':
				language[0, 24] = 1
			else:
				print('language: ' + data)
				language[0, 25] = 1
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
			segment = 1
			loyal = matlab.zeros((1, segment)).astype(int)
			if signup_app == 'iOS' or first_device_type == 'Mac Desktop' or first_device_type == 'iPhone' or \
			first_device_type == 'iPad' or first_browser == 'Safari' or first_browser == 'Mobile Safari':
				loyal[0, 0] = 1
			return loyal

		def getCountry(country_destination):
			return country_destination
				
		self.first_booking = getFirstBooking(first_booking)
		self.gender = getGender(gender)
		self.age = getAge(age)
		self.sign_up_method = getSignUpMethod(sign_up_method)
		self.signup_flow = getFLow(signup_flow)
		self.language = getLanguage(language)
		self.affiliate_channel = getChannel(affiliate_channel)
		self.affiliate_provider = getProvider(affiliate_provider)
		self.first_affiliate_tracked = getrack(first_affiliate_tracked)
		self.loyal = getAppleLoyal(signup_app, first_device_type, first_browser)
		self.country_destination = getCountry(country_destination)


	def getData(self):
		feature = self.gender
		feature = np.concatenate((feature, self.age), axis=1)
		feature = np.concatenate((feature, self.sign_up_method), axis=1)
		feature = np.concatenate((feature, self.signup_flow), axis=1)
		feature = np.concatenate((feature, self.language), axis=1)
		feature = np.concatenate((feature, self.affiliate_channel), axis=1)
		feature = np.concatenate((feature, self.affiliate_provider), axis=1)
		feature = np.concatenate((feature, self.first_affiliate_tracked), axis=1)
		feature = np.concatenate((feature, self.loyal), axis=1)
		label =  np.matrix([self.country_destination])
		return label, feature

