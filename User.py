import numpy as np
class User:
	def __init__(self, idd, account_created, first_active, first_booking, gender, age,
	 sign_up_method, signup_flow, language, affiliate_channel, affiliate_provider,
	  first_affiliate_tracked, signup_app, first_device_type, first_browser, country_destination):
		def translateGender(gender):
			if 'MALE' == gender:
				return 1
			elif 'FEMALE' == gender:
				return 2
			elif '-unknown-' == gender:
				return 3
			else:
				return 0
		def translateLanguage(language):
			if language == 'en':
				return 1
			elif language == 'zh':
				return 2
			elif language == 'fr':
				return 3
			elif language == 'es':
				return 4
			elif language == 'de':
				return 5
			elif language == 'ko':
				return 6
			else:
				return 0
		def translateCountry(country_destination):
			if country_destination == 'NDF':
				return 1
			elif country_destination == 'US':
				return 2
			elif country_destination == 'FR':
				return 3
			elif country_destination == 'IT':
				return 4
			elif country_destination == 'GB':
				return 5
			elif country_destination == 'ES':
				return 6
			elif country_destination == 'CA':
				return 7
			elif country_destination == 'other':
				return 0
		self.id = idd
		self.account_created = account_created
		self.first_active = first_active
		self.first_booking = first_booking
		self.gender = translateGender(gender)
		self.age = age
		self.sign_up_method = sign_up_method
		self.signup_flow = signup_flow
		self.language = translateLanguage(language)
		self.affiliate_channel = affiliate_channel
		self.affiliate_provider = affiliate_provider
		self.first_affiliate_tracked = first_affiliate_tracked
		self.signup_app = signup_app
		self.first_device_type = first_device_type
		self.first_browser = first_browser
		self.country_destination = translateCountry(country_destination)

	def getData(self):
		info = np.matrix([self.gender, self.language, self.country_destination])
		return info