# import matplotlib.pyplot as plt
import fileRead

def main():
	destination, users = fileRead.read()
	# print('main')
	# print(users.T[0])
	# a = np.histogram(users.T[0], bins=1)
	print(users.shape)
	# plt.hist(a, )
	# plt.show()

if __name__ == "__main__":
	main()
