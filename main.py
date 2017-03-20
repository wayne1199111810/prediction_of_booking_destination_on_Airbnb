import fileRead 
import User
import numpy as np

def main():
	users = fileRead.read()
	print('main')
	#print(users)
	print(users[0].id)
	print(users[1].id)
	print(users[2].id)	

if __name__ == "__main__":
	main()
