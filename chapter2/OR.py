import numpy as np

def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	if x.dot(w) <= 0:
		return 0
	else:
		return 1

if __name__ == '__main__':
	print(OR(0, 0))
	print(OR(0, 1))
	print(OR(1, 0))
	print(OR(1, 1))
