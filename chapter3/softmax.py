import numpy as np

def softmax(x):
	m = np.max(x)
	return np.exp(x - m) / np.sum(np.exp(x - m))

if __name__ == "__main__":
	print(softmax(np.array([1010, 1000, 990])))