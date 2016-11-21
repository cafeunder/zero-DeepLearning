import numpy as np

def softmax(x):
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis=0)
		y = np.exp(x) / np.sum(np.exp(x), axis=0)
		return y.T

	m = np.max(x)
	return np.exp(x - m) / np.sum(np.exp(x - m))

if __name__ == "__main__":
	print(softmax(np.array([1010, 1000, 990])))