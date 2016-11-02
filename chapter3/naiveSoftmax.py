import numpy as np

def softmax(x):
	exp_x = np.exp(x)
	sum_exp_x = np.sum(exp_x)
	return exp_x / sum_exp_x

if __name__ == "__main__":
	print(softmax(np.array([0.3, 2.9, 4.0])));