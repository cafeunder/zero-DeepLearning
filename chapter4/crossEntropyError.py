import numpy as np

def cross_entropy_error(y, t):
	delta = 1e-7

	# 配列の配列になっていないなら、外側を配列で囲む
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	batch_size = y.shape[0]
	return -np.sum(t * np.log(y + delta)) / batch_size

if __name__ == "__main__" :
	t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

	y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
	print(cross_entropy_error(np.array(y), np.array(t)))

	y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]
	print(cross_entropy_error(np.array(y), np.array(t)))