import numpy as np

def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)

	if x.ndim == 1:
		for c in range(x.shape[0]):
			tmp_val = x[c]

			x[c] = tmp_val + h
			fxh1 = f(x)

			x[c] = tmp_val - h
			fxh2 = f(x)

			grad[c] = (fxh1 - fxh2) / (2*h)
			x[c] = tmp_val
	elif x.ndim == 2:
		for c in range(x.shape[0]):
			for r in range(x.shape[1]):
				tmp_val = x[c, r]

				x[c, r] = tmp_val + h
				fxh1 = f(x)

				x[c, r] = tmp_val - h
				fxh2 = f(x)

				grad[c, r] = (fxh1 - fxh2) / (2*h)
				x[c, r] = tmp_val

	return grad
