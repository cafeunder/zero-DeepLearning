import numpy as np
from chapter6.optimize import optimize

class AdaGrad:
	def __init__(self, lr=0.01):
		self.lr = lr
		self.h = None

	def update(self, params, grads):
		if self.h is None:
			self.h = {}
			for key, val in params.items():
				self.h[key] = np.zeros_like(val)

		for key in params.keys():
			self.h[key] += grads[key] * grads[key]
			sqrt = np.sqrt(self.h[key] + 1e-7)
			params[key] -= self.lr * grads[key] / sqrt

if __name__ == "__main__":
	optimize(AdaGrad())