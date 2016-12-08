import numpy as np
from chapter6.optimize import optimize

class RMSprop:
	def __init__(self, lr=0.01, decay_rate=0.99):
		self.lr = lr
		self.decay_rate = decay_rate
		self.h = None

	def dispose(self):
		self.h = None

	def update(self, params, grads):
		if self.h is None:
			self.h = {}
			for key, val in params.items():
				self.h[key] = np.zeros_like(val)

		for key in params.keys():
			self.h[key] *= self.decay_rate
			self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
			params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

if __name__ == "__main__":
	optimize(RMSprop())