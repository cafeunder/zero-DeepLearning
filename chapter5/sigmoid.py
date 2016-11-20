import numpy as np

class Sigmoid:
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = 1/ (1 + np.exp(-x))
		self.out = out

		return out

	def backward(self, dout):
		# dL/dy * y(1-y)
		dx = dout * self.out * (1.0 - self.out)

		return dx