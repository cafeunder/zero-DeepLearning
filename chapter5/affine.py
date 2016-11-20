import numpy as np

class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None

	def forward(self, x):
		self.x = x
		out = np.dot(x, self.W) + self.b

		return out

	def backward(self, dout):
		# dL/dx = dL/dy * W^
		dx = np.dot(dout, self.W.T)
		# dL/dW = x^ * dL/dy
		self.dW = np.dot(self.x.T, dout)
		# バイアスは全データ分足し合わせる
		self.db = np.sum(dout, axis=0)

		return dx