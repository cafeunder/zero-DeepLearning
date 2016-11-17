import numpy as np
from chapter3.softmax import softmax
from chapter4.crossEntropyError import cross_entropy_error
from chapter4.numericalGradient import numerical_gradient

class SimpleNet:
	def __init__(self):
		self.W = np.random.randn(2, 3)

	def predict(self, x):
		return np.dot(x, self.W)

	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y, t)

		return loss

if __name__ == "__main__":
	net = SimpleNet()
	print(net.W)

	x = np.array([0.6, 0.9])
	p = net.predict(x)
	print(p)

	t = np.array([0, 0, 1])
	l = net.loss(x, t)
	print(l)

	def f(W):
		return net.loss(x, t)

	# 目的関数 <- 損失関数
	dW = numerical_gradient(f, net.W)
	print(dW)