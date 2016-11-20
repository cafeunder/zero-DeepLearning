from chapter3.softmax import softmax
from chapter4.crossEntropyError import cross_entropy_error

class SoftMaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)

		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		# データ1個あたりの誤差の期待値
		dx = (self.y - self.t) / batch_size

		return dx