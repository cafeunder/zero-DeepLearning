import numpy as np
from collections import OrderedDict
from chapter5.affine import Affine
from chapter5.relu import Relu
from chapter5.softmaxWithLoss import SoftmaxWithLoss
from chapter4.numericalGradient import numerical_gradient

class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		self.params = {}
		# 入力層 → 隠れ層
		self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.params["b1"] = np.zeros(hidden_size)
		# 隠れ層 → 出力層
		self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params["b2"] = np.zeros(output_size)

		self.layers = OrderedDict()
		self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
		self.layers["Relu1"] = Relu()
		self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

		self.lastLayer = SoftmaxWithLoss()

	def predict(self, x):
		# OrderedDict#values():順番付きで値を返す
		for layer in self.layers.values():
			x = layer.forward(x)

		return x

	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def accuracy(self, x, t):
		y = self.predict(x)
		# 最もスコアが高いラベルの配列
		y = np.argmax(y, axis=1)
		# t.ndim≠1はone-hotなので、正解ラベルの配列に直す
		if t.ndim != 1: t = np.argmax(t, axis=1)

		accuracy = np.sum(y == t) / float(x.shape[0])
		return accuracy

	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)

		grads = {}
		grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
		grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
		grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
		grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

		return grads

	def gradient(self, x, t):
		# loss->predict->forwardで順方向の重みを求める
		self.loss(x, t)

		# 逆方向
		dout = 1
		dout = self.lastLayer.backward(dout) #出力層

		# 隠れ層
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)

		grads = {}
		grads["W1"] = self.layers["Affine1"].dW
		grads["b1"] = self.layers["Affine1"].db
		grads["W2"] = self.layers["Affine2"].dW
		grads["b2"] = self.layers["Affine2"].db

		return grads

