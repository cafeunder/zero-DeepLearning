from chapter6.optimize import optimize

class SGD:
	def __init__(self, lr=0.01):
		self.lr = lr

	def update(self, params, grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key]

if __name__ == "__main__":
	optimize(SGD())