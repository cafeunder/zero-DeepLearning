import random

def function(x, y):
	return (1 / 20) * x * x + y * y

def gradient(grads, x, y):
	grads["x"] = 1 / 10 * x
	grads["y"] = 2 * y

def optimize(optimizer):
	epsilon = 1e-4
	limit = int(1e+5)

	params = {"x": random.random(), "y": random.random()}
	grads = {}
	gradient(grads, params["x"], params["y"])

	for i in range(limit):
		gradient(grads, params["x"], params["y"])
		optimizer.update(params, grads)

		print(str(i) + ": " + "(" + str(params["x"]) + ", " + str(params["y"]) + ") = " + str(function(params["x"], params["y"])))
		if function(params["x"], params["y"]) < epsilon:
			return;