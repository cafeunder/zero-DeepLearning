import numpy as np
from chapter4.gradientDescent import gradient_descent

def function_2(x):
	return x[0]**2 + x[1]**2

if __name__ == "__main__":
	init_x = np.array([-3.0, 4.0])
	grad = gradient_descent(function_2, init_x, 0.1, 100)
	print(grad)