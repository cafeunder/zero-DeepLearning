import numpy as np
import matplotlib.pylab as plt
from chapter4.numericalDiff import numerical_diff

def function_1(x):
	# 0.01(x^2) + 0.1x
	return 0.01*x**2 + 0.1*x

def numerical_diff_plot(x, func):
	f = func(x)
	df = numerical_diff(x, func)

	r = plt.axis() # range <- [xmin, xmax, ymin, ymax]
	px = np.linspace(r[0], r[1], 3) # xmin から xmax まで3等分
	py = df * (px - x) + f # 直線の方程式
	plt.plot(px, py, "g-")

if __name__ == "__main__":
	x = np.arange(0.0, 20.0, 0.1)
	y = function_1(x)
	plt.xlabel("x")
	plt.ylabel("f(x)")
	plt.ylim(0, 6);
	plt.plot(x, y, "b-")
	numerical_diff_plot(10, function_1)
	plt.show()