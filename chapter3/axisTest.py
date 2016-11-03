import numpy as np

x = np.array([
	[
		[0.1, 1.0],
		[0.3, 0.2],
		[0.4, 0.5],
		[0.4, 0.5],
		[0.6, 0.5],
		[0.7, 0.8],
	],
	[
		[0.2, 2.2],
		[0.6, 0.4],
		[0.7, 2.0],
		[0.8, 1.0],
		[1.2, 1.0],
		[1.3, 1.4],
	]
])

y = np.argmax(x, axis=0)
print(y)

y = np.argmax(x, axis=1)
print(y)

y = np.argmax(x, axis=2)
print(y)
