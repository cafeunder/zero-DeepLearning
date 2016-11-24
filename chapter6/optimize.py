import random

# テスト関数
def function(x, y):
	return (1 / 20) * x * x + y * y

# テスト関数の勾配
def gradient(grads, x, y):
	grads["x"] = 1 / 10 * x
	grads["y"] = 2 * y

def optimize(optimizer):
	epsilon = 1e-4
	limit = int(1e+5)

	gen = 0
	noOfTrial = 100
	# 10回試行の平均
	for trial in range(noOfTrial):
		# 最適化メイン
		p = {"x": random.random(), "y": random.random()}
		g = {}

		for i in range(limit):
			# 勾配を求めてパラメータを更新
			gradient(g, p["x"], p["y"])
			optimizer.update(p, g)

			# 終了判定
			if function(p["x"], p["y"]) < epsilon:
				gen += i
				break
		else: gen += limit

	print(gen / noOfTrial);