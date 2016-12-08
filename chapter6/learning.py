import numpy as np
from dataset.mnist import load_mnist
from chapter5.twoLayerNet import TwoLayerNet
from chapter6.Adam import Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

optimizer = Adam()
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
train_loss = []

for i in range(max_iterations):
	# train_size-1 までを batch_size個ランダムに取ってくる
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	grads = network.gradient(x_batch, t_batch)
	optimizer.update(network.params, grads)

	loss = network.loss(x_batch, t_batch)
	train_loss.append(loss)

	if i % 100 == 0:
		print("step {0} : {1}".format(i, loss))