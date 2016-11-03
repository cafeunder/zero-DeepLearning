from chapter3.MNISTnnetwork import *

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_count = 0

for i in range(0, len(x), batch_size):
	x_batch = x[i:i+batch_size]
	y_batch = predict(network, x_batch)
	print(y_batch)
	p = np.argmax(y_batch, axis=1)
	print(p)
	accuracy_count += np.sum(p == t[i:i+batch_size])

print("Accuracy:"+str(float(accuracy_count / len(x))))