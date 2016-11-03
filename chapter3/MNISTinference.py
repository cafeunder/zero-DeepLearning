from chapter3.MNISTnnetwork import *

x, t = get_data()
network = init_network()

accuracy_count = 0
for i in range(len(x)):
	y = predict(network, x[i])
	ans = np.argmax(y)
	if ans == t[i]:
		accuracy_count += 1

print("Accuracy:"+str(float(accuracy_count / len(x))))