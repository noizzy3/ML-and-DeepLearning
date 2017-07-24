import theano
import theano.tensor as T
import numpy as np 
import random
import matplotlib.pyplot as plt
import lasagne.updates as Lupdates

k = 2.0
h = 2.0
g = -4.0
x_train = np.arange(-5, h-0.1,0.01)
#x_train = np.concatenate((np.arange(-10,int(h)-1,0.001), np.arange(int(h)+1,10,0.001)), axis=0)
y_train = k + g/(x_train - h)

plt.plot(x_train, y_train)
#plt.show()
lr = 0.0001
total = 100000
batch_size = 20

x = T.dvector()
y = T.dvector()
alpha = theano.shared(0.0, name="alpha")
beta = theano.shared(0.0, name="beta")
gamma = theano.shared(0.0, name="gamma")
#lamb = theano.shared(1.0, name="lambda")

val = ((gamma*x - beta)*(y - alpha) - 1)**2
#val = ((gamma*x - beta)*(y - alpha) - 1 - lamb*gamma*x*y)**2 Doesn't work because the coefficient of x*y contains both lambda and gamma and hence they can't be learnt explicitly
cost = T.mean(val)

Lupdates.adam(cost, [alpha, beta, gamma], learning_rate, epsilon=1e-04)
updates = [process_updates(*up) for up in update.items()]
train = theano.function(
		inputs=[x,y],
		outputs=[cost],
		updates=updates)

for epoch in range(total):
	temp = zip(list(x_train),list(y_train))
	random.shuffle(temp)
	x, y = zip(*temp)
	x_train = np.asarray(x)
	y_train = np.asarray(y)

	for i in range(int(x_train.size/batch_size)-1):
			loss = train(x_train[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
	if epoch % 30 == 0:
		print 100*float(epoch)/total, "loss: ", loss[0], alpha.get_value(), beta.get_value(), gamma.get_value()