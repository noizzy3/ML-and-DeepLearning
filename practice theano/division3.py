import theano
import theano.tensor as T
import numpy as np 
import random
import matplotlib.pyplot as plt
import lasagne.updates as Lupdates

k = -1.0
h = 2.0
g = 1.0
x_train = np.arange(h + 0.1,5 ,0.01)
#x_train = np.concatenate((np.arange(-10,int(h)-1,0.001), np.arange(int(h)+1,10,0.001)), axis=0)
y_train = 5 + 0*x_train
#y_train = k + g/(x_train - h)

plt.plot(x_train, y_train)
#plt.show()

def activation(w,x,b, thresh):
	return T.switch(w*x + b < thresh, 0, 1/(w*x + b))
lr = 0.0001
x = T.dvector()
y = T.dvector()
w = theano.shared(-1.0, name="weight")
b = theano.shared(0.0, name="bias")
b2 = theano.shared(0.0, name="bias2")
thresh = T.dscalar()


val = b2 + activation(w, x, b, thresh)
#cost = T.mean((val - y)**2) + T.mean((w*x +b)*(w*x+b < 0.01))
cost = T.mean((val - y)**2 - (w*x + b - thresh)*(w*x + b <= thresh))

grad_w, grad_b, grad_b2 = T.grad(cost, [w, b, b2])
train = theano.function(
		inputs=[x,y,thresh],
		outputs=[cost],
		updates=((w, w - lr*grad_w), (b, b - lr*grad_b), (b2, b2 - lr*grad_b2)))

total = 1000000
batch_size = 20
with open("div.txt", 'w') as FILE:
	for epoch in range(total):
		temp = zip(list(x_train),list(y_train))
		random.shuffle(temp)
		x, y = zip(*temp)
		x_train = np.asarray(x)
		y_train = np.asarray(y)
		loss = 0
		for i in range(int(x_train.size/batch_size)-1):
			loss = train(x_train[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size], 1.0/np.sqrt(epoch+1))
		if epoch % 1000 == 0:
			print "%3.2f " % (100*float(epoch)/total), "loss: %5.5f" % (loss[0]), "w: %5.5f" % (w.get_value()), "b: %5.5f" % (b.get_value()), "b2: %5.5f" % (b2.get_value())
