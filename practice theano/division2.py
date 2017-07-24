import theano
import theano.tensor as T
import numpy as np 
import random
import matplotlib.pyplot as plt


k = 0.0
h = 2.0
g = 1.0
x_train = np.arange(-5, h-0.1,0.01)
#x_train = np.concatenate((np.arange(-10,int(h)-1,0.001), np.arange(int(h)+1,10,0.001)), axis=0)
y_train = k + g/(x_train - h)

plt.plot(x_train, y_train)
#plt.show()

def activation(w,x,b):
	return T.switch(w*x + b < 0.01, -100*(w*x + b) + 1001.0, 1/(w*x + b))
lr = 0.0001
x = T.dvector()
y = T.dvector()
w = theano.shared(0.0, name="weight")
b = theano.shared(0.0, name="bias")

val = activation(w, x, b)
cost = T.mean((val - y)**2)
#cost = T.mean((val - y)**2) - (w*x + b)*(w*x + b < 0.01))
grad_w, grad_b = T.grad(cost, [w, b])
train = theano.function(
		inputs=[x,y],
		outputs=[cost],
		updates=((w, w - lr*grad_w), (b, b - lr*grad_b)))

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
				loss = train(x_train[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
		if epoch % 1000 == 0:
			print "%3.2f percent" % (100*float(epoch)/total), "loss: %5.5f" % (loss[0]), "w: %5.5f" % (w.get_value()), "b: %5.5f" % (b.get_value())
		weight =w.get_value()
		bias =b.get_value()
		lss = loss[0]
		FILE.write("\
epoch = %(epoch)s, loss = %(lss)s, w = %(weight)s, b = %(bias)s\n" % locals())