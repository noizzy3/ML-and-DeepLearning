import tensorflow as tf 
import numpy as np 
import random
import matplotlib.pyplot as plt

a = 5.0
b = 2.0
g = -3.0

x_ = np.concatenate((np.arange(-10,int(b)-1,0.001), np.arange(int(b)+1,10,0.001)), axis=0)
x = np.asarray([[x] for x in x_])
y = a + g/(x - b)

plt.plot(x_, y)
class graph(): #predict and update
	def __init__(self):
		self.input = tf.placeholder(tf.float32,[20,1])
		self.target = tf.placeholder(tf.float32,[20,1])
		self.alpha = tf.Variable([0.0], dtype=tf.float32, name='alpha')
		self.beta = tf.Variable([0.0], dtype=tf.float32, name='beta')
		self.gamma = tf.Variable([0.0], dtype=tf.float32, name='gamma')

		self.output = (self.input - self.beta)*(self.target - self.alpha) - self.gamma
		
		self.cost = tf.reduce_mean(self.output**2)

		t_vars = tf.trainable_variables()
		self.list_rest = [var for var in t_vars if not ('gamma' in var.name)]
		self.list_gamma = [var for var in t_vars if 'gamma' in var.name]

		self.loss = tf.train.AdamOptimizer(0.0001).minimize(self.cost, var_list=self.list_rest)
		self.loss_gamma = tf.train.AdamOptimizer(0.0001).minimize(self.cost, var_list=self.list_gamma)

	def predict(self, input):
		return sess.run(self.output, feed_dict = {self.input:input})

	def update(self, input, target):
		sess.run(self.loss, feed_dict = {self.input:input, self.target:target})

	def update_gamma(self, input, target):
		sess.run(self.loss_gamma, feed_dict = {self.input:input, self.target:target})


g = graph()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	batch_size = 20
	x_train = np.ones((batch_size,1))
	y_train = np.ones((batch_size,1))

	total = 500000
	for epochs in range(total):
		l = random.sample(range(x_.size - 1), batch_size)
		g.update(x[l],y[l])
		g.update_gamma(x[l],y[l])
		if epochs%10000 == 0:
			print "[",epochs*100.0/total,"%] ", sess.run(g.cost, feed_dict = {g.input:x[l], g.target:y[l]}),\
			"alpha: ", sess.run(g.alpha)[0], "beta: ", sess.run(g.beta)[0], "gamma: ", sess.run(g.gamma)[0]

	alpha = sess.run(g.alpha)[0]
	beta = sess.run(g.beta)[0]
	gamma = sess.run(g.gamma)[0]
	
	plt.plot(x_, alpha + gamma/(x_ - beta))
	plt.show()