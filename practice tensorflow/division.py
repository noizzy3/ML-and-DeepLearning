import tensorflow as tf 
import numpy as np 
import random
import matplotlib.pyplot as plt

a = 2.0
b = 2.0
g = -4.0

#x_ = np.concatenate((np.arange(-10,b-0.5,0.001), np.arange(b+0.5,10,0.001)), axis=0)
x_ = np.arange(-10,b-0.5,0.001)
x = np.asarray([[x] for x in x_])
#y = 5*x + 8
y = a + g/(x - b)

plt.plot(x_, y)
class graph(): #predict and update
	def __init__(self):
		self.input = tf.placeholder(tf.float32,[20,1])
		self.target = tf.placeholder(tf.float32,[20,1])
		self.alpha = tf.Variable([0.0], dtype=tf.float32, name='alpha')
		self.beta = tf.Variable([0.0], dtype=tf.float32, name='beta')
		self.w = tf.Variable([0.0], dtype=tf.float32, name='w')
		self.lamb_ = tf.Variable([1.0], dtype=tf.float32, name='lambda')
		self.lamb = tf.clip_by_value(self.lamb_, 0.0, 1.0)
		self.output = (self.w*self.input - self.beta)*(self.target - self.alpha) - 1
		#self.output = (self.w*self.input - self.beta)*(self.target - self.alpha) - 1 - self.lamb*self.w*self.input*self.target
		self.cost = tf.reduce_mean(self.output**2)
		self.cost_reg = tf.reduce_mean(self.output**2) + 0.0001*self.lamb*(1-self.lamb)

		self.loss = tf.train.GradientDescentOptimizer(0.0001).minimize(self.cost)
		self.loss_reg = tf.train.AdamOptimizer(0.01).minimize(self.cost)

	def predict(self, input):
		return sess.run(self.output, feed_dict = {self.input:input})

	def update(self, input, target):
		sess.run(self.loss, feed_dict = {self.input:input, self.target:target})

	def update_reg(self, input, target):
		sess.run(self.loss_reg, feed_dict = {self.input:input, self.target:target})


g = graph()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	batch_size = 20

	total = 5000
	for epochs in range(total):
		temp = zip(list(x),list(y))
		random.shuffle(temp)
		x, y = zip(*temp)
		x = np.asarray(x)
		y = np.asarray(y)

		l = random.sample(range(x_.size - 1), batch_size)
		for i in range(int(x.size/batch_size)-1):
			g.update(x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
		if epochs%10 == 0:
			print "[",epochs*100.0/total,"%] ", sess.run(g.cost, feed_dict = {g.input:x[l], g.target:y[l]}),\
			"alpha: ", sess.run(g.alpha)[0], "beta: ", sess.run(g.beta)[0], "w: ", sess.run(g.w)[0], "lambda: ", sess.run(g.lamb)[0]

	alpha = sess.run(g.alpha)[0]
	beta = sess.run(g.beta)[0]
	w = sess.run(g.w)[0]
	
	#plt.plot(x_, alpha + 1/(w*x_ - beta) + l*w*x_*y)
	#plt.show()