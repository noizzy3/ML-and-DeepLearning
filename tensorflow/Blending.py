import tensorflow as tf 
import numpy as np 
import random
import matplotlib.pyplot as plt
t = 1.0
x_ = np.arange(-1*t,t, 2*t/1000)
x = np.asarray([[x] for x in x_])
y = np.sin(5*x)
#y = 2*x

i = 10*x
#plt.plot(np.resize(i,(1000)), np.resize(2*i,(1000)))
plt.plot(np.resize(i,(1000)), np.resize(np.sin(5*i),(1000)))

class graph(): #predict and update
	def __init__(self):
		self.input = tf.placeholder(tf.float32,[20,1])
		self.target = tf.placeholder(tf.float32,[20,1])
		self.b_ = tf.Variable([0.5], dtype=tf.float32, name='beta')
		self.b = tf.clip_by_value(self.b_, 0.0, 1.0)
		self.b2 = tf.Variable([0], dtype=tf.float32, name='bias2')
		self.b3 = tf.Variable([0], dtype=tf.float32, name='bias3')
		self.w1 = tf.Variable([1], dtype=tf.float32, name='w1')
		#self.w1 = tf.maximum(self.w1_, 0)
		self.w2= tf.Variable([1], dtype=tf.float32, name='w2')
		
		self.output = self.w2*(self.b*(self.w1*self.input + self.b2) + (1.0-self.b)*tf.sin(self.w1*self.input + self.b2))
		
		self.cost_initial = tf.reduce_mean((self.output - self.target)**2)  
		#self.cost_initial = tf.reduce_mean((self.output - self.target)**2 + 3*(self.w1*self.input + self.b2 - self.input)**2)  
		self.cost_later = tf.reduce_mean((self.output - self.target)**2)  
		self.cost_later_reg = tf.reduce_mean((self.output - self.target)**2) + 0.0001*self.b*(1.0-self.b)
		
		t_vars = tf.trainable_variables()
		self.list_w1 = [var for var in t_vars if 'w1' in var.name]
		self.list_nobeta_now1 = [var for var in t_vars if not ('beta' in var.name or 'w1' in var.name)]
		self.list_betaonly = [var for var in t_vars if 'beta' in var.name]

		self.loss_initial = tf.train.AdamOptimizer(0.001).minimize(self.cost_initial, var_list=self.list_nobeta_now1)
		self.loss_initial_w1 = tf.train.AdamOptimizer(0.01).minimize(self.cost_initial, var_list=self.list_w1)
		self.loss_initial_beta = tf.train.AdamOptimizer(0.0001).minimize(self.cost_later, var_list=self.list_betaonly)
		self.loss_later = tf.train.AdamOptimizer(0.001).minimize(self.cost_later)
		self.loss_later_reg = tf.train.AdamOptimizer(0.0001).minimize(self.cost_later_reg)

	def predict(self, input):
		return sess.run(self.output, feed_dict = {self.input:input})

	def update_initial(self, input, target):
		sess.run(self.loss_initial, feed_dict = {self.input:input, self.target:target})

	def update_initial_beta(self, input, target):
		sess.run(self.loss_initial_beta, feed_dict = {self.input:input, self.target:target})

	def update_initial_w1(self, input, target):
		sess.run(self.loss_initial_w1, feed_dict = {self.input:input, self.target:target})

	def update_later(self, input, target):
		sess.run(self.loss_later, feed_dict = {self.input:input, self.target:target})

	def update_later_reg(self, input, target):
		sess.run(self.loss_later_reg, feed_dict = {self.input:input, self.target:target})

g = graph()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	batch_size = 20
	x_train = np.ones((batch_size,1))
	y_train = np.ones((batch_size,1))

	total = 1000000
	for epochs in range(total):
		l = random.sample(range(999), batch_size)
		if epochs < 0.9*total:
			beta = sess.run(g.b)[0]
			if epochs < 0.3*total:
				g.update_initial(x[l], y[l])
				g.update_initial_w1(x[l], y[l])
				g.update_initial_beta(x[l], y[l])
				
			else:
				g.update_later(x[l],y[l])
			'''
			if epochs < 0.3*total:
				g.update_initial(x[l], y[l])
			else:
				g.update_later(x[l],y[l])

			if epochs > 100 and epochs%50 == 0:
				g.update_later(x[l],y[l])
			'''
			if epochs%10000 == 0:
				print "[",epochs*100.0/total,"%] ", sess.run(g.cost_later, feed_dict = {g.input:x[l], g.target:y[l]}),\
				"w2: ", sess.run(g.w2)[0], "w1: ", sess.run(g.w1)[0], "b2: ", sess.run(g.b2)[0], "beta: ", sess.run(g.b)[0]
		else:
			g.update_later_reg(x[l],y[l])
			if epochs%10000 == 0:
				print "[",epochs*100.0/total,"%] ", sess.run(g.cost_later_reg, feed_dict = {g.input:x[l], g.target:y[l]}),\
				"w2: ", sess.run(g.w2)[0], "w1: ", sess.run(g.w1)[0], "bias: ", sess.run(g.b2)[0], "beta: ", sess.run(g.b)[0]

	weight = sess.run(g.w1)[0]
	weight2 = sess.run(g.w2)[0]
	bias = sess.run(g.b)[0]
	bias2 = sess.run(g.b2)[0]
	bias3 = sess.run(g.b3)[0]

	plt.plot(np.resize(i,(1000)), np.resize(weight2*(bias*(weight*i + bias2)+ (1-bias)*np.sin(weight*i + bias2)) + bias3,(1000)))
	plt.show()
	print "DONE!!"
