import tensorflow as tf 
import numpy as np 
import random
import matplotlib.pyplot as plt
t = 1.0
x_ = np.arange(-1*t,t, 2*t/1000)
x = np.asarray([[x] for x in x_])
y = 1.0/(1.0 +np.exp(-30*x))
#y = 2*x

i = 10*x
#plt.plot(np.resize(i,(1000)), np.resize(2*i,(1000)))
plt.plot(np.resize(i,(1000)), np.resize(1.0/(1.0 +np.exp(-30*i)),(1000)))

class graph(): #predict and update
	def __init__(self):
		self.input = tf.placeholder(tf.float32,[20,1])
		self.target = tf.placeholder(tf.float32,[20,1])
		self.beta_ = tf.Variable([0.5], dtype=tf.float32, name='beta')
		self.beta = tf.clip_by_value(self.beta_, 0.0, 1.0)
		self.b_id = tf.Variable([0], dtype=tf.float32, name='bias_id')
		self.b_exp = tf.Variable([0], dtype=tf.float32, name='bias_exp')
		self.w_id = tf.Variable([1], dtype=tf.float32, name='w_id')
		#self.w1 = tf.maximum(self.w1_, 0)
		self.w_exp = tf.Variable([1], dtype=tf.float32, name='w_exp')
		
		self.output_id = self.w_id*self.input + self.b_id
		self.output_exp = 1.0/(1.0+tf.exp(self.w_exp*self.input + self.b_exp))
		self.blended_output = self.beta*self.output_id + (1.0 - self.beta)*self.output_exp
		
		self.cost_id = tf.reduce_mean((self.output_id - self.target)**2) 
		self.cost_exp = tf.reduce_mean((self.output_exp - self.target)**2)
		self.cost_blended = tf.reduce_mean((self.blended_output - self.target)**2) + 0.0001*self.beta*(1.0-self.beta)
		
		t_vars = tf.trainable_variables()
		self.list_Beta = [var for var in t_vars if 'beta' in var.name]
		self.list_NoBeta = [var for var in t_vars if not ('beta' in var.name)]

		self.loss_id = tf.train.AdamOptimizer(0.001).minimize(self.cost_id, var_list=self.list_NoBeta)
		self.loss_exp = tf.train.AdamOptimizer(0.001).minimize(self.cost_exp, var_list=self.list_NoBeta)
		self.loss_blended = tf.train.AdamOptimizer(0.001).minimize(self.cost_blended, var_list=self.list_Beta)

	def predict(self, input):
		return sess.run(self.output, feed_dict = {self.input:input})

	def update_id(self, input, target):
		sess.run(self.loss_id, feed_dict = {self.input:input, self.target:target})

	def update_exp(self, input, target):
		sess.run(self.loss_exp, feed_dict = {self.input:input, self.target:target})

	def update_beta(self, input, target):
		sess.run(self.loss_blended, feed_dict = {self.input:input, self.target:target})

g = graph()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	batch_size = 20
	x_train = np.ones((batch_size,1))
	y_train = np.ones((batch_size,1))

	total = 100000
	for epochs in range(total):
		l = random.sample(range(999), batch_size)
		if epochs < 0.75*total:
			g.update_id(x[l], y[l])
			g.update_exp(x[l], y[l])
			if epochs%1000 == 0:
				print "[",epochs*100.0/total,"%] ", "w_id: ", sess.run(g.w_id)[0], "w_exp: ", sess.run(g.w_exp)[0], "b_id: ", sess.run(g.b_id)[0], \
				"b_exp: ", sess.run(g.b_exp)[0], "beta: ", sess.run(g.beta)[0]
		else:
			g.update_beta(x[l],y[l])
			if epochs%1000 == 0:
				print "[",epochs*100.0/total,"%] ", sess.run(g.cost_blended, feed_dict = {g.input:x[l], g.target:y[l]}),\
				"w_id: ", sess.run(g.w_id)[0], "w_exp: ", sess.run(g.w_exp)[0], "b_id: ", sess.run(g.b_id)[0], \
				"b_exp: ", sess.run(g.b_exp)[0], "beta: ", sess.run(g.beta)[0]

	w_id = sess.run(g.w_id)[0]
	w_exp = sess.run(g.w_exp)[0]
	b_id = sess.run(g.b_id)[0]
	b_exp = sess.run(g.b_exp)[0]
	beta = sess.run(g.beta)[0]

	plt.plot(np.resize(i,(1000)), np.resize(beta*(w_id*i + b_id) + (1-beta)/(1.0+np.exp(w_exp*i + b_exp)),(1000)))
	plt.show()
	print "DONE!!"
