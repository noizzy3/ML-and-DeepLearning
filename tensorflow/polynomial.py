import tensorflow as tf 
import numpy as np 
import random

x_dat = range(1000)
#x_dat[1] = 1.001
x_data = [i/1000.0 for i in x_dat]
y_data = np.power(x_data,3) + np.power(x_data,2) + np.power(x_data,1) + 1#when exponent was 5 the  bias didn't decrease

class graph(): #predict and update
	def __init__(self):
		self.input = tf.placeholder(tf.float32,[100,1])
		self.target = tf.placeholder(tf.float32,[100,1])
		self.p = tf.Variable(tf.ones([1]))
		self.bias1 = tf.Variable(tf.random_normal([1]))
		self.bias2 = tf.Variable(tf.ones([1]))
		'''self.output = tf.cond((self.input - self.bias2)*(tf.pow(self.input,self.p) - self.bias1) < 0.0000001,\
		 lambda:tf.pow(self.input + self.bias,self.p) - self.bias1 , lambda: (tf.pow(self.input,self.p) - self.bias1)/(self.input - self.bias2))
		'''
		self.output = (tf.pow(self.input,self.p) - self.bias1)/(self.input - self.bias2)
		self.cost = tf.reduce_mean(tf.pow(self.output - self.target,2))
		self.loss = tf.train.AdamOptimizer(0.001).minimize(self.cost)

	def predict(self, input):
		print (sess.run(self.output, feed_dict = {self.input:input}))

	def update(self, input, target):
		sess.run(self.loss, feed_dict = {self.input:input, self.target:target})

g = graph()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#print g.predict(np.asarray(range(1, 101)).reshape(100,1))
	for epochs in range(300000):
		x = random.sample(x_data,100)
		y = [y_data[int(i*1000)] for i in x]
		g.update(np.asarray(x).reshape(100,1), np.asarray(y).reshape(100,1))
		if epochs%2000 == 0:
			print sess.run(g.cost, feed_dict = {g.input:np.asarray(x).reshape(100,1), g.target:np.asarray(y).reshape(100,1)})
			print sess.run(g.p)[0], sess.run(g.bias1)[0], sess.run(g.bias2)[0]
	#print g.predict(np.asarray(range(100)).reshape(100,1))