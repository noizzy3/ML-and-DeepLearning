import numpy as np 
import tensorflow as tf

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.2 +4

w = tf.Variable(tf.random_uniform([1],0.0,2.0))
b = tf.Variable(tf.zeros([1]))
y = x_data*w + b

loss = tf.reduce_mean((y - y_data)**2)
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for k in range(200):
	sess.run(train)
	if k%20 == 0:
		print (k, sess.run(w), sess.run(b))