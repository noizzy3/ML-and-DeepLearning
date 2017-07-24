import numpy as np 
import tensorflow as tf
import matplotlib.pyplot  as plt

x_data = [[0,1.41,1.85,
2.03,
2.37,
2.46,
2.56,
2.59,
2.65,
2.73,
2.9,
3.07,
3.14,
3.27,
3.48,
3.61,
3.83,
3.92,
4.03,
4.14,
4.18,
4.26,
4.4,
4.45,
4.59,
4.74,
4.81,
4.85,
4.9,
4.97,
5.11,
5.23,
5.27,
5.49,
5.83
]]
y_data = [[12.18,
12.18,
12.17,
12.07,
11.73,
11.58,
11.4,
11.32,
11.2,
11,
10.54,
10,
9.8,
9.3,
8.5,
7.95,
7,
6.55,
6.03,
5.45,
5.2,
4.8,
4.02,
3.71,
2.96,
2.18,
1.84,
1.7,
1.51,
1.37,
1.24,
1.15,
1.11,
1.01,
0.89

]]
n = len(x_data[0])

weights = tf.Variable(tf.random_normal([n,n]))
biases = tf.Variable(tf.ones([1,n]))
y = tf.add(tf.matmul(x_data,weights), biases)

loss = tf.reduce_mean((y - y_data)**2)
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for k in range(2500):
	sess.run(train)
	print (k, sess.run(loss))
	if k%50 == 0:
		y_ = sess.run(y)
		plt.plot(range(0,6,n),y_[0],'r')
		plt.axis([0,15,0,15])
		plt.show()
y_ = sess.run(y)
print y_
plt.plot(x_data[0],y_[0],'r')
plt.axis([0,15,0,15])
plt.show()
'''
xc = []
yc = []
for x in range(100):
	xc.append(x)
	yc.append(x*m + c)



import matplotlib.pyplot  as plt

x = [r for r in range(500)]
y = [p*2 +4 for p in x]

plt.plot(x,y,'r')
plt.axis([0,500,0,500])
plt.show()
'''