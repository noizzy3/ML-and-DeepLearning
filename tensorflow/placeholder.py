import tensorflow as tf 

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.add(input1,input2)

with tf.Session() as sess:
	a = sess.run(output, feed_dict = {input1:[3.0], input2:[4.0]})
	print (a)