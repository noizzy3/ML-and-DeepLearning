import tensorflow as tf 

a = tf.Variable(1) #declaring this as tf.constant is not going to work! think why?
b = tf.Variable(0)
ans = tf.add(a,b)
d = tf.assign(b,ans)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init) #if we put this line inside the for loop only 2 is printed
	for k in range(5):
		sess.run(d)
		print (sess.run(ans))