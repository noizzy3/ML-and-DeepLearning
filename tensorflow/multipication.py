import tensorflow as tf
import numpy as np 

a = tf.constant([[3,3]])
b = tf.constant([[3],[3]])
result = tf.matmul(a,b)

sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#            init is not required as the tensorflow variables are already initialised

print (sess.run(result))

sess.close()