#t=mlp.network([784,30,10],30,10,3.0)

import random
import cPickle
import gzip
import numpy as np

class network(object):
	def __init__(self,sizes,epochs,minibatch,eta):
		
		f=gzip.open('mnist.pkl.gz','rb')
		trd,vd,td=cPickle.load(f)
		f.close()
		tri=[np.reshape(x,(784,1)) for x in trd[0]]
		trv=[self.vectorise(x) for x in trd[1]]
		training_data=zip(tri,trv)
		ti=[np.reshape(x,(784,1)) for x in td[0]]
		test_data=zip(ti,td[1])

		self.size=sizes
		self.w=[np.random.randn(y,x) for y,x in zip(sizes[1:], sizes[:-1])]
		self.b=[np.random.randn(x,1) for x in self.size[1:]]
		for outer in range(epochs):
			random.shuffle(training_data)#list is shuffled
			for inner in range(0,len(training_data),minibatch)	:
				nabla_w=[np.zeros((y,x)) for y,x in zip(sizes[1:], sizes[:-1])]
				nabla_b=[np.zeros((x,1)) for x in self.size[1:]]
				for k in range(inner,inner+minibatch):
					a=training_data[k][0]
					val=training_data[k][1]# 10 X 1 array
					activation=[a]#list containing final activation
					for i in range(len(self.size)-1):
						a=self.sigmoid(i,a)
						activation.append(a)

					dw,db=self.backprop(activation,val)
					nabla_w=[x+y for x,y in zip(nabla_w,dw)]
					nabla_b=[x+y for x,y in zip(nabla_b,db)]

				"The below update takes place for each minibatch "
					
				self.w=[x-(eta/minibatch)*y for x,y in zip(self.w,nabla_w)]
				self.b=[x-(eta/minibatch)*y for x,y in zip(self.b,nabla_b)]
			
			self.test(test_data,outer+1)
	

	def sigmoid(self,i,act):
		z = np.dot(self.w[i],act) + self.b[i]
		return 1/(1+np.exp(-z))


	def backprop(self,activation,val): #For one training data
		db=[np.zeros(b.shape) for b in self.b]
		dw=[np.zeros(w.shape) for w in self.w]
		c=activation[-1]-val
		for i in range(len(self.size)-2,-1,-1):
			c=c*activation[i+1]*(1-activation[i+1])
			db[i]=c
			dw[i]=np.dot(c,activation[i].transpose())
			c=np.dot(self.w[i].transpose(),c)

		return (dw,db)


	def test(self,t,e):
		c=0
		for i,j in t:
			ar=i
			for k in range(len(self.size)-1):
				ar=self.sigmoid(k,ar)
			c=c+int(np.argmax(ar)==j)
		print "epoch: %d    accuracy: %f percent" %(e,c/100.0)


	def vectorise(self,y):
		z=np.zeros((10,1))
		z[y]=1
		return z

#network([784,30,10],30,10,3.0)
