import numpy as np
import matplotlib.pyplot as plt
import math
import  scipy  as sp
import  scipy.optimize  as opt

sig = 1.0
#Wtest = np.random.randn(20)
X = np.linspace(0.0, 4.0*np.pi, 100)

x1 = X*np.cos(X)
x2 = X*np.sin(X)
x1 = x1[:,None]
x2 = x2[:,None]

Xnon = np.concatenate((x1,x2),1)
A = np.random.randn(10,2)
Y = np.dot(Xnon,np.transpose(A))

def L(w):
	w = np.reshape(w,(10,2))
	C = np.dot(w,np.transpose(w)) + sig*np.eye(10)
	inv = np.linalg.inv(C)
	p1 = np.dot(np.dot(Y,inv),np.transpose(Y))

	t1 = 0.5*100*10*np.log(np.pi*2)
	t2 = 0.5*100*np.log(np.linalg.det(C))
	t3 = 0.5*np.trace(p1)
	return (t1 + t2 + t3)

def dL(w):
	w = np.reshape(w,(10,2))
	C = np.dot(w,np.transpose(w)) + sig*np.eye(10)
	inv = np.linalg.inv(C)
	dLdW = np.zeros([10,2])
	for i in range(10):
		for j in range(2):
			dW = np.zeros([10,2])
			dW[i,j] = 1
			dWW = np.dot(dW,np.transpose(w)) + np.dot(w,np.transpose(dW))
			p1 = np.dot(inv,np.dot(dWW,inv))
			p2 = np.dot(Y,np.dot(p1,np.transpose(Y)))

			t1 = 0.5*100*np.trace(np.dot(inv,dWW))
			t2 = -0.5*np.trace(p2)
			dLdW[i,j] = t1 + t2

	dLdW = np.reshape(dLdW,(20,))
	return dLdW

W0 = np.random.randn(20)
W = opt.fmin_cg(L,W0,fprime=dL)
W = np.reshape(W,(10,2))

# Remember Y = X * A^T, but you cannot do X = Y * inv(A^T) because A is not a square matrix
# Instead Y = X * A^T ... Y * A = X * (A^T * A) ... ((A^T)A) can be inverted giving X = Y*A*inv((A^T)A)
WTW = np.dot(np.transpose(W),W)
x_pred = np.dot(Y,np.dot(W,np.linalg.inv(WTW)))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(Xnon[:,0],Xnon[:,1], 'ro')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x_pred[:,0],x_pred[:,1], 'bo')

plt.show()
