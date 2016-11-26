import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from random import randint

# set up global variables, these dont change
W = np.matrix([[-1.3],[0.5]])	# W = D x 1   
noise_sigma = 0.3

X = np.arange(-1,1.01,0.01)
X = np.reshape(X,(-1,1))
Xlen = X.shape[0]
X1 = np.ones(Xlen)
X1 = np.reshape(X1,(-1,1))
X = np.concatenate((X,X1),1) # X = N x D  

noisevec = np.random.normal(0,np.sqrt(noise_sigma),Xlen)
noisevec = np.reshape(noisevec,(-1,1))

Y = np.dot(X,W) + noisevec # Y = N x 1

# create the grid
W_graph = np.arange(-2,2,0.05)
xv, yv = np.meshgrid(W_graph, W_graph)
xv = np.reshape(xv,(np.prod(xv.shape),-1))
yv = np.reshape(yv,(np.prod(yv.shape),-1))
grid = np.concatenate((xv,yv),1)

# initial parameters for first run
W_tau = 0.3
W_mu = np.matrix([0,0])
W_mu = np.reshape(W_mu,(-1,1))
W_cov = np.diag([W_tau,W_tau])

def gauss_print(mu,cov):
	# grid and W_graph are global
    gauss = stats.multivariate_normal.pdf(grid, mu, cov)
    gauss = np.reshape(gauss,(len(W_graph),-1))
    plt.imshow(gauss,origin="lower",aspect='auto',extent=(xv.min(),xv.max(),yv.min(),yv.max()))
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.show()

def compute_posterior(X, Y, prior_mu, prior_cov):
    # X = 1 x D     Y = 1 x 1	prior_mu = D x 1	prior_cov = D x D
    # noise_sigma is global
    cov1 = np.dot(np.transpose(X),X)/noise_sigma
    cov2 = np.linalg.inv(prior_cov)
    covar = np.linalg.inv(cov1 + cov2)
    
    mu1 = np.dot(np.transpose(X),Y)/noise_sigma
    mu2 = np.dot(np.linalg.inv(prior_cov),prior_mu)
    mu = np.dot(covar,(mu1+mu2))
    
    return [mu, covar]

mu = W_mu
covar = W_cov
idx = []
for i in range(100):
	num = randint(0,Xlen-1)
	X1 = X[num,:]
	X1 = np.reshape(X1,(1,-1))
	Y1 = Y[num]
	Y1 = np.reshape(Y1,(-1,1))
	mu, covar = compute_posterior(X1, Y1, mu, covar)
	idx.append(num)

print mu
print covar

Y_nonoise = np.dot(X,mu)

plt.plot(X[idx,0], Y[idx,0], 'ro')
plt.plot(X[idx,0], Y_nonoise[idx,0])
plt.axis([-1.0, 1.0, -3, 4])
plt.figure()

gauss_print([mu[0,0],mu[1,0]], covar)
plt.show()
