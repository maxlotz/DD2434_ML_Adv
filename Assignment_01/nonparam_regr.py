import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist, pdist

# Generate data
l = 1.5
sigma = 1.0
cardinality = 7
noise_sigma = 0.5

X = np.linspace(-1.5*np.pi, 1.5*np.pi, num=cardinality)
Xlen = X.shape[0]
Y = np.sin(X) + np.sqrt(noise_sigma)*np.random.randn(Xlen)
x_s = np.linspace(-2*np.pi, 2*np.pi, 500)
x_s_len = x_s.shape[0]
X = X[:,None]
Y = Y[:,None]
x_s = x_s[:,None]

def kernel(xi, xj, scale):
	return (sigma**2)*np.exp(-cdist(xi, xj, 'sqeuclidean')/(scale**2))

def compute_posterior(scale):
    k = kernel(x_s,X,scale)
    Cinv = np.linalg.inv(kernel(X,X,scale))
    mu = np.dot(np.dot(k,Cinv),Y)

    c = kernel(x_s, x_s,scale)
    sig = c - np.dot(np.dot(k,Cinv),np.transpose(k))
    return mu, sig

def plot_sample_prior(scale):
	samples = 5
	mu = np.zeros((x_s_len))
	K =  kernel(x_s,x_s,scale)
	Z = np.random.multivariate_normal(mu,K,samples)
	plt.figure()
	for i in range(samples):
		plt.plot(x_s[:],Z[i,:])
	title = 'length-scale '+str(scale)
	plt.title(title)
	plt.show()

def plot_samples_posterior(mu, sig):
	samples = 5
	mu = np.reshape(mu,(x_s_len,))
	Z = np.random.multivariate_normal(mu,np.nan_to_num(sig),samples)
	plt.figure()
	plt.plot(X,Y,'ro')
	for i in range(samples):
		plt.plot(x_s[:],Z[i,:])
	plt.plot(x_s[:],mu[:],color="black",linewidth=2)
	title = 'length-scale '+str(l)
	plt.title(title)
	plt.show()

def plot_posterior(mu, sig):
    plt.plot(X, Y,'ro')
    plt.plot(x_s,np.sin(x_s), color = 'green')
    plt.plot(x_s,mu, color = 'blue')
    mu = np.reshape(mu,(x_s_len,))
    xs = np.reshape(x_s,(x_s_len,))
    sig_vec = np.diag(sig)
    upper = mu + 2*sig_vec
    lower = mu - 2*sig_vec
    ax = plt.gca()
    ax.fill_between(xs, upper, lower, facecolor='pink', interpolate=True, alpha=0.1)
    title = 'length-scale '+str(l)
    plt.title(title)
    plt.show()

mu, sig = compute_posterior(l)

plot_posterior(mu,sig)