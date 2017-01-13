import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot
import matplotlib as mpl
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import KernelCenterer
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import sys

# Author: Sebastian Leborg

datapath = 'datasets/'

def show(image):  # plots an image
	if len(image) == 256:
		image.shape = (16, 16)  # reshape 'cause dataset is (256, 1)
	fig = pyplot.figure()
	ax = fig.add_subplot(1, 1, 1)
	imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
	imgplot.set_interpolation('nearest')
	ax.xaxis.set_ticks_position('top')
	ax.yaxis.set_ticks_position('left')
	pyplot.show()


def gaussian_noise(dataset, prob):  # adds gaussian noise with probability prob
	d = np.array([[0.0 for x in range(len(dataset[0]))] for y in range(len(dataset))])
	for image in range(len(dataset)):
		for pixel in range(len(dataset[image])):
			if np.random.random() < prob:
				d[image][pixel] = np.random.normal(0, 0.5) + dataset[image][pixel]  # gaussian noise sigma = 0.5
			else:
				d[image][pixel] = dataset[image][pixel]
	return d


def speckle_noise(dataset, prob):  # dno if this is right, looks ok
	d = np.array([[0.0 for x in range(len(dataset[0]))] for y in range(len(dataset))])
	for image in range(len(dataset)):
		for pixel in range(len(dataset[image])):
			if np.random.random() < prob:
				d[image][pixel] = -dataset[image][pixel]
			else:
				d[image][pixel] = dataset[image][pixel]
	return d


# Takes single normalized image
def random_noise(image, prob):
	d = []
	for x in range(len(image)):
		if np.random.random() < prob:
			d.append(np.random.random())
		else:
			d.append(image[x])

	return d

def normalize(dataset):  # normalizes data 0 -> 1
	return dataset - np.mean(dataset,0)

def PCA(dataset, n_components=20):
	dataset = normalize(dataset)
	C = np.cov(dataset, rowvar=False)
	eigvals, eigvecs = eigh(C)
	X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

	return X_pc # shape: 256 x n_components

# modified from http://sebastianraschka.com/Articles/2014_kernel_pca.html
def kPCA(dataset, c, n_components=20):
	''' Performs kPCA
	inputs:
		dataset: the dataset, shape (l x D). Each data point is a row, each dimension is a column.
		c: constant, 2 * (standard deviation**2)
		n_components: number of PCA eigenvectors to be used to reconstruct data
	outputs:
		Y = data projected onto normalized eigenvectors
	'''

	dataset = normalize(dataset)

	# Calculating the squared Euclidean distances for every pair of points
	sq_dists = pdist(dataset, 'sqeuclidean')

	# Converting the pairwise distances into a symmetric MxM matrix.
	mat_sq_dists = squareform(sq_dists)

	# Computing the MxM kernel matrix.
	K = exp(-mat_sq_dists/c)

	# Centering the symmetric NxN kernel matrix.
	N = K.shape[0]
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	# Obtaining eigenvalues in descending order with corresponding eigenvectors
	lambdas, alphas = eigh(K)

	# normalization so that lambda_k*alpha_k*alpha_k = 1 (where alpha_k is an eigenvector in alphas)
	for k in range(N):
		val = lambdas[k] * np.dot(alphas[:,k], alphas[:,k])
		alphas[:,k] /= np.sqrt(val)

	# Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
	alphas = np.column_stack((alphas[:,-i] for i in range(1,n_components+1)))

	return alphas

def projection(dataset, datapoint, c, alphas):
	''' Projects as SINGLE datapoint onto given eigenvectors
	inputs:
		dataset: the dataset, shape (l x n). Each data point is a row, each dimension is a column.
		datapoint: the test point we wish to project on our eigenvectors
		c: constant, 2 * (standard deviation**2)
		alphas: Our matrix of eigenvectors, shape (l x n), where the top d eigenvectors have been chosen.
	outputs:
		Beta = data projected onto eigenvectors (n x 1)
	'''
	l, n = alphas.shape
	
	# Kernel for the new point
	K = np.exp(-np.sum((datapoint - dataset)**2,1)/c)
	betas = np.dot(alphas.T,K)

	#Reshape for output format.
	betas.shape = (n,1)

	return betas

def kPCA_PreImage(beta,alphas,dataset,c, z_start):
	''' Finds preimage for ONE single data points
	inputs:
		beta: a single column vector data point projected onto the eigenvectors obtained from kPCA (n x 1)
		alphas: Matrix containing the eigenvectors obtained from kPCA (l x n)
		dataset: the dataset, shape (l x D). Each data point is a row, each dimension is a column.
		c: constant, 2 * (standard deviation**2)
	outputs:
		z = preimage, shape (256,)
	'''
	iters = 1000
		
	num = 0
	den = 0
	z = z_start
	gamma = np.dot(alphas,beta) # shape (l x 1)
	N = gamma.shape[0]

	for it in range(iters):
		pre_z = z
		for i in range(N):
			val = gamma[0]*np.exp(-sum((z - dataset[0,:])**2)/c)
			den += val
			num += val*dataset[0,:]
		z = num/den
		convergence = np.linalg.norm(pre_z - z)/np.linalg.norm(z)
		print "convergence: " + str(convergence)
		if convergence<0.000000001:
			break
	
	return z
	


# GET DATA, CREATE TRAIN AND TEST SETS
dataset = fetch_mldata('usps', data_home=datapath)  # Save dataset at path (19.1Mb)
X, y = dataset.data, dataset.target.astype(np.int)
y -= 1
X = X[y == 5]
X = X[:700]
y = y[:700]
N = X.shape[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
X_test = speckle_noise(X_test, 0.1)

# SET PARAMS HERE
prob = 0.3 # probability of noise generated
n_comps = 1 # the number of components
C = 0.5 # gaussian kernel param, largely affects result
C = N*C/10
n = 1 # pick a specific datapoint

# ALGORITHM RUNS HERE. Performs kPCA, gets the eigenvalues and the projected data, then computes the pre-image for the chosen data point.
test_point = X_test[n,:]
Alphas = kPCA(X_train,C,n_comps)
Beta = projection(X_train, test_point, C, Alphas)
z = kPCA_PreImage(Beta, Alphas, X_train, C, test_point)

# Show results
show(test_point)
show(z)


'''
# USING SK-LEARN KPCA
testpoint = X_test[n,:]
testpoint = testpoint[None,:]

kpca = KernelPCA(n_components = 10, kernel="rbf", fit_inverse_transform=True, gamma=C) # Class containg info about the type of kernel, functions etc.
kpca.fit(X_train) # Creates the kernel based on the training data
Y = kpca.transform(testpoint) # projects a given point based on the eigenvectors
image = kpca.inverse_transform(Y) # computes the pre-image of the given point

testpoint.shape = (256,)
image.shape = (256,)

show(testpoint)
show(image)
'''