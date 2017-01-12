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

# modified from http://sebastianraschka.com/Articles/2014_kernel_pca.html
def kPCA(dataset, C, n_components=256):
    '''
    inputs:
        dataset: the dataset, shape (M x N). Each data point is a row, each dimension is a column.
        C: constant, 2 * (standard deviation**2)
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
    K = exp(-mat_sq_dists/C) # possibly C*N instead of just C

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding eigenvectors
    # This is already normalized
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    eigvecs = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    Y = np.dot(K,eigvecs)

    return Y, eigvecs

def PCA(dataset, n_components=256):
    dataset = normalize(dataset)
    C = np.cov(dataset, rowvar=False)
    eigvals, eigvecs = eigh(C)
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc # shape: 256 x n_components

def kPCA_PreImage(y,eigVector,dataset,C):
    # Finds preimage for one piece of data
    iters = 1000;
    N = dataset.shape[0]
    d = np.max(y.shape);

    gamma = np.dot(eigvecs,y)
    gamma = gamma[:,None].T

    z = np.mean(dataset,0)

    for count in range(iters):
        pre_z = z
        xx = -((z - dataset)**2)/C
        xx = np.exp(xx)
        num = np.dot(gamma,xx)
        den = xx*dataset
        z = np.sum(num,0)/np.sum(den,0)
        convergence = np.linalg.norm(pre_z - z)/np.linalg.norm(z)
        print convergence

    return z

dataset = fetch_mldata('usps', data_home=datapath)  # Save dataset at path (19.1Mb)
X, y = dataset.data, dataset.target.astype(np.int)
C = 1.0

X = X[:300,:]
Y, eigvecs = kPCA(X,0.5)
Y_ = Y[0,:]

z = kPCA_PreImage(Y_,eigvecs,X,C)
