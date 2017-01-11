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

#navigate to where you want to save the dataset in the terminal then type 'pwd' to print the working directory
datapath = '/afs/kth.se/home/m/a/maxlotz/Desktop/DD2434_ML_Adv/Project'

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
    return (dataset - min(dataset)) / (max(dataset) - min(dataset))

# modified from http://sebastianraschka.com/Articles/2014_kernel_pca.html
def kpca(dataset, gamma, n_components=256):

    N, M = dataset.shape

    # Calculating the squared Euclidean distances for every pair of points
    sq_dists = pdist(dataset, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-mat_sq_dists/gamma) # possibly gamma*N instead of just gamma

    # Centering the symmetric NxN kernel matrix.
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding eigenvectors
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc


dataset = fetch_mldata('usps', data_home=datapath)  # Save dataset at path (long download)
X, y = dataset.data, dataset.target.astype(np.int)

vecs = kpca(X[:300,:], 0.5)
print vecs.shape


