import numpy as np
import matplotlib.pyplot as plt
from  math  import  pi
from  scipy.spatial.distance  import  cdist

'''
# To  sample  from a multivariate  Gaussian
f = np.random.multivariate_normal(mu ,K);

# To  compute a distance  matrix  between  two  sets of  vectors
D = cdist(x1 ,x2)

# To  compute  the  exponetial  of all  elements  in a matrix
E = np.exp(D)

plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
'''

X = np.linspace(-1,1,201)
X = np.reshape(X,[-1,1])
N = X.shape[0]

E = s = np.random.normal(0, 0.3, N)
E = np.reshape(E,[-1,1])

W = np.matrix([-1.3,0.5])

Y = X*W[0,0] + W[0,1] + E