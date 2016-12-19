import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from scipy.special import gamma as Gamma

# True Distribution Parameters
def calc_true(Data, A_0, B_0, Mu_0, Kappa_0):
	xbar = np.mean(Data)
	N = data.shape[0]
	A_n = A_0 + N/2
	Kappa_n = Kappa_0 + N
	Mu_n = (Kappa_0*Mu_0 + N*xbar)/(Kappa_0 + N)
	B_n = B_0 + 1.0/2*sum((Data - xbar)**2) + 1.0/2*Kappa_0*N*(xbar - Mu_0)**2/(Kappa_0 + N)

	return A_n, Kappa_n, Mu_n, B_n

def normalgammapdf(mu, Lambda, muprior, kappa, alpha, beta):
	C = (np.sqrt(kappa)*beta**alpha)/(Gamma(alpha)*np.sqrt(2*np.pi))
	lam = C*np.power(Lambda,(alpha-0.5))*np.exp(-beta*Lambda)
	diff = (mu - muprior)**2
	p = lam*np.exp(-kappa/2*(np.dot(Lambda,np.transpose(diff))))
	return p

def calc_VB(Data, A_n, B_n, Mu_n, Kappa_n):
	xbar = np.mean(Data)
	s = sum(Data)
	sSq = sum(Data**2)
	iters = 1
	maxiters = 51

	while (iters < maxiters):
		# Update Mu
		elambda = A_n/B_n
		Mu_n = (kappa_0*mu_0 + N*xbar)/(kappa_0 + N)
		Kappa_n = (kappa_0 + N)*elambda
	    
	    # Update Lambda
		emu = Mu_n
		emuSquare = 1.0/Kappa_n + Mu_n**2
		A_n = a_0 + (N+1.0)/2.0
		B_n = b_0 + 0.5*((sSq + kappa_0*mu_0**2)-2*emu*(s + kappa_0 * mu_0) + emuSquare*(kappa_0+N))

		if (iters == 1 or iters == 2 or iters == 5):
			z = normalgammapdf(x[:,None], y[:,None], Mu_n, Kappa_n, A_n, B_n)
			Z = normalgammapdf(x[:,None], y[:,None], true_mu, true_kappa, true_a, true_b)

			CS = plt.contour(X, Y, z, colors='b',linestyles='dashed')
			plt.ylabel('lambda')
			plt.xlabel('mu')
			plt.title('VB inference ' + str(iters))
			
			CS2 = plt.contour(X, Y, Z, colors='g')
			plt.ylabel('lambda')
			plt.xlabel('mu')
			plt.title('True posterior vs. VB-infered ' + str(iters) + ' iterations')

			plt.show()

		iters+=1


# Generate Data
N = 10
mu = 0.0
var = 1.0
data = np.random.normal(mu, var, N)

# Creating meshgrid
delta = 0.05
x = np.arange(-2.0, 2.0, delta)
y = np.arange(0.0, 2.0, delta)
X, Y = np.meshgrid(x, y)

# Initial parameters
a_0, b_0, mu_0, kappa_0 = 0.0,0.0,0.0,0.0

# Intiial guess for VB
a_n = 0.5
b_n = 5
mu_n = 10.0
kappa_n = 20.0

# Find true parameters
true_a, true_kappa, true_mu, true_b = calc_true(data, a_0, b_0, mu_0, kappa_0)

calc_VB(data, a_n, b_n, mu_n, kappa_n)