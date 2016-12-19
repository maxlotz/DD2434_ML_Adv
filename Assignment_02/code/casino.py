import numpy as np
import matplotlib.pyplot as plt

#generates a sequence of length K of the variable Z (a sequence of hidden states)
# Z = 1 if the table is primed and 0 if the table is not primed
def generateTables(K, initProb):
    sequence =[]
    #sample first table
    s = np.random.binomial(1, initProb, 1)
    #sample k-1 tables
    sequence.append(s[0])
    for k in range(K-1):
        if sequence[len(sequence)-1] == 0:
            s = np.random.binomial(1,3.0/4,1)
        else:
            s = np.random.binomial(1,1.0/4,1)
        sequence.append(s[0])

    return sequence

#Samples from the outcome of the table's dice
#primed and unPrimed are the categorical distributions for K primed and K unprimed Tables
def sampleTableDice(primed, unPrimed, tables):
    sequence = []
    for k in range(len(tables)):
        if tables[k] == 0:
            s = np.random.multinomial(1, unPrimed)
        else:
            s = np.random.multinomial(1, primed)
        outcome = np.argmax(s)+ 1
        sequence.append(outcome)
    return(sequence)

#Samples from the outcome of the player's dice
def samplePlayerDice(playerDice, tables):
    sequence = []
    for k in range(len(tables)):
        s =  np.random.multinomial(1, playerDice)
        outcome = np.argmax(s)+ 1
        sequence.append(outcome)
    return(sequence)

def demo(nTables, primed, unPrimed, playerDice):

    tables = generateTables(nTables, 0.5)
    tableOutcome = np.asarray(sampleTableDice(primed, unPrimed, tables))
    playerOutcome = np.asarray(samplePlayerDice(playerDice, tables))
    sum = tableOutcome + playerOutcome
    return sum

def DiceHist():
	nTables = 10

	# set dice distributions here
	#unPrimed = np.ones(6)*1./10
	#unPrimed[5] = 1./2
	primed = np.ones(6)*1./10
	primed[5] = 1./2
	#playerDice = np.ones(6)*1./10
	#playerDice[5] = 1./2
	#primed = np.ones(6)*1.0/6
	playerDice = np.ones(6)*1.0/6
	unPrimed = np.ones(6)*1.0/6

	iters = 10000
	hist = []
	for i in range(iters):
	    obs = demo(nTables, primed, unPrimed, playerDice)
	    hist = np.append(hist, obs)

	H = np.diff(np.unique(hist)).min()
	left_of_first_bin = hist.min() - float(H)/2 
	right_of_last_bin = hist.max() + float(H)/2 
	plt.hist(hist, np.arange(left_of_first_bin, right_of_last_bin + H, H))
	plt.show()

# compute alpha given a sequence of observations and the parameters of the model
# primed: distribution of the dice in the primed tables
# unprimed: distribution of the dice in the unprimed tables
# playerDice : distribution of the player's dice
# A: transition matrix (for the states)
# pi: intial state distribution
def SampleFromPosterior(obs, primed, unPrimed, playerDice, pi, A):
    #number of states
    states = np.size(A[0,:])
    observations = len(obs)
    #table of alphas
    alpha = np.zeros((observations,states))
    #alpha(timestep, state)
    for i in range(states):
        alpha[0][i] = pi[i]*computeB(obs[0],0,i,primed, unPrimed,playerDice)

    for k in range(1,observations):
        for i in range(states):
            for j in range(states):
                alpha[k][i] += alpha[k-1][j]*A[j][i]*computeB(obs[k],k,i, primed, unPrimed, playerDice)
    norm = 0
    for i in range(states):
        norm += alpha[observations-1][i]
    prob1 = alpha[observations-1][1]/norm
    #prob0 = alpha[observations-1][0]/norm
    #print("prob 0",prob0)
    #print("prob 1", prob1)
    zk = np.random.binomial(1,prob1)
    stateSequence = []
    for i in reversed(range(observations-1)):
        zkPrev = []
        for previous in range(states):
            zkPrev.append(A[previous][zk]*alpha[i][previous])
        factor = 0
        for i in range(states):
            factor += zkPrev[i]
        probZkPrev = zkPrev[1]/factor
        stateSequence.append(np.random.binomial(1,probZkPrev))

    #in the order of the observations
    stateSequence = stateSequence[::-1]
    stateSequence.append(zk)
    return stateSequence


def CreateMatrices(primed, unPrimed, playerDice):
	primed = primed[:,None]
	unPrimed = unPrimed[:,None]
	playerDice = playerDice[:,None]

	PI =  np.ones((2))/2.0

	A = np.matrix([[1./4,3./4],[3./4,1./4]])

	B = np.zeros((2,11))
	m1 = np.fliplr(np.dot(playerDice,np.transpose(unPrimed)))
	m2 = np.fliplr(np.dot(playerDice,np.transpose(primed)))
	print m1
	print m2
	for i in range(11):
		B[0,i] = sum(np.diag(m1,5-i)) 
		B[1,i] = sum(np.diag(m2,5-i))

	return A, B, PI

primed = np.ones(6)*1./10
primed[5] = 1./2
playerDice = np.ones(6)*1.0/6
unPrimed = np.ones(6)*1.0/6