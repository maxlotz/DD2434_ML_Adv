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
    #print tables
    tableOutcome = np.asarray(sampleTableDice(primed, unPrimed, tables))
    playerOutcome = np.asarray(samplePlayerDice(playerDice, tables))
    sum_ = tableOutcome + playerOutcome
    return sum_, tables

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
	    obs, _ = demo(nTables, primed, unPrimed, playerDice)
	    hist = np.append(hist, obs)

	H = np.diff(np.unique(hist)).min()
	left_of_first_bin = hist.min() - float(H)/2 
	right_of_last_bin = hist.max() + float(H)/2 
	plt.hist(hist, np.arange(left_of_first_bin, right_of_last_bin + H, H))
	plt.show()

def Viterbi(A,B,PI,O):
	T = O.shape[0]
	N = A.shape[0]
	M = B.shape[1]

	delta = np.zeros((T,N))
	deltaidx = np.zeros((T-1,N))
	maxO = np.zeros(T)

	delta[0,:] = np.log(PI*np.transpose(B[:,O[0]]))

	for t in range(T-1):
		for i in range(N):
			p1 = np.zeros(N)
			for j in range(N):
				p1[j] = delta[t,j] + np.log(A[j,i]) + np.log(B[i,O[t+1]])
			delta[t+1,i] = p1.max()
			deltaidx[t,i] = np.argmax(p1)

	maxO[T-1] = np.argmax(delta[T-1,:])

	for t in range(T-1):
		maxO[T-t-2] = deltaidx[T-t-2,maxO[T-t-1]]

	return maxO

def CreateMatrices(primed, unPrimed, playerDice):
	primed = primed[:,None]
	unPrimed = unPrimed[:,None]
	playerDice = playerDice[:,None]

	PI =  np.ones((1,2))/2.0

	A = np.matrix([[1./4,3./4],[3./4,1./4]])

	B = np.zeros((2,11))
	m1 = np.fliplr(np.dot(playerDice,np.transpose(unPrimed)))
	m2 = np.fliplr(np.dot(playerDice,np.transpose(primed)))
	for i in range(11):
		B[0,i] = sum(np.diag(m1,5-i)) 
		B[1,i] = sum(np.diag(m2,5-i))

	return A, B, PI

def PrintViterbi():
	Primed = np.ones(6)*1./10
	Primed[5] = 1./2
	PlayerDice = np.ones(6)*1.0/6
	UnPrimed = np.ones(6)*1.0/6

	nTables = 20
	A_, B_, PI_ = CreateMatrices(Primed, UnPrimed, PlayerDice)
	O_, tables = demo(nTables, Primed, UnPrimed, PlayerDice) # Observation is from 2 to 12, indexed as 0 to 10
	max_tables = Viterbi(A_,B_,PI_,O_-2)
	print O_
	print np.array(tables)
	print max_tables.astype(int)

PrintViterbi()