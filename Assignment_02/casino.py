import numpy as np
import matplotlib.pyplot as plt

def GenerateTables(K, initProb):
	sequence = []
	s = np.random.binomial(1, initProb, 1)
	sequence.append(s[0])
	for k in range(K-1):
		if sequence[len(sequence)-1] == 0:
			s = np.random.binomial(1,3.0/4.0,1)
		else:
			s = np.random.binomial(1,1.0/4.0,1)
		sequence.append(s[0])
	return sequence

def SampleTableDice(primed, unPrimed, tables):
	sequence = []
	for k in range(len(tables)):
		if tables[k] == 0:
			s = np.random.multinomial(1, unPrimed[k,:])
		else:
			s = np.random.multinomial(1, primed[k,:])
		outcome = np.argmax(s) + 1
		sequence.append(outcome)
	return sequence

def SamplePlayerDice(playerDice, tables):
	sequence = []
	for k in range(len(tables)):
		s = np.random.multinomial(1, playerDice)
		outcome = np.argmax(s) + 1
		sequence.append(outcome)
	return sequence

def Demo(nTables, primed, unPrimed, playerDice):
	tables = GenerateTables(nTables, 0.5)
	table_outcome = np.asarray(SampleTableDice(primed, unPrimed, tables))
	player_outcome = np.asarray(SamplePlayerDice(playerDice, tables))
	sum_ = table_outcome + player_outcome
	return sum_

nTables = 10
pi = [0.5, 0.5]

#unbiased dice
primed = np.ones((nTables,6))/6.0
unPrimed = np.ones((nTables,6))/6.0
playerDice = np.ones(6)/6.0

observations = Demo(nTables, primed, unPrimed, playerDice)

test = GenerateTables(1000,0.5)
print test