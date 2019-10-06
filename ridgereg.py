import numpy as np
import pandas as pd
from numpy.linalg import inv
import random
import matplotlib.pyplot as plt

# read the data file
df = pd.read_csv("data.csv", header=None)

#
# break the data file up into components as numpy arrays
#
#ftest = df.iloc[:,0].to_numpy()
#ftrain = df.iloc[:,1].to_numpy()
Xtest = df.iloc[:,2:21].to_numpy()
Xtrain = df.iloc[:,21:40].to_numpy()
ytest = df.iloc[:,40].to_numpy()
ytrain = df.iloc[:,41].to_numpy()

#
# append a 1 for a bias weight
# transpose so each xi is a column
Xtrain = np.append(Xtrain,np.ones((len(ytrain),1)),axis=1).T
Xtest = np.append(Xtest,np.ones((len(ytest),1)),axis=1).T

# calculate the trace because gamma
# is a scale of the trace
#trace = np.trace(K)

def train_calc_error(gamma, Xtrain, ytrain, Xtest, ytest):
	#
	# calculate the K (kernel) matrix
	K = np.matmul(Xtrain.T,Xtrain)
	# calculate the dual space coefficients
	alpha = np.matmul(inv(K + gamma*np.eye(len(K))),ytrain)
	# calculate the weights
	w = np.matmul(Xtrain,alpha)
	# calculate the prediction
	yhat = np.matmul(w.T,Xtest)
	# calculate error
	e = ytest - yhat
	# calculate the error^2
	e2 = np.dot(e,e)
	# calculate the loss
	loss = e2/len(yhat) + gamma*np.dot(w,w)
	return e2, loss, yhat

# for a given gamma, X, and y,
# split data randomly 80/20 into train and test 
# calculate the average error, and loss of
# the trained machine on the test data
def calc_avg_error(gamma, X, y, iterations):
	#
	# split the training data into an 80/20
	# split between train and test
	idx=list(range(0,len(y)))
	split=len(idx)*8//10

	etotal = 0
	losstotal = 0
	for i in range(0,iterations):
		# shuffle the data
		random.shuffle(idx)
		# determine train and test samples
		train=idx[0:split]
		test=idx[split:]
		Xtrain = X[:,train]
		ytrain = y[train]
		Xtest = X[:,test]
		ytest = y[test]

		e, loss, yhat = train_calc_error(gamma, Xtrain, ytrain, Xtest, ytest)
		etotal = etotal + e
		losstotal = losstotal + loss

	return etotal/iterations, losstotal/iterations

#
#
# keep track of the best
# value for gamma
beste = 1e6;
gammas = []
losses = []
errors = []

iterations = 5000

# vary gamma from 10^-2 to 10^1
for p in np.linspace(-2,1,num=100):
	gamma = 10**p

	e2,loss = calc_avg_error(gamma, Xtrain, ytrain, iterations)

	# save values for plots
	gammas.append(gamma)
	errors.append(e2)
	losses.append(loss)

	# keep track of the best values
	if (e2 < beste):
		beste = e2
		bestgamma = gamma

# we know the best gamma, so build a machine from
# entire training set and test
e2, loss, yhat = train_calc_error(bestgamma, Xtrain, ytrain, Xtest, ytest)

plt.plot(gammas,errors,label="Error")
plt.xlabel("$\gamma$")
plt.ylabel("$e^2$")
plt.xscale(value="log")
plt.title("Average Error vs. $\gamma$ (%d iterations)"%(iterations))
plt.show()

time = list(range(0,len(yhat)))
plt.plot(time,ytest,label="$y_{test}$")
plt.plot(time,yhat,label="$\hat{y}$")
plt.xlabel("Time")
plt.legend()
plt.title("Real vs. Predicted ($\gamma=%.3f,e^2=%.2f$)"%(bestgamma,e2))
plt.show()
