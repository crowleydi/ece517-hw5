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
Xval = df.iloc[:,2:21].to_numpy()
X = df.iloc[:,21:40].to_numpy()
yval = df.iloc[:,40].to_numpy()
y = df.iloc[:,41].to_numpy()

#
# append a 1 for a bias weight
# transpose so each xi is a column
X = np.append(X,np.ones((len(y),1)),axis=1).T
Xval = np.append(Xval,np.ones((len(yval),1)),axis=1).T

#
# split the training data into an 80/20
# split between train and test
idx = list(range(0,len(y)))
split=len(idx)*8//10

# shuffle the data
random.shuffle(idx)
# determine train and test samples
train=idx[0:split]
test=idx[split:]
Xtrain = X[:,train]
ytrain = y[train]
Xtest = X[:,test]
ytest = y[test]

#
# calculate the K (kernel) matrix
K = np.matmul(Xtrain.T,Xtrain)

# calculate the trace because gamma
# is a scale of the trace
trace = np.trace(K)

#
# keep track of the best
# value for gamma
bestloss = 1e6;
gammas = []
losses = []
errors = []

#
# vary gamma from 10^-3 to 10^3
for p in np.linspace(-3,1):
	gamma = trace*10**p
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

	# save values for plots
	losses.append(loss)
	gammas.append(gamma)
	errors.append(e2)

	# keep track of the best values
	if (e2 < bestloss):
		bestloss = e2
		bestgamma = gamma
		beste = e
		bestw = w

#plt.plot(gammas,losses,label="Loss")
plt.plot(gammas,errors,label="Error")
plt.xlabel("$\gamma$")
plt.xscale(value="log")
plt.legend()
plt.title("Error vs. $\gamma$")
plt.show()

yhat = np.matmul(bestw.T,Xval)
time = list(range(0,len(yhat)))
e = yval - yhat
e2 = np.dot(e,e)
plt.plot(time,yval,label="yval")
plt.plot(time,yhat,label="yhat")
plt.xlabel("Time")
plt.legend()
plt.title("Real vs. Predicted ($\gamma=%.3f$,e^2=%.2f)"%(bestgamma,e2))
plt.show()
