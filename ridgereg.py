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

iterations = 500
ngamma = 100

# values for plotting gamma vs. error, etc.
gammas = np.zeros((ngamma+1,))
losses = np.zeros((ngamma+1,))
errors = np.zeros((ngamma+1,))

# calculate ranges for 80/20 split
idx=list(range(0,len(ytrain)))
split=len(idx)*8//10

for i in range(0,iterations):
	# shuffle the data indexes
	random.shuffle(idx)
	# split train and test samples
	# training portion
	tr=idx[0:split]
	Xtr = Xtrain[:,tr]
	ytr = ytrain[tr]
	# testing portion
	te=idx[split:]
	Xte = Xtrain[:,te]
	yte = ytrain[te]
	#
	# calculate the K (kernel) matrix
	K = np.matmul(Xtr.T,Xtr)

	# vary gamma from 10^-2 to 10^1
	pmin = -2
	pmax = 1
	for j in range(0,ngamma+1):
		gamma = 10**(j*(pmax-pmin)/ngamma+pmin)
		gammas[j] = gamma

		# calculate the dual space coefficients
		alpha = np.matmul(inv(K + gamma*np.eye(len(K))),ytr)
		# calculate the weights
		w = np.matmul(Xtr,alpha)
		# calculate the prediction
		yhat = np.matmul(w.T,Xte)
		# calculate error
		e = yte - yhat
		# calculate the error^2
		e2 = np.dot(e,e)/len(yhat)
		# calculate the loss
		loss = e2 + gamma*np.dot(w,w)

		errors[j] = errors[j] + e2
		losses[j] = losses[j] + loss

# now find the lowest error/gamma
lowe = np.inf
errors = errors / iterations
losses = losses / iterations
for i in range(0,len(gammas)):
	if (errors[i] < lowe):
		lowe = errors[i]
		gamma = gammas[i]

# we know the best gamma, so build a machine from
# entire training set and test
# calculate the K (kernel) matrix
K = np.matmul(Xtrain.T,Xtrain)
# calculate coefficients
alpha = np.matmul(inv(K + gamma*np.eye(len(K))),ytrain)
w = np.matmul(Xtrain,alpha)
# now test it against the full test set
yhat = np.matmul(w.T,Xtest)
e = ytest - yhat
e2 = np.dot(e,e)/len(ytest)

plt.plot(gammas,errors,label="Error")
#plt.plot(gammas,losses,label="Loss")
plt.scatter([gamma],[lowe])
plt.xlabel("$\gamma$")
plt.ylabel("$e^2$")
plt.xscale(value="log")
#plt.legend()
plt.suptitle("Training Mean Square Error vs. $\gamma$")
plt.title("min at $\gamma=%.3f, e^2=%.3g$"%(gamma,lowe))
plt.show()

time = list(range(0,len(yhat)))
plt.plot(time,ytest,label="$y_{test}$")
plt.plot(time,yhat,label="$\hat{y}$")
plt.xlabel("Time")
plt.legend()
plt.suptitle("Real vs. Predicted")
plt.title("$\gamma=%.3f,e^2=%.3g$"%(gamma,e2))
plt.show()
